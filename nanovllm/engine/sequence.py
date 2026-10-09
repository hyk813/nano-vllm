from copy import copy
from enum import Enum, auto
from itertools import count

from nanovllm.sampling_params import SamplingParams


class SequenceStatus(Enum):
    # 调度器维护的请求生命周期：等待（含分段 prefill）、运行、完成。
    WAITING = auto()
    RUNNING = auto()
    FINISHED = auto()


class Sequence:
    # 类属性：引擎初始化时统一设置块大小；计数器在当前进程内生成递增请求 ID。
    block_size = 256
    counter = count()

    def __init__(self, token_ids: list[int], sampling_params = SamplingParams()):
        self.seq_id = next(Sequence.counter)
        self.status = SequenceStatus.WAITING
        # 主进程持有提示词和生成 token 的完整历史；浅复制避免追加时修改调用者的列表。
        self.token_ids = copy(token_ids)
        self.last_token = token_ids[-1]
        self.num_tokens = len(self.token_ids)
        self.num_prompt_tokens = len(token_ids)  # 固定边界，用于区分提示词与生成部分。
        # 已有 KV 的前缀长度与本轮待执行长度不同；新采样 token 尚未写入 KV Cache。
        self.num_cached_tokens = 0
        self.num_scheduled_tokens = 0
        # 决定进程间序列化时传完整历史还是仅最后一个 token；抢占重算时重新置 True。
        self.is_prefill = True
        # 按逻辑块顺序保存物理 KV 块 ID；这里只存映射，GPU K/V 张量由 ModelRunner 持有。
        self.block_table = []
        # 复制参数字段，调度与采样无需继续持有 SamplingParams 对象。
        self.temperature = sampling_params.temperature
        self.max_tokens = sampling_params.max_tokens
        self.ignore_eos = sampling_params.ignore_eos

    def __len__(self):
        return self.num_tokens

    def __getitem__(self, key):
        # 转发索引/切片，使 seq[i]、seq[start:end] 可用于输入准备和前缀哈希。
        return self.token_ids[key]

    @property
    def is_finished(self):
        return self.status == SequenceStatus.FINISHED

    @property
    def num_completion_tokens(self):
        return self.num_tokens - self.num_prompt_tokens

    @property
    def prompt_token_ids(self):
        return self.token_ids[:self.num_prompt_tokens]

    @property
    def completion_token_ids(self):
        return self.token_ids[self.num_prompt_tokens:]

    @property
    def num_blocks(self):
        # 整数向上取整：尾部不足一个完整块也计一块；此处不分配实际 KV 块。
        return (self.num_tokens + self.block_size - 1) // self.block_size

    @property
    def last_block_num_tokens(self):
        # 尾块已包含的 token 数；恰好整除时为 block_size，不能简单使用取余。
        return self.num_tokens - (self.num_blocks - 1) * self.block_size

    def block(self, i):
        # 返回第 i 个逻辑块的 token 内容，供 BlockManager 计算/核对前缀哈希。
        assert 0 <= i < self.num_blocks
        return self.token_ids[i*self.block_size: (i+1)*self.block_size]

    def append_token(self, token_id: int):
        # 只更新逻辑 token 历史；新 token 的 K/V 要到下一轮模型执行才产生。
        self.token_ids.append(token_id)
        self.last_token = token_id
        self.num_tokens += 1

    def __getstate__(self):
        # pickle 序列化钩子：rank 0 通过共享内存发送执行所需的最小状态。
        # prefill 可能重算/处理多 token，必须传完整历史；decode 只需最后一个 token。
        last_state = self.last_token if not self.is_prefill else self.token_ids
        return (self.num_tokens, self.num_prompt_tokens, self.num_cached_tokens, self.num_scheduled_tokens, self.block_table, last_state)

    def __setstate__(self, state):
        # 不恢复 seq_id、status、采样参数等主进程字段，因此不是完整的可调度请求。
        self.num_tokens, self.num_prompt_tokens, self.num_cached_tokens, self.num_scheduled_tokens, self.block_table, last_state = state
        if isinstance(last_state, list):
            self.token_ids = last_state
            self.last_token = self.token_ids[-1]
        else:
            # decode 不传历史，逻辑长度仍由 num_tokens 保存；工作 rank 只用 last_token。
            self.token_ids = []
            self.last_token = last_state
