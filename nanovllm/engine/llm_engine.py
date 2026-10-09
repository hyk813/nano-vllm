import atexit
from dataclasses import fields
from time import perf_counter
from tqdm.auto import tqdm
from transformers import AutoTokenizer
import torch.multiprocessing as mp

from nanovllm.config import Config
from nanovllm.sampling_params import SamplingParams
from nanovllm.engine.sequence import Sequence # 保存一条请求的 token、生成参数、状态和 KV Cache 块表
from nanovllm.engine.scheduler import Scheduler # 决定本轮处理哪些请求，以及它们做 prefill 还是 decode
from nanovllm.engine.model_runner import ModelRunner


class LLMEngine:
    # 引擎负责串联请求、调度和模型执行；模型计算由 ModelRunner 完成。

    def __init__(self, model, **kwargs):
        # 集合/字典推导式只保留合法配置项
        config_fields = {field.name for field in fields(Config)}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        config = Config(model, **config_kwargs)
        # 修改类属性，让所有 Sequence 的逻辑块大小与实际 KV Cache 一致。
        Sequence.block_size = config.kvcache_block_size
        self.ps = [] # 启动的子进程对象
        self.events = [] # 通知这些子进程的事件对象
        ctx = mp.get_context("spawn")
        # `spawn` 是进程启动方式：新进程启动自己的 Python 解释器，再运行指定目标。对于 CUDA 多进程，这比直接继承已有 CUDA 状态的 `fork` 更合适。
        # rank 0 在主进程运行；这里只启动 rank 1 到 N-1。N=1 时不进入循环。
        for i in range(1, config.tensor_parallel_size):
            # Event 用于通知工作进程；target/args 指定其运行 ModelRunner(config, i, event)。
            event = ctx.Event()
            process = ctx.Process(target=ModelRunner, args=(config, i, event))
            process.start()
            self.ps.append(process)
            self.events.append(event)
        # 初始化包含加载权重、预热、估算/分配 KV Cache，以及可选的 CUDA Graph 捕获。
        # rank 0 接收事件列表，通过共享内存和事件向其他 rank 分发执行命令。
        self.model_runner = ModelRunner(config, 0, self.events)
        self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)
        config.eos = self.tokenizer.eos_token_id
        # 此时 KV 块数和 EOS ID 已确定，调度器才能据此初始化。
        self.scheduler = Scheduler(config)
        atexit.register(self.exit)

    def exit(self):
        # call 会让各 rank 清理执行资源；join 等待子进程退出。
        # del 删除属性引用；此方法没有重复调用保护。
        self.model_runner.call("exit")
        del self.model_runner
        for p in self.ps:
            p.join()

    def add_request(self, prompt: str | list[int], sampling_params: SamplingParams):
        # 类型注解允许文本或 token ID 列表，但不会自动验证输入。
        if isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt)
        seq = Sequence(prompt, sampling_params)
        # 这里只加入 waiting 队列，尚未执行模型。
        self.scheduler.add(seq)

    def step(self):
        # 拆包得到本轮批次和阶段：prefill 处理上下文，decode 每条序列处理一个 token。
        seqs, is_prefill = self.scheduler.schedule()
        # 正数记录 prefill token 数，负数记录 decode token 数，供 generate 区分吞吐指标。
        num_tokens = sum(seq.num_scheduled_tokens for seq in seqs) if is_prefill else -len(seqs)
        # 执行器组织张量、运行模型并抽样；返回与 seqs 一一对应的新 token ID。
        token_ids = self.model_runner.call("run", seqs, is_prefill)
        # 更新缓存和请求状态；完整 prefill/decode 后追加 token，结束时释放 KV 块。
        # 分段 prefill 尚未处理完上下文时，本轮抽样结果会被忽略。
        self.scheduler.postprocess(seqs, token_ids, is_prefill)
        # 只返回本轮已完成请求的全部生成 token，因此这里不是逐 token 流式输出。
        outputs = [(seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished]
        return outputs, num_tokens

    def is_finished(self):
        # waiting 和 running 队列均为空才表示整个引擎没有未完成请求。
        return self.scheduler.is_finished()

    def generate(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
        use_tqdm: bool = True,
    ) -> list[str]:
        # 实际返回 list[dict]（text 与 token_ids），此处 list[str] 注解与实现不一致。
        # 进度条按完成请求数推进；use_tqdm=False 只关闭显示。
        pbar = tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True, disable=not use_tqdm)
        if not isinstance(sampling_params, list):
            # 列表乘法重复的是同一参数对象的引用，Sequence 随后读取它的各个字段。
            sampling_params = [sampling_params] * len(prompts)
        # zip 将提示词与参数配对；长度不一致时按较短列表截断，这里没有长度检查。
        for prompt, sp in zip(prompts, sampling_params):
            self.add_request(prompt, sp)
        outputs = {}
        prefill_throughput = decode_throughput = 0.
        while not self.is_finished():
            # 计时包含调度、输入准备、模型执行、采样与后处理，并非纯 GPU 运算时间。
            t = perf_counter()
            output, num_tokens = self.step()
            # 显示最近一轮对应阶段的吞吐；另一阶段保留旧值，不是累计平均吞吐。
            if num_tokens > 0:
                prefill_throughput = num_tokens / (perf_counter() - t)
            else:
                decode_throughput = -num_tokens / (perf_counter() - t)
            pbar.set_postfix({
                "Prefill": f"{int(prefill_throughput)}tok/s",
                "Decode": f"{int(decode_throughput)}tok/s",
            })
            for seq_id, token_ids in output:
                outputs[seq_id] = token_ids
                pbar.update(1)
        pbar.close()
        # 请求完成顺序可能不同；seq_id 按创建顺序递增，排序恢复提交顺序。
        outputs = [outputs[seq_id] for seq_id in sorted(outputs.keys())]
        # completion_token_ids 只含生成部分；解码为文本，同时保留原始 token ID。
        outputs = [{"text": self.tokenizer.decode(token_ids), "token_ids": token_ids} for token_ids in outputs]
        return outputs
