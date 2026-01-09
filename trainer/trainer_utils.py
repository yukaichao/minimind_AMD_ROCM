"""
训练工具函数集合（trainer_utils.py）

变更记录（真实迭代）：
v1 -> v2（为 AMD/Windows 的裁剪版 torch.distributed 做兼容）
1) torch.distributed 改为可选能力：增加 dist_is_initialized / dist_get_rank / dist_get_world_size / dist_destroy
2) init_distributed_mode 增加 try/except，backend 优先尝试 nccl，失败则回退 gloo，仍失败则退化单进程
3) checkpoint 保存改为 .tmp + os.replace 原子写入，避免半截文件
4) resume 读取增加 try/except：遇到 zip 损坏（failed finding central directory）自动改名为 .bad 并跳过
5) is_main_process / Logger 统一基于 dist_get_rank，避免 dist 缺 API 时再次报错
"""

import os
import sys

__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import random
import math
import numpy as np
import torch

# --------------------------
# 1) 分布式：可选能力封装（兼容某些 torch 构建缺失 dist.is_initialized 等 API）
# --------------------------
try:
    import torch.distributed as dist  # 可能存在，但 API 不一定齐全
except Exception:
    dist = None


def dist_is_initialized() -> bool:
    """
    判断分布式进程组是否初始化。
    - dist 不存在 或 缺 is_initialized：返回 False
    - 否则调用 dist.is_initialized()
    """
    return (dist is not None) and hasattr(dist, "is_initialized") and dist.is_initialized()


def dist_get_rank() -> int:
    """
    获取当前 rank。
    - 分布式已初始化且具备 get_rank：返回 dist.get_rank()
    - 否则（单进程）：返回 0
    """
    if dist_is_initialized() and hasattr(dist, "get_rank"):
        return dist.get_rank()
    return 0


def dist_get_world_size() -> int:
    """
    获取 world_size（进程总数）。
    - 分布式已初始化且具备 get_world_size：返回 dist.get_world_size()
    - 否则（单进程）：返回 1
    """
    if dist_is_initialized() and hasattr(dist, "get_world_size"):
        return dist.get_world_size()
    return 1


def dist_destroy():
    """
    销毁进程组（训练结束清理）。
    - 分布式已初始化且具备 destroy_process_group：执行销毁
    - 否则忽略
    """
    if dist_is_initialized() and hasattr(dist, "destroy_process_group"):
        dist.destroy_process_group()


from torch.utils.data import Sampler
from transformers import AutoTokenizer
from model.model_minimind import MiniMindForCausalLM


def Logger(content: str):
    """只在主进程打印日志，避免多进程重复输出。"""
    if is_main_process():
        print(content)


def is_main_process() -> bool:
    """主进程判定：分布式时 rank==0；单进程恒 True。"""
    return dist_get_rank() == 0


def get_lr(current_step: int, total_steps: int, lr: float) -> float:
    """
    余弦退火学习率（与原项目保持一致）。
    """
    return lr * (0.1 + 0.45 * (1 + math.cos(math.pi * current_step / total_steps)))


def setup_seed(seed: int):
    """
    设定随机种子，保证可复现性。
    注意：Windows + 多 worker DataLoader 时可复现性仍可能受进程调度影响。
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def init_distributed_mode() -> int:
    """
    初始化 DDP（分布式数据并行）。
    约定：若未设置环境变量 RANK，则视为非 DDP，返回 local_rank=0。

    兼容策略：
    - 若 dist 不可用 / 缺 init_process_group：直接退化为单进程
    - backend 先尝试 nccl（ROCm/RCCL 在很多环境下沿用该名称）
    - 若失败，再尝试 gloo（更通用，但 GPU 通信能力依赖实现）
    - 仍失败：退化单进程
    """
    if int(os.environ.get("RANK", -1)) == -1:
        return 0  # 非 DDP 模式

    if dist is None or (not hasattr(dist, "init_process_group")):
        Logger("[WARN] torch.distributed 不可用或缺少 init_process_group，已退化为单进程。")
        return 0

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    # 初始化进程组：先 nccl 再 gloo
    last_err = None
    for backend in ("nccl", "gloo"):
        try:
            dist.init_process_group(backend=backend)
            if torch.cuda.is_available():
                torch.cuda.set_device(local_rank)
            Logger(f"[INFO] DDP 初始化成功，backend={backend}, local_rank={local_rank}, "
                   f"rank={dist_get_rank()}, world_size={dist_get_world_size()}")
            return local_rank
        except Exception as e:
            last_err = e

    Logger(f"[WARN] DDP 初始化失败（已尝试 nccl/gloo），已退化为单进程。错误：{last_err}")
    return 0


def get_model_params(model, config):
    """
    打印模型参数量（支持 MoE 的 active params 估算逻辑，保持与原项目一致）。
    """
    total = sum(p.numel() for p in model.parameters()) / 1e6
    n_routed = getattr(config, 'n_routed_experts', getattr(config, 'num_experts', 0))
    n_active = getattr(config, 'num_experts_per_tok', 0)
    n_shared = getattr(config, 'n_shared_experts', 0)
    expert = sum(p.numel() for n, p in model.named_parameters() if 'mlp.experts.0.' in n) / 1e6
    shared_expert = sum(p.numel() for n, p in model.named_parameters() if 'mlp.shared_experts.0.' in n) / 1e6
    base = total - (expert * n_routed) - (shared_expert * n_shared)
    active = base + (expert * n_active) + (shared_expert * n_shared)
    if active < total: Logger(f'Model Params: {total:.2f}M-A{active:.2f}M')
    else: Logger(f'Model Params: {total:.2f}M')


def _safe_torch_load(path: str, map_location="cpu"):
    """
    安全加载 torch.save 文件。
    - 若文件损坏（例如 zip central directory 缺失）会抛异常；
      上层会捕获并改名为 .bad。
    """
    return torch.load(path, map_location=map_location)


def lm_checkpoint(
    lm_config,
    weight: str = 'full_sft',
    model=None,
    optimizer=None,
    epoch: int = 0,
    step: int = 0,
    wandb=None,
    save_dir: str = '../checkpoints',
    **kwargs
):
    """
    保存/加载 checkpoint。

    保存模式（model != None）：
    - 保存权重：{save_dir}/{weight}_{hidden}{_moe}.pth
    - 保存续训：{save_dir}/{weight}_{hidden}{_moe}_resume.pth（含 optimizer/scaler/epoch/step 等）

    关键鲁棒性：
    - 所有保存采用 .tmp + os.replace 原子替换，避免半截文件
    - resume 加载捕获异常（损坏文件自动改名为 .bad 并跳过）
    """
    os.makedirs(save_dir, exist_ok=True)
    moe_path = '_moe' if getattr(lm_config, "use_moe", False) else ''
    ckp_path = f'{save_dir}/{weight}_{lm_config.hidden_size}{moe_path}.pth'
    resume_path = f'{save_dir}/{weight}_{lm_config.hidden_size}{moe_path}_resume.pth'

    # ---------------- 保存模式 ----------------
    if model is not None:
        from torch.nn.parallel import DistributedDataParallel

        # 1) 获取 state_dict（DDP 取 module）
        state_dict = model.module.state_dict() if isinstance(model, DistributedDataParallel) else model.state_dict()
        # 2) 转 half + cpu，节省磁盘
        state_dict = {k: v.half().cpu() for k, v in state_dict.items()}

        # 3) 原子写入模型权重
        ckp_tmp = ckp_path + '.tmp'
        torch.save(state_dict, ckp_tmp)
        os.replace(ckp_tmp, ckp_path)

        # 4) 取 wandb/swanlab run id（尽可能兼容多种对象）
        wandb_id = None
        if wandb:
            if hasattr(wandb, 'get_run'):
                run = wandb.get_run()
                wandb_id = getattr(run, 'id', None) if run else None
            else:
                wandb_id = getattr(wandb, 'id', None)

        # 5) 组织 resume 数据（尽量可续训）
        resume_data = {
            'model': state_dict,
            'optimizer': optimizer.state_dict() if optimizer is not None else None,
            'epoch': epoch,
            'step': step,
            'world_size': dist_get_world_size(),
            'rank': dist_get_rank(),
            'wandb_id': wandb_id
        }

        # 6) 额外状态（如 scaler）通过 kwargs 注入
        #    - 如果 value 有 state_dict，就存其 state_dict
        #    - 否则原样存
        for key, value in kwargs.items():
            if value is None:
                continue
            if hasattr(value, 'state_dict'):
                # 特判 DDP 包裹对象（不常见，但保持兼容）
                if isinstance(value, DistributedDataParallel):
                    resume_data[key] = value.module.state_dict()
                else:
                    resume_data[key] = value.state_dict()
            else:
                resume_data[key] = value

        # 7) 原子写入 resume
        resume_tmp = resume_path + '.tmp'
        torch.save(resume_data, resume_tmp)
        os.replace(resume_tmp, resume_path)

        # 8) 清理
        del state_dict, resume_data
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return None

    # ---------------- 加载模式 ----------------
    if os.path.exists(resume_path):
        try:
            ckp_data = _safe_torch_load(resume_path, map_location='cpu')
        except Exception as e:
            # 遇到损坏文件：隔离为 .bad，避免 --from_resume 直接崩
            Logger(f"[WARN] checkpoint 读取失败（可能损坏），已跳过：{resume_path}\n错误：{e}")
            try:
                os.replace(resume_path, resume_path + ".bad")
            except Exception:
                pass
            return None

        # world_size 改变时自动换算 step（保留原项目意图）
        saved_ws = ckp_data.get('world_size', 1) or 1
        current_ws = dist_get_world_size()
        if saved_ws != current_ws and current_ws > 0:
            old_step = ckp_data.get('step', 0) or 0
            ckp_data['step'] = old_step * saved_ws // current_ws
            Logger(f'GPU数量变化({saved_ws}→{current_ws})，step已自动转换为{ckp_data["step"]}')
        return ckp_data

    return None


def init_model(
    lm_config,
    from_weight: str = 'pretrain',
    tokenizer_path: str = '../model',
    save_dir: str = '../out',
    device: str = 'cuda'
):
    """
    初始化模型与 tokenizer，并按需加载权重。

    注意：
    - from_weight='none' 时不加载权重
    - 权重默认从 {save_dir}/{from_weight}_{hidden}{_moe}.pth 加载
    """
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    model = MiniMindForCausalLM(lm_config)

    if from_weight != 'none':
        moe_suffix = '_moe' if getattr(lm_config, "use_moe", False) else ''
        weight_path = f'{save_dir}/{from_weight}_{lm_config.hidden_size}{moe_suffix}.pth'
        if not os.path.exists(weight_path):
            raise FileNotFoundError(
                f"未找到权重文件：{weight_path}\n"
                f"请确认权重已保存到该路径，或通过 --save_dir/--from_weight 参数调整。"
            )

        # map_location 支持 'cuda:0' / 'cpu' 等
        weights = torch.load(weight_path, map_location=device)
        model.load_state_dict(weights, strict=False)

    get_model_params(model, lm_config)
    Logger(f'Trainable Params: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.3f}M')
    return model.to(device), tokenizer


class SkipBatchSampler(Sampler):
    """
    用于断点续训时跳过前 N 个 batch。
    - sampler: 原始采样器（DistributedSampler 或 range(len(ds))）
    - batch_size: batch 大小
    - skip_batches: 需要跳过的 batch 数
    """
    def __init__(self, sampler, batch_size: int, skip_batches: int = 0):
        self.sampler = sampler
        self.batch_size = batch_size
        self.skip_batches = skip_batches

    def __iter__(self):
        batch = []
        skipped = 0
        for idx in self.sampler:
            batch.append(idx)
            if len(batch) == self.batch_size:
                if skipped < self.skip_batches:
                    skipped += 1
                    batch = []
                    continue
                yield batch
                batch = []
        # 处理尾 batch
        if len(batch) > 0 and skipped >= self.skip_batches:
            yield batch

    def __len__(self):
        total_batches = (len(self.sampler) + self.batch_size - 1) // self.batch_size
        return max(0, total_batches - self.skip_batches)
