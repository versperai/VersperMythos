"""Shared training utilities for RLM."""

import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from src.rlm.model import RLM, RLMConfig


def is_main_process():
    return not dist.is_initialized() or dist.get_rank() == 0


def Logger(content):
    if is_main_process():
        print(content)


def get_model_params(model):
    total = sum(p.numel() for p in model.parameters()) / 1e6
    Logger(f'Model Params: {total:.2f}M')


def init_distributed_mode():
    if int(os.environ.get("RANK", -1)) == -1:
        return 0
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank


def rlm_checkpoint(cfg, weight='pretrain', model=None, optimizer=None,
                   epoch=0, step=0, wandb=None, save_dir='checkpoints/rlm', **kwargs):
    os.makedirs(save_dir, exist_ok=True)
    ckp_path = f'{save_dir}/{weight}_{cfg.dim}.pth'
    resume_path = f'{save_dir}/{weight}_{cfg.dim}_resume.pth'

    if model is not None:
        raw_model = model.module if isinstance(model, DistributedDataParallel) else model
        raw_model = getattr(raw_model, '_orig_mod', raw_model)
        state_dict = raw_model.state_dict()
        state_dict = {k: v.half().cpu() for k, v in state_dict.items()}
        ckp_tmp = ckp_path + '.tmp'
        torch.save(state_dict, ckp_tmp)
        os.replace(ckp_tmp, ckp_path)

        wandb_id = None
        if wandb:
            if hasattr(wandb, 'get_run'):
                run = wandb.get_run()
                wandb_id = getattr(run, 'id', None) if run else None
            else:
                wandb_id = getattr(wandb, 'id', None)

        resume_data = {
            'model': state_dict,
            'optimizer': optimizer.state_dict() if optimizer else None,
            'epoch': epoch,
            'step': step,
            'world_size': dist.get_world_size() if dist.is_initialized() else 1,
            'wandb_id': wandb_id,
        }
        for key, value in kwargs.items():
            if value is not None:
                if hasattr(value, 'state_dict'):
                    raw_value = value.module if isinstance(value, DistributedDataParallel) else value
                    raw_value = getattr(raw_value, '_orig_mod', raw_value)
                    resume_data[key] = raw_value.state_dict()
                else:
                    resume_data[key] = value

        resume_tmp = resume_path + '.tmp'
        torch.save(resume_data, resume_tmp)
        os.replace(resume_tmp, resume_path)
        del state_dict, resume_data
        torch.cuda.empty_cache()
    else:
        if os.path.exists(resume_path):
            ckp_data = torch.load(resume_path, map_location='cpu')
            saved_ws = ckp_data.get('world_size', 1)
            current_ws = dist.get_world_size() if dist.is_initialized() else 1
            if saved_ws != current_ws:
                ckp_data['step'] = ckp_data['step'] * saved_ws // current_ws
                Logger(f'GPU count change ({saved_ws} -> {current_ws}), step auto-adjusted to {ckp_data["step"]}')
            return ckp_data
        return None


def init_rlm_model(cfg, from_weight=None, save_dir='out/rlm', device='cuda'):
    model = RLM(cfg)
    if from_weight:
        weight_path = f'{save_dir}/{from_weight}_{cfg.dim}.pth'
        weights = torch.load(weight_path, map_location=device)
        model.load_state_dict(weights, strict=False)
    get_model_params(model)
    return model.to(device)
