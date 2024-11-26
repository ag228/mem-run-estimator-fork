import argparse
import contextlib
import copy
import os
from pathlib import Path
import time
from typing import Any, ContextManager, Dict, List, Tuple

import torch
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
from torch.distributed._tools import MemTracker, RuntimeEstimator
from torch._subclasses.fake_tensor import FakeTensorMode

from exp_utils import create_training_setup, DEVICE, gpu_types, model_names, Precision, runtime_est_modes, ExpType, BASE_DIR, OUT_DIR, TestMode, write_to_logfile, override_args_with_configs

torch.backends.cuda.enable_flash_sdp(enabled=True)

input_configs = {
    "hf_T5": [
        {"batch_size": 6, "seq_len": 512, "precision": Precision.MP, "ac": False, "image_size": -1, "num_denoising_steps": -1},
        {"batch_size": 4, "seq_len": 1024, "precision": Precision.HP, "ac": False, "image_size": -1, "num_denoising_steps": -1},
        {"batch_size": 1, "seq_len": 2048, "precision": Precision.HP, "ac": True, "image_size": -1, "num_denoising_steps": -1},
        {"batch_size": 2, "seq_len": 1024, "precision": Precision.FP, "ac": True, "image_size": -1, "num_denoising_steps": -1},
        {"batch_size": 1, "seq_len": 2048, "precision": Precision.MP, "ac": True, "image_size": -1, "num_denoising_steps": -1},
        {"batch_size": 1, "seq_len": 2048, "precision": Precision.HP, "ac": True, "image_size": -1, "num_denoising_steps": -1},
        {"batch_size": 1, "seq_len": 2048, "precision": Precision.FP, "ac": True, "image_size": -1, "num_denoising_steps": -1},
    ],
    "hf_GPT2": [
        {"batch_size": 16, "seq_len": 512, "precision": Precision.MP, "ac": False, "image_size": -1, "num_denoising_steps": -1},
        {"batch_size": 16, "seq_len": 1024, "precision": Precision.HP, "ac": False, "image_size": -1, "num_denoising_steps": -1},
        {"batch_size": 16, "seq_len": 2048, "precision": Precision.HP, "ac": True, "image_size": -1, "num_denoising_steps": -1},
        {"batch_size": 8, "seq_len": 4096, "precision": Precision.HP, "ac": True, "image_size": -1, "num_denoising_steps": -1},
        {"batch_size": 8, "seq_len": 1024, "precision": Precision.MP, "ac": False, "image_size": -1, "num_denoising_steps": -1},
        {"batch_size": 8, "seq_len": 2048, "precision": Precision.FP, "ac": True, "image_size": -1, "num_denoising_steps": -1},
        {"batch_size": 2, "seq_len": 8192, "precision": Precision.HP, "ac": True, "image_size": -1, "num_denoising_steps": -1},
        {"batch_size": 16, "seq_len": 2048, "precision": Precision.FP, "ac": True, "image_size": -1, "num_denoising_steps": -1},
    ],
    "timm_vit": [
        {"batch_size": 32, "seq_len": -1, "precision": Precision.FP, "ac": False, "image_size": 224, "num_denoising_steps": -1},
        {"batch_size": 64, "seq_len": -1, "precision": Precision.MP, "ac": False, "image_size": 224, "num_denoising_steps": -1},
        {"batch_size": 64, "seq_len": -1, "precision": Precision.HP, "ac": False, "image_size": 224, "num_denoising_steps": -1},
        {"batch_size": 128, "seq_len": -1, "precision": Precision.HP, "ac": True, "image_size": 224, "num_denoising_steps": -1},
        {"batch_size": 64, "seq_len": -1, "precision": Precision.MP, "ac": False, "image_size": 224, "num_denoising_steps": -1},
        {"batch_size": 256, "seq_len": -1, "precision": Precision.HP, "ac": True, "image_size": 224, "num_denoising_steps": -1},
        {"batch_size": 64, "seq_len": -1, "precision": Precision.FP, "ac": True, "image_size": 224, "num_denoising_steps": -1},
    ],
    "hf_clip": [
        {"batch_size": 32, "seq_len": 20, "precision": Precision.FP, "ac": False, "image_size": 336, "num_denoising_steps": -1},
        {"batch_size": 64, "seq_len": 20, "precision": Precision.MP, "ac": False, "image_size": 336, "num_denoising_steps": -1},
        {"batch_size": 64, "seq_len": 20, "precision": Precision.HP, "ac": True, "image_size": 336, "num_denoising_steps": -1},
        {"batch_size": 32, "seq_len": 20, "precision": Precision.FP, "ac": False, "image_size": 336, "num_denoising_steps": -1},
        {"batch_size": 64, "seq_len": 20, "precision": Precision.MP, "ac": False, "image_size": 336, "num_denoising_steps": -1},
        {"batch_size": 128, "seq_len": 20, "precision": Precision.HP, "ac": True, "image_size": 336, "num_denoising_steps": -1},
        {"batch_size": 64, "seq_len": 20, "precision": Precision.FP, "ac": True, "image_size": 336, "num_denoising_steps": -1},
    ],
    "llama_v3_1b": [
        {"batch_size": 4, "seq_len": 1024, "precision": Precision.FP, "ac": False, "image_size": -1, "num_denoising_steps": -1},
        {"batch_size": 4, "seq_len": 2048, "precision": Precision.HP, "ac": True, "image_size": -1, "num_denoising_steps": -1},
        {"batch_size": 4, "seq_len": 4096, "precision": Precision.HP, "ac": True, "image_size": -1, "num_denoising_steps": -1},
        {"batch_size": 8, "seq_len": 2048, "precision": Precision.HP, "ac": True, "image_size": -1, "num_denoising_steps": -1},
        {"batch_size": 2, "seq_len": 8192, "precision": Precision.HP, "ac": True, "image_size": -1, "num_denoising_steps": -1},
        {"batch_size": 4, "seq_len": 1024, "precision": Precision.MP, "ac": False, "image_size": -1, "num_denoising_steps": -1},
        {"batch_size": 4, "seq_len": 2048, "precision": Precision.FP, "ac": True, "image_size": -1, "num_denoising_steps": -1},
        {"batch_size": 1, "seq_len": 16384, "precision": Precision.HP, "ac": True, "image_size": -1, "num_denoising_steps": -1},
        {"batch_size": 8, "seq_len": 2048, "precision": Precision.FP, "ac": True, "image_size": -1, "num_denoising_steps": -1},
    ],
    "gemma_2b": [
        {"batch_size": 8, "seq_len": 512, "precision": Precision.MP, "ac": False, "image_size": -1, "num_denoising_steps": -1},
        {"batch_size": 8, "seq_len": 1024, "precision": Precision.HP, "ac": False, "image_size": -1, "num_denoising_steps": -1},
        {"batch_size": 4, "seq_len": 2048, "precision": Precision.HP, "ac": True, "image_size": -1, "num_denoising_steps": -1},
        {"batch_size": 2, "seq_len": 4096, "precision": Precision.HP, "ac": True, "image_size": -1, "num_denoising_steps": -1},
        {"batch_size": 4, "seq_len": 1024, "precision": Precision.FP, "ac": True, "image_size": -1, "num_denoising_steps": -1},
        {"batch_size": 2, "seq_len": 2048, "precision": Precision.FP, "ac": True, "image_size": -1, "num_denoising_steps": -1},
        {"batch_size": 2, "seq_len": 2048, "precision": Precision.MP, "ac": False, "image_size": -1, "num_denoising_steps": -1},
    ],
    "timm_convnext_v2": [
        {"batch_size": 16, "seq_len": -1, "precision": Precision.FP, "ac": False, "image_size": 224, "num_denoising_steps": -1},
        {"batch_size": 32, "seq_len": -1, "precision": Precision.MP, "ac": False, "image_size": 224, "num_denoising_steps": -1},
        {"batch_size": 64, "seq_len": -1, "precision": Precision.MP, "ac": False, "image_size": 224, "num_denoising_steps": -1},
        {"batch_size": 64, "seq_len": -1, "precision": Precision.HP, "ac": False, "image_size": 224, "num_denoising_steps": -1},
        {"batch_size": 128, "seq_len": -1, "precision": Precision.HP, "ac": True, "image_size": 224, "num_denoising_steps": -1},
        {"batch_size": 32, "seq_len": -1, "precision": Precision.FP, "ac": True, "image_size": 224, "num_denoising_steps": -1},
        {"batch_size": 256, "seq_len": -1, "precision": Precision.HP, "ac": True, "image_size": 224, "num_denoising_steps": -1},
        {"batch_size": 128, "seq_len": -1, "precision": Precision.FP, "ac": True, "image_size": 224, "num_denoising_steps": -1},
    ],
    "stable_diffusion": [
        {"batch_size": 16, "seq_len": 77, "precision": Precision.FP, "ac": False, "image_size": 128, "num_denoising_steps": 50},
        {"batch_size": 32, "seq_len": 77, "precision": Precision.FP, "ac": False, "image_size": 128, "num_denoising_steps": 50},
        {"batch_size": 64, "seq_len": 77, "precision": Precision.FP, "ac": False, "image_size": 128, "num_denoising_steps": 50},
        {"batch_size": 128, "seq_len": 77, "precision": Precision.FP, "ac": False, "image_size": 128, "num_denoising_steps": 50},
        {"batch_size": 16, "seq_len": 77, "precision": Precision.FP, "ac": False, "image_size": 256, "num_denoising_steps": 50},
        {"batch_size": 32, "seq_len": 77, "precision": Precision.FP, "ac": False, "image_size": 256, "num_denoising_steps": 50},
        {"batch_size": 64, "seq_len": 77, "precision": Precision.FP, "ac": False, "image_size": 256, "num_denoising_steps": 50},
        {"batch_size": 128, "seq_len": 77, "precision": Precision.FP, "ac": False, "image_size": 256, "num_denoising_steps": 50},
        {"batch_size": 16, "seq_len": 77, "precision": Precision.FP, "ac": False, "image_size": 512, "num_denoising_steps": 50},
        {"batch_size": 32, "seq_len": 77, "precision": Precision.FP, "ac": False, "image_size": 512, "num_denoising_steps": 50},
        {"batch_size": 64, "seq_len": 77, "precision": Precision.FP, "ac": False, "image_size": 512, "num_denoising_steps": 50},
        {"batch_size": 128, "seq_len": 77, "precision": Precision.FP, "ac": False, "image_size": 512, "num_denoising_steps": 50},
    ],
    "flux": [
        {"batch_size": 1, "seq_len": 256, "precision": Precision.FP, "ac": False, "image_size": 16, "num_denoising_steps": 50},
        {"batch_size": 1, "seq_len": 256, "precision": Precision.FP, "ac": False, "image_size": 64, "num_denoising_steps": 50},
        {"batch_size": 16, "seq_len": 256, "precision": Precision.FP, "ac": False, "image_size": 64, "num_denoising_steps": 50},
        {"batch_size": 32, "seq_len": 256, "precision": Precision.FP, "ac": False, "image_size": 64, "num_denoising_steps": 50},
        {"batch_size": 64, "seq_len": 256, "precision": Precision.FP, "ac": False, "image_size": 64, "num_denoising_steps": 50},
        {"batch_size": 1, "seq_len": 256, "precision": Precision.FP, "ac": False, "image_size": 128, "num_denoising_steps": 50},
        {"batch_size": 16, "seq_len": 256, "precision": Precision.FP, "ac": False, "image_size": 128, "num_denoising_steps": 50},
        {"batch_size": 32, "seq_len": 256, "precision": Precision.FP, "ac": False, "image_size": 128, "num_denoising_steps": 50},
        {"batch_size": 64, "seq_len": 256, "precision": Precision.FP, "ac": False, "image_size": 128, "num_denoising_steps": 50},
        {"batch_size": 1, "seq_len": 256, "precision": Precision.FP, "ac": False, "image_size": 512, "num_denoising_steps": 50},
        {"batch_size": 16, "seq_len": 256, "precision": Precision.FP, "ac": False, "image_size": 512, "num_denoising_steps": 50},
        {"batch_size": 32, "seq_len": 256, "precision": Precision.FP, "ac": False, "image_size": 512, "num_denoising_steps": 50},
        {"batch_size": 64, "seq_len": 256, "precision": Precision.FP, "ac": False, "image_size": 512, "num_denoising_steps": 50},
    ],
    "stable_diffusion_mmdit": [
        {"batch_size": 1, "seq_len": 256, "precision": Precision.HP, "ac": False, "image_size": 16, "num_denoising_steps": 50},
        {"batch_size": 1, "seq_len": 256, "precision": Precision.HP, "ac": False, "image_size": 64, "num_denoising_steps": 50},
        {"batch_size": 16, "seq_len": 256, "precision": Precision.HP, "ac": False, "image_size": 64, "num_denoising_steps": 50},
        {"batch_size": 32, "seq_len": 256, "precision": Precision.HP, "ac": False, "image_size": 64, "num_denoising_steps": 50},
        {"batch_size": 64, "seq_len": 256, "precision": Precision.HP, "ac": False, "image_size": 64, "num_denoising_steps": 50},
        {"batch_size": 1, "seq_len": 256, "precision": Precision.HP, "ac": False, "image_size": 128, "num_denoising_steps": 50},
        {"batch_size": 16, "seq_len": 256, "precision": Precision.HP, "ac": False, "image_size": 128, "num_denoising_steps": 50},
        {"batch_size": 32, "seq_len": 256, "precision": Precision.HP, "ac": False, "image_size": 128, "num_denoising_steps": 50},
        {"batch_size": 64, "seq_len": 256, "precision": Precision.HP, "ac": False, "image_size": 128, "num_denoising_steps": 50},
        {"batch_size": 1, "seq_len": 256, "precision": Precision.HP, "ac": False, "image_size": 512, "num_denoising_steps": 50},
        {"batch_size": 16, "seq_len": 256, "precision": Precision.HP, "ac": False, "image_size": 512, "num_denoising_steps": 50},
        {"batch_size": 32, "seq_len": 256, "precision": Precision.HP, "ac": False, "image_size": 512, "num_denoising_steps": 50},
        {"batch_size": 64, "seq_len": 256, "precision": Precision.HP, "ac": False, "image_size": 512, "num_denoising_steps": 50},
    ],
}


class Experiment:

    def __init__(self, args):  

        self.exp_type: ExpType
        if args.real_execution:
            self.exp_type = ExpType.real_execution
        elif args.memory_estimation:
            self.exp_type = ExpType.memory_est
        elif args.runtime_estimation:
            self.exp_type = ExpType.runtime_est
            self.est_mode = args.runtime_estimation_mode
        elif args.test:
            self.exp_type = ExpType.test

        init_mode = contextlib.nullcontext() if self.exp_type in [ExpType.real_execution, ExpType.test] else FakeTensorMode()
        dev = torch.device(DEVICE)
        self.execution_ctx = init_mode
        self.device = dev
        self.setup_cfg = {
            "model_name": args.model_name,
            "batch_size": args.batch_size,
            "seq_len": args.seq_len,
            "precision": Precision(args.precision),
            "ac": args.enable_ac,
            "image_size": args.image_size,
            "init_mode": init_mode,
            "dev": dev,
            "num_denoising_steps": args.num_denoising_steps,
        }
        self.gpu_type = args.gpu_type
        self.model, self.optimizer, self.train_step = create_training_setup(**self.setup_cfg)
        self.model.train()
        print(self.model)
        
        # for name, module in self.model.named_modules():
        #     print(name)
        #     param_dtypes = set()
        #     param_count = 0
        #     param_size = 0
        #     for p in module.parameters():
        #         param_numel = p.numel()
        #         param_count += param_numel
        #         param_size += param_numel * p.dtype.itemsize
        #         param_dtypes.add(p.dtype)

        #     print(f"Model has {param_count} parameters.")
        #     print(f"Model has {param_dtypes} dtypes.")
        #     print(f"Parameter Memory: {param_size / 2**30:.3f} GiB")

    def real_execution(self) -> Tuple[float, int, int]:
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.reset_accumulated_memory_stats()
        warm_up_iters, benchmark_iters = 2, 3
        total_iters = warm_up_iters + benchmark_iters
        start_events = [torch.cuda.Event(enable_timing=True) for _ in range(total_iters)]
        end_events = [torch.cuda.Event(enable_timing=True) for _ in range(total_iters)]
        for i in range(5):
            start_events[i].record()
            with self.execution_ctx:
                self.train_step(self.model, self.optimizer)
            end_events[i].record()
        torch.cuda.synchronize()
        iter_time = (
            sum(start_events[i].elapsed_time(end_events[i]) for i in range(warm_up_iters, total_iters)) / benchmark_iters
        )
        mem_stats = torch.cuda.memory_stats()
        peak_active = mem_stats["active_bytes.all.peak"]
        peak_reserved = mem_stats["reserved_bytes.all.peak"]
        print(f"Iter time: {iter_time} ms")
        print(f"Peak Active Memory: {peak_active / 2**30} GiB")
        print(f"Peak Reserved Memory: {peak_reserved / 2**30} GiB")

        return iter_time, peak_active, peak_reserved
    
    def memory_estimation(self) -> Tuple[int, float]:
        iters = 2
        mem_tracker = MemTracker()
        mem_tracker.track_external(self.model, self.optimizer)

        for iter in range(iters):
            track_start_time = time.time()
            with self.execution_ctx:
                with mem_tracker:
                    self.train_step(self.model, self.optimizer)
            track_end_time = time.time()
            if iter == 0:
                mem_tracker.reset_mod_stats()
        peak_tracker = mem_tracker.get_tracker_snapshot("peak")[self.device]["Total"]
        mem_tracker.display_snapshot("peak", units="GiB", tabulate=True)
        tracking_time = (track_end_time - track_start_time) * 1e3
        print(f"Memory Tracking time (ms): {tracking_time}")
        return (peak_tracker, tracking_time)
    
    def runtime_estimation(self, estimate_mode: str) -> Tuple[float, float]:
        runtime_estimator = RuntimeEstimator()
        with self.execution_ctx:
            self.train_step(self.model, self.optimizer)
        est_start_time = time.time()
        with self.execution_ctx:
            with runtime_estimator(estimate_mode_type=estimate_mode):
                self.train_step(self.model, self.optimizer)
        torch.cuda.synchronize()
        est_end_time = time.time()
        estimation_time = (est_end_time - est_start_time) * 1e3
        run_est = runtime_estimator.total_runtime
        print(f"Estimation time (ms): {estimation_time}")
        return (run_est, estimation_time)

    def test(self) -> Tuple[float, int, int]:
        with self.execution_ctx:
            self.train_step(self.model, self.optimizer)
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.reset_accumulated_memory_stats()

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()
        with self.execution_ctx:
            self.train_step(self.model, self.optimizer)
        end_event.record()
        torch.cuda.synchronize()
        iter_time = start_event.elapsed_time(end_event)
        mem_stats = torch.cuda.memory_stats()
        peak_active = mem_stats["active_bytes.all.peak"]
        peak_reserved = mem_stats["reserved_bytes.all.peak"]
        print(f"Iter time: {iter_time} ms")
        print(f"Peak Active Memory: {peak_active / 2**30} GiB")
        print(f"Peak Reserved Memory: {peak_reserved / 2**30} GiB")

        return iter_time, peak_active, peak_reserved

    def run(self,):
        Path(f"{OUT_DIR}/").mkdir(parents=True, exist_ok=True)
        if self.exp_type == ExpType.runtime_est:
            out_file = f"{OUT_DIR}/{self.exp_type.value}_{self.est_mode}_{self.gpu_type}_CFG2.csv"
        else:
            out_file = f"{OUT_DIR}/{self.exp_type.value}_{self.gpu_type}.csv"

        cfg = self.setup_cfg
        log_record = [
            cfg['model_name'], cfg['batch_size'], cfg["seq_len"], cfg["image_size"], cfg["num_denoising_steps"], cfg['precision'].value, cfg['ac']
        ]
        if self.exp_type == ExpType.test:
            iter_time, peak_active, peak_reserved = self.test()
            log_record.extend([iter_time, peak_active, peak_reserved])
        elif self.exp_type == ExpType.real_execution:
            iter_time, peak_active, peak_reserved = self.real_execution()
            log_record.extend([iter_time, peak_active, peak_reserved])
        elif self.exp_type == ExpType.runtime_est:
            run_est, est_time = self.runtime_estimation(self.est_mode)
            log_record.extend([self.est_mode, run_est, est_time])
        elif self.exp_type == ExpType.memory_est:
            peak_mem_est, est_time = self.memory_estimation()
            log_record.extend([peak_mem_est, est_time])
            if peak_mem_est > (70 * 2**30):
                print(f"Delete: {log_record}")

        write_to_logfile(out_file, log_record)


def experiment_runner(args):
    if args.preset_config:
        m_args = override_args_with_configs(args, input_configs[args.model_name][args.config_idx])  
    else: 
        m_args = args
    try:
        if m_args.precision == "HP":
            torch.set_default_dtype(torch.float16)
        exp = Experiment(m_args)
        exp.run()
    except Exception as e:
        print(f"Experiment failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        type=str,
        default="gemma_2b",
        choices=model_names,
        help=f"Model name",
    )    
    parser.add_argument(
        "--batch_size",
        default=2,
        type=int,
        help="Training batch size"
    )
    parser.add_argument(
        "--seq_len",
        default=64,
        type=int,
        help="Training equence length"
    )
    parser.add_argument(
        "--image_size",
        default=224,
        type=int,
        help="Training image size"
    )
    parser.add_argument(
        "--num_denoising_steps", 
        default=50,
        type=int,
        help="Number of denoising steps for diffusion"
    )
    parser.add_argument(
        "--precision",
        type=str,
        default=Precision.HP.value,
        choices=[p.value for p in Precision],
        help=f"Training precision"
    )
    parser.add_argument(
        "--enable_ac", 
        action="store_true", 
        help="Enables activation checkpointing"
    )
    parser.add_argument(
        "--gpu_type",
        type=str,
        default="H100",
        choices=gpu_types,
        help="GPU type to use",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--real_execution", 
        action="store_true", 
        help="Execute a training iteration"
    )
    group.add_argument(
        "--memory_estimation", 
        action="store_true", 
        help="Estimate training memory"
    )
    group.add_argument(
        "--test", 
        action="store_true", 
        help="Test an actual model run"
    )
    group.add_argument(
        "--runtime_estimation", 
        action="store_true", 
        help="Estimate training runtime"
    )
    group2 = parser.add_mutually_exclusive_group()
    group2.add_argument(
        "--benchmark", 
        action="store_true", 
        help="Estimation methods benchmarking"
    )
    group2.add_argument(
        "--preset_config", 
        action="store_true", 
        help="Choose from existing configs"
    )
    parser.add_argument(
        "--config_idx",
        type=int,
        default=0,
        help=f"Preset config index for the model"
    )
    parser.add_argument(
        "--runtime_estimation_mode",
        type=str,
        default="operator-level-learned-model",
        choices=runtime_est_modes,
        help="Runtime estimation modes",
    )
    args = parser.parse_args()
    print("args = ", args)
    
    if not args.benchmark:
        if args.preset_config:
            m_args = override_args_with_configs(args, input_configs[args.model_name][args.config_idx])  
        else: 
            m_args = args
        try:
            if m_args.precision == "HP":
                torch.set_default_dtype(torch.float16)
            exp = Experiment(m_args)
            exp.run()
        except Exception as e:
            print(f"Experiment failed: {e}")
            import traceback
            traceback.print_exc()
    else:
        assert((not args.test) and (not args.real_execution) and (not args.preset_config)), "No bechmark mode for real execution"
        import concurrent.futures

        with concurrent.futures.ProcessPoolExecutor() as executor:
            futures = []
            for config in input_configs[args.model_name]:
                b_args = override_args_with_configs(args, config)
                if args.runtime_estimation:
                    bench_est_modes = {'operator-level-cost-model', 'operator-level-learned-model'}
                    # bench_est_modes = ['operator-level-learned-model',]
                    for est_mode in bench_est_modes:
                        r_args = copy.deepcopy(b_args)
                        r_args.runtime_estimation_mode = est_mode
                        futures.append(executor.submit(experiment_runner, r_args))
                else:
                    futures.append(executor.submit(experiment_runner, b_args))

            for future in concurrent.futures.as_completed(futures):
                future.result()
