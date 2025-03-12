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
# from torch.distributed._tools.auto_sac import (
#     apply_auto_sac_policies,
#     get_auto_sac_policies,
#     SACAlgorithm,
# )
from torch._subclasses.fake_tensor import FakeTensorMode

from exp_utils import create_training_setup, AC, DEVICE, gpu_types, model_names, Precision, runtime_est_modes, ExpType, OUT_DIR, write_to_logfile, override_args_with_configs
from configs import input_configs
torch.backends.cuda.enable_flash_sdp(enabled=True)
_GiB = 2**30
class Experiment:

    def __init__(self, args):  

        self.exp_type: ExpType
        if args.real_execution:
            self.exp_type = ExpType.real_execution
        elif args.memory_estimation:
            self.exp_type = ExpType.memory_est
        elif args.runtime_estimation:
            self.exp_type = ExpType.runtime_est
        elif args.test:
            self.exp_type = ExpType.test
        elif args.auto_sac:
            self.exp_type = ExpType.auto_sac

        self.runtime_kwargs = {"estimate_mode_type": args.runtime_estimation_mode}
        self.ac_mode = AC(args.ac_mode)
        # self.sac_algo = SACAlgorithm(args.sac_algo)
        self.memory_budget = args.memory_budget

        init_mode = contextlib.nullcontext() if self.exp_type in [ExpType.real_execution, ExpType.test] else FakeTensorMode()
        dev = torch.device(DEVICE)
        self.execution_ctx = init_mode
        self.device = dev
        self.setup_cfg = {
            "model_name": args.model_name,
            "batch_size": args.batch_size,
            "seq_len": args.seq_len,
            "precision": Precision(args.precision),
            "ac": self.ac_mode,
            "image_size": args.image_size,
            "init_mode": init_mode,
            "dev": dev,
            "num_denoising_steps": args.num_denoising_steps,
        }
        self.gpu_type = args.gpu_type
        self.train_step, self.models, self.optimizers, self.inputs = create_training_setup(**self.setup_cfg)
        # if self.ac_mode == AC.AUTO:
        #     with init_mode:
        #         self.optimization_times = self.auto_sac(self.memory_budget, self.sac_algo, self.runtime_kwargs)
        
        for model in self.models:
            param_count = 0
            param_size = 0
            for p in model.parameters():
                param_numel = p.numel()
                param_count += param_numel
                param_size += param_numel * p.dtype.itemsize

            print(f"Model has {model.__class__.__name__} {param_count} parameters.")
            print(f"Parameter Memory: {param_size / _GiB:.3f} GiB")

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
                self.train_step(self.models, self.optimizers, self.inputs)
            end_events[i].record()
        torch.cuda.synchronize()
        iter_time = (
            sum(start_events[i].elapsed_time(end_events[i]) for i in range(warm_up_iters, total_iters)) / benchmark_iters
        )
        mem_stats = torch.cuda.memory_stats()
        peak_active = mem_stats["active_bytes.all.peak"]
        peak_reserved = mem_stats["reserved_bytes.all.peak"]
        print(f"Iter time: {iter_time} ms")
        print(f"Peak Active Memory: {peak_active / _GiB} GiB")
        print(f"Peak Reserved Memory: {peak_reserved / _GiB} GiB")

        return iter_time, peak_active, peak_reserved
    
    def memory_estimation(self) -> Tuple[int, Dict[torch.device, Dict[str, int]], float]:
        iters = 2
        mem_tracker = MemTracker()
        mem_tracker.track_external(*self.models, *self.optimizers, self.inputs)

        for iter in range(iters):
            track_start_time = time.time()
            with self.execution_ctx:
                with mem_tracker:
                    self.train_step(self.models, self.optimizers, self.inputs)
            track_end_time = time.time()
            if iter == 0:
                mem_tracker.reset_mod_stats()
        peak_tracker = mem_tracker.get_tracker_snapshot("peak")[self.device]["Total"]
        mem_tracker.display_snapshot("peak", units="GiB", tabulate=True)
        peak_snapshot = mem_tracker.get_tracker_snapshot("peak")
        tracking_time = (track_end_time - track_start_time) * 1e3
        print(f"Memory Tracking time (ms): {tracking_time}")
        return (peak_tracker, peak_snapshot, tracking_time)
    
    def runtime_estimation(self, runtime_kwargs: Dict[str, Any]) -> Tuple[float, float]:
        runtime_estimator = RuntimeEstimator()
        with self.execution_ctx:
            self.train_step(self.models, self.optimizers, self.inputs)
        est_start_time = time.time()
        with self.execution_ctx:
            with runtime_estimator(**runtime_kwargs):
                self.train_step(self.models, self.optimizers, self.inputs)
        torch.cuda.synchronize()
        est_end_time = time.time()
        estimation_time = (est_end_time - est_start_time) * 1e3
        run_est = runtime_estimator.total_compute_time
        print(f"Estimation time (ms): {estimation_time}")
        return (run_est, estimation_time)

    def test(self) -> Tuple[float, int, int]:
        with self.execution_ctx:
            self.train_step(self.models, self.optimizers, self.inputs)
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.reset_accumulated_memory_stats()

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()
        with self.execution_ctx:
            self.train_step(self.models, self.optimizers, self.inputs)
        end_event.record()
        torch.cuda.synchronize()
        iter_time = start_event.elapsed_time(end_event)
        mem_stats = torch.cuda.memory_stats()
        peak_active = mem_stats["active_bytes.all.peak"]
        peak_reserved = mem_stats["reserved_bytes.all.peak"]
        print(f"Iter time: {iter_time} ms")
        print(f"Peak Active Memory: {peak_active / _GiB} GiB")
        print(f"Peak Reserved Memory: {peak_reserved / _GiB} GiB")

        return iter_time, peak_active, peak_reserved

    # def auto_sac(self, memory_budget: float, sac_algorithm: SACAlgorithm, runtime_kwargs: Dict[str, Any]):
    #     auto_sac_result, optimization_times = get_auto_sac_policies(
    #         self.train_step,
    #         self.models,
    #         self.optimizers,
    #         self.inputs,
    #         torch.device(DEVICE),
    #         memory_budget,
    #         sac_algorithm,
    #         runtime_kwargs=runtime_kwargs
    #     )
    #     for model in self.models:
    #         apply_auto_sac_policies(
    #             model, auto_sac_result.sac_policies, preserve_rng_state=False
    #         )
    #     print(
    #         f"Memory Budget: {memory_budget} GiB\n"
    #         f"Auto-SAC Estimated Memory: {auto_sac_result.peak_mem / _GiB} GiB\n"
    #         f"Estimated recomputation time: {auto_sac_result.recomputation_time} ms"
    #     )
    #     print("Auto-SAC Decisions: ")
    #     for m_name, budget in auto_sac_result.ac_decisions.items():
    #         print(f"{m_name}: {budget}")
    #     return optimization_times

    def run(self,):
        cfg = self.setup_cfg
        log_record = [
            # cfg['model_name'], cfg['batch_size'], cfg["seq_len"], cfg["image_size"], cfg["num_denoising_steps"], cfg['precision'].value, cfg['ac'].value, self.sac_algo.value,
            cfg['model_name'], cfg['batch_size'], cfg["seq_len"], cfg["image_size"], cfg["num_denoising_steps"], cfg['precision'].value, cfg['ac'].value,
            
        ]
        if self.exp_type == ExpType.test:
            iter_time, peak_active, peak_reserved = self.test()
            log_record.extend([iter_time, peak_active, peak_reserved])

        if self.exp_type == ExpType.real_execution:
            iter_time, peak_active, peak_reserved = self.real_execution()
            log_record.extend([iter_time, peak_active, peak_reserved])

        if self.exp_type == ExpType.runtime_est:
            run_est, est_time = self.runtime_estimation(self.runtime_kwargs)
            log_record.extend([self.runtime_kwargs["estimate_mode_type"], run_est, est_time])

        if self.exp_type == ExpType.memory_est:
            peak_mem_est, peak_snapshot, est_time = self.memory_estimation()
            snapshot_log_record = copy.deepcopy(log_record)
            cuda_snapshot = peak_snapshot[torch.device(DEVICE)]
            snapshot_log_record.extend(cuda_snapshot.values())
            log_record.extend([peak_mem_est, est_time])

        if self.exp_type == ExpType.auto_sac:
            peak_mem_est, peak_snapshot, est_time = self.memory_estimation()
            snapshot_log_record = copy.deepcopy(log_record)
            cuda_snapshot = peak_snapshot[torch.device(DEVICE)]
            snapshot_log_record.extend(cuda_snapshot.values())
            mem_log_record = copy.deepcopy(log_record)
            mem_log_record.extend([peak_mem_est, est_time])
            run_est, est_time = self.runtime_estimation(self.runtime_kwargs)
            run_log_record = copy.deepcopy(log_record)
            run_log_record.extend([self.runtime_kwargs["estimate_mode_type"], run_est, est_time])
            sac_log_record = copy.deepcopy(log_record)
            sac_log_record.extend([*self.optimization_times])


        Path(f"{OUT_DIR}/").mkdir(parents=True, exist_ok=True)
        if self.exp_type == ExpType.runtime_est:
            out_file = f"{OUT_DIR}/{self.exp_type.value}_{self.runtime_kwargs['estimate_mode_type']}_{self.gpu_type}.csv"
            write_to_logfile(out_file, log_record)
        elif self.exp_type == ExpType.auto_sac:
            run_out_file = f"{OUT_DIR}/{ExpType.runtime_est.value}_{self.runtime_kwargs['estimate_mode_type']}_{self.gpu_type}.csv"
            mem_out_file = f"{OUT_DIR}/{ExpType.memory_est.value}_{self.gpu_type}.csv"
            snapshot_out_file = f"{OUT_DIR}/{ExpType.memory_est.value}_snapshot_{self.gpu_type}.csv"
            sac_out_file = f"{OUT_DIR}/{self.exp_type.value}_{self.gpu_type}.csv"
            write_to_logfile(run_out_file, run_log_record)
            write_to_logfile(mem_out_file, mem_log_record)
            write_to_logfile(snapshot_out_file, snapshot_log_record)
            write_to_logfile(sac_out_file, sac_log_record)
            
        else:
            out_file = f"{OUT_DIR}/{self.exp_type.value}_{self.gpu_type}.csv"
            write_to_logfile(out_file, log_record)

        if self.exp_type == ExpType.memory_est:
            snapshot_out_file = f"{OUT_DIR}/{self.exp_type.value}_snapshot_{self.gpu_type}.csv"
            write_to_logfile(snapshot_out_file, snapshot_log_record)


def experiment_runner(args):
    if args.preset_config:
        m_args = override_args_with_configs(args, input_configs[args.model_name][args.config_idx])  
    else: 
        m_args = args
    try:
        print("args = ", m_args)
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
        "--ac_mode",
        type=AC,
        default=AC.NONE.value, 
        choices=[ac.value for ac in AC], 
        help="Activation Checkpointing modes"
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
    group.add_argument(
        "--auto_sac",
        action="store_true",
        help="Estimate runtime and memory for Auto-SAC"
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
        "--memory_budget",
        type=float,
        default=68.0,
        help=f"Memory Budget for Auto SAC"
    )
    # parser.add_argument(
    #     "--sac_algo",
    #     type=str,
    #     default=SACAlgorithm.OPTIMAL.value,
    #     choices=[algo.value for algo in SACAlgorithm], 
    #     help=f"SAC Algorithm to use for Auto SAC"
    # )
    parser.add_argument(
        "--runtime_estimation_mode",
        type=str,
        default="operator-level-cost-model",
        choices=runtime_est_modes,
        help="Runtime estimation modes",
    )
    args = parser.parse_args()
    
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
                    # bench_est_modes = {'operator-level-cost-model', 'operator-level-learned-model'}
                    bench_est_modes = ['operator-level-cost-model',]
                    for est_mode in bench_est_modes:
                        r_args = copy.deepcopy(b_args)
                        r_args.runtime_estimation_mode = est_mode
                        futures.append(executor.submit(experiment_runner, r_args))
                # elif args.auto_sac:
                #     sac_algos = [alg.value for alg in SACAlgorithm]
                #     for sac_algo in sac_algos:
                #         s_args = copy.deepcopy(b_args)
                #         s_args.sac_algo = sac_algo
                #         futures.append(executor.submit(experiment_runner, s_args))
                else:
                    futures.append(executor.submit(experiment_runner, b_args))

            for future in concurrent.futures.as_completed(futures):
                future.result()
