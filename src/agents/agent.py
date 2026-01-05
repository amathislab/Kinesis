# Copyright (c) 2025 Mathis Group for Computational Neuroscience and AI, EPFL
# All rights reserved.
#
# Licensed under the BSD 3-Clause License.
#
# This file contains code adapted from:
#
# 1. SMPLSim (https://github.com/ZhengyiLuo/SMPLSim)
#    Copyright (c) 2024 Zhengyi Luo
#    Licensed under the BSD 3-Clause License.

import math
import time
import os
import torch
import numpy as np
import traceback
import signal
import psutil
import logging
import sys
import faulthandler
from datetime import datetime

import torch.multiprocessing as multiprocessing

import gymnasium as gym

from src.learning.policy_moe import PolicyMOE, PolicyMOEWithPrev
from src.learning.memory import Memory
from src.learning.trajbatch import TrajBatch
from src.learning.logger_rl import LoggerRL
from src.learning.learning_utils import to_test, to_cpu, rescale_actions
import random
random.seed(0)

from typing import Any, Optional, List, Tuple

os.environ["OMP_NUM_THREADS"] = "1"

# Enable fault handler to get Python stack trace on segfault
faulthandler.enable()


def setup_worker_logger(pid: int, log_dir: str = "tmp/rl_debug"):
    """Setup file-based logging for each worker"""
    os.makedirs(log_dir, exist_ok=True)
    
    # Create a logger for this worker
    logger = logging.getLogger(f'worker_{pid}')
    logger.setLevel(logging.DEBUG)
    
    # Create file handler
    log_file = os.path.join(log_dir, f'worker_{pid}.log')
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    
    # Add handler to logger
    logger.addHandler(fh)
    
    # Also log to a shared file for critical errors
    shared_fh = logging.FileHandler(os.path.join(log_dir, 'all_workers.log'))
    shared_fh.setLevel(logging.ERROR)
    shared_fh.setFormatter(formatter)
    logger.addHandler(shared_fh)
    
    return logger


def monitor_resources(logger, pid: int, tag: str = ""):
    """Log current resource usage"""
    try:
        process = psutil.Process()
        memory_info = process.memory_info()
        
        # System-wide resources
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory_percent = psutil.virtual_memory().percent
        
        # Process-specific resources
        proc_memory_mb = memory_info.rss / 1024 / 1024
        proc_cpu_percent = process.cpu_percent(interval=0.1)
        num_fds = process.num_fds() if hasattr(process, 'num_fds') else 'N/A'
        
        # logger.info(f"[{tag}] Worker {pid} - Process Memory: {proc_memory_mb:.1f}MB, "
        #            f"Process CPU: {proc_cpu_percent:.1f}%, FDs: {num_fds}, "
        #            f"System CPU: {cpu_percent:.1f}%, System Memory: {memory_percent:.1f}%")
        
        # Check for high resource usage
        if memory_percent > 90:
            logger.warning(f"HIGH MEMORY USAGE: {memory_percent}%")
        if proc_memory_mb > 10000:  # 10GB per process
            logger.warning(f"HIGH PROCESS MEMORY: {proc_memory_mb}MB")
            
    except Exception as e:
        logger.error(f"Failed to monitor resources: {e}")


class Agent:

    def __init__(
        self,
        env: gym.Env,
        policy_net: torch.nn.Module,
        value_net: torch.nn.Module,
        dtype: torch.dtype,
        device: torch.device,
        gamma: float,
        mean_action: bool = False,
        headless: bool = False,
        num_threads: int = 1,
        clip_obs: bool = False,
        clip_actions: bool = False,
        clip_obs_range: Optional[List[float]] = None,
        debug_dir: str = "tmp/rl_debug",
    ):
        self.env = env
        self.policy_net = policy_net
        self.value_net = value_net
        self.dtype = dtype
        self.device = device
        self.np_dtype = np.float32
        self.gamma = gamma
        self.mean_action = mean_action
        self.headless = headless
        self.num_threads = num_threads
        self.noise_rate = 1.0
        self.num_steps = 0
        self.traj_cls = TrajBatch
        self.logger_rl_cls = LoggerRL
        self.sample_modules = [policy_net]
        self.update_modules = [policy_net, value_net]
        self.clip_obs = clip_obs
        self.clip_actions = clip_actions
        self.obs_low = clip_obs_range[0] if clip_obs_range else None
        self.obs_high = clip_obs_range[1] if clip_obs_range else None
        self.debug_dir = debug_dir
        self._setup_action_space()
        
        # Setup main process logger
        if (not self.env.cfg.run.im_eval) or (not self.env.cfg.run.test):
            self.main_logger = setup_worker_logger('main', self.debug_dir)
            self.main_logger.info(f"Agent initialized with {num_threads} threads")

    def _setup_action_space(self) -> None:
        action_space = self.env.action_space
        self.actions_num = action_space.shape[0]
        self.actions_low = action_space.low.copy()
        self.actions_high = action_space.high.copy()

    def seed_worker(self, pid: int) -> None:
        if pid > 0:
            random.seed(self.epoch)
            seed = random.randint(0, 5000) * pid

            torch.manual_seed(seed)
            np.random.seed(seed)
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            torch.use_deterministic_algorithms(True)

    def sample_worker_wrapper(
        self, pid: int, queue: Optional[multiprocessing.Queue], min_batch_size: int,
        error_queue: multiprocessing.Queue
    ):
        """Wrapper to catch any catastrophic failures"""
        logger = setup_worker_logger(pid, self.debug_dir)
        # logger.info(f"Worker {pid} wrapper started, PID: {os.getpid()}")
        
        try:
            # Set up signal handlers
            def emergency_handler(signum, frame):
                logger.critical(f"Worker {pid} received signal {signum}")
                error_queue.put((pid, f"Signal {signum}", traceback.format_stack(frame)))
                sys.exit(1)
            
            signal.signal(signal.SIGSEGV, emergency_handler)
            signal.signal(signal.SIGBUS, emergency_handler)
            
            # Run the actual worker
            self.sample_worker(pid, queue, min_batch_size, logger, error_queue)
            
        except Exception as e:
            logger.critical(f"Worker {pid} catastrophic failure: {e}\n{traceback.format_exc()}")
            error_queue.put((pid, str(e), traceback.format_exc()))
            if queue is not None:
                queue.put([pid, None, None, f"Catastrophic: {e}"])

    def sample_worker(
        self, pid: int, queue: Optional[multiprocessing.Queue], min_batch_size: int,
        logger: logging.Logger, error_queue: multiprocessing.Queue
    ) -> Optional[Tuple[Memory, LoggerRL]]:
        """Worker function with extensive debugging"""
        
        # logger.info(f"Worker {pid} started, PID: {os.getpid()}")
        monitor_resources(logger, pid, "START")
        
        try:
            # Log system limits
            import resource
            soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
            # logger.info(f"Worker {pid} file descriptor limits: soft={soft}, hard={hard}")
            
            soft, hard = resource.getrlimit(resource.RLIMIT_AS)
            # logger.info(f"Worker {pid} virtual memory limits: soft={soft}, hard={hard}")
            
        except Exception as e:
            logger.warning(f"Could not get resource limits: {e}")

        self.seed_worker(pid)
        # logger.info(f"Worker {pid} seeded")

        # Create memory and logger instances
        memory = None
        rl_logger = None
        
        try:
            memory = Memory()
            # logger.info(f"Worker {pid} created Memory")
            
            rl_logger = self.logger_rl_cls()
            # logger.info(f"Worker {pid} created LoggerRL")

            # Execute pre-sample operations
            self.pre_sample()
            # logger.info(f"Worker {pid} completed pre_sample")

            episode_count = 0
            step_count = 0
            last_log_time = time.time()
            
            while rl_logger.num_steps < min_batch_size:
                try:
                    # Log progress every 10 seconds
                    if time.time() - last_log_time > 10:
                        # logger.info(f"Worker {pid} progress: {rl_logger.num_steps}/{min_batch_size} steps, "
                        #            f"{episode_count} episodes")
                        monitor_resources(logger, pid, f"STEP_{step_count}")
                        last_log_time = time.time()
                    
                    # Environment reset with timeout
                    # logger.debug(f"Worker {pid} resetting environment")
                    obs_dict, info = self.env.reset()
                    state = self.preprocess_obs(obs_dict)
                    expert_idx = torch.tensor(0, dtype=torch.int64)  # Initialize expert index
                    
                    rl_logger.start_episode(self.env)
                    episode_count += 1

                    
                    for t in range(10000):
                        step_count += 1
                        
                        # Detailed logging every 1000 steps
                        # if step_count % 1000 == 0:
                            # logger.debug(f"Worker {pid} at step {step_count}")
                            
                        mean_action = self.mean_action or self.env.np_random.binomial(
                            1, 1 - self.noise_rate
                        )
                        
                        # Policy action selection
                        state_tensor = torch.from_numpy(state).to(self.dtype)
                        if isinstance(self.policy_net, PolicyMOE):
                            actions, expert_idx = self.policy_net.select_action(
                                torch.from_numpy(state).to(self.dtype), True
                            )
                            actions = actions[0].numpy()
                        elif isinstance(self.policy_net, PolicyMOEWithPrev):
                            expert_idx_oh = torch.nn.functional.one_hot(
                                expert_idx, num_classes=self.cfg.num_experts
                            ).float()
                            actions, expert_idx = self.policy_net.select_action(
                                torch.from_numpy(state).to(self.dtype),
                                expert_idx_oh,
                                True,
                            )
                            actions = actions[0].numpy()
                        else:
                            actions = self.policy_net.select_action(state_tensor, mean_action)[0].numpy()
                        
                        # Environment step
                        processed_actions = self.preprocess_actions(actions)
                        next_obs, reward, terminated, truncated, info = self.env.step(processed_actions)
                        
                        episode_done = terminated or truncated
                        next_state = self.preprocess_obs(next_obs)

                        rl_logger.step(self.env, reward, info)

                        mask = 0 if episode_done else 1
                        exp = 1 - mean_action
                        
                        if hasattr(self.policy_net, "experts"):
                            self.push_memory(
                                memory,
                                state.squeeze(),
                                actions,
                                mask,
                                next_state.squeeze(),
                                reward,
                                exp,
                                expert_idx.item()
                            )
                        else:
                            self.push_memory(
                                memory,
                                state.squeeze(),
                                actions,
                                mask,
                                next_state.squeeze(),
                                reward,
                                exp,
                            )

                        if pid == 0 and not self.headless:
                            self.env.render()
                            
                        if episode_done:
                            # logger.debug(f"Worker {pid} episode {episode_count} done after {t} steps")
                            break
                            
                        state = next_state

                    rl_logger.end_episode(self.env)
                    
                except Exception as e:
                    logger.error(f"Worker {pid} error in episode {episode_count}: {e}\n{traceback.format_exc()}")
                    monitor_resources(logger, pid, "ERROR")
                    # Try to continue if possible
                    continue
                    
        except Exception as e:
            error_msg = f"Worker {pid} failed: {e}\n{traceback.format_exc()}"
            logger.critical(error_msg)
            monitor_resources(logger, pid, "CRASH")
            error_queue.put((pid, str(e), traceback.format_exc()))
            
            if queue is not None:
                queue.put([pid, None, None, error_msg])
            return None
        
        # logger.info(f"Worker {pid} finished sampling")
        monitor_resources(logger, pid, "END")
        
        if rl_logger is not None:
            rl_logger.end_sampling()

        if queue is not None:
            # logger.info(f"Worker {pid} putting results in queue")
            queue.put([pid, memory, rl_logger, None])
            # logger.info(f"Worker {pid} done")
        else:
            return memory, rl_logger

    def sample(self, min_batch_size: int):
        """Sample with comprehensive debugging"""

        os.system("rm -rf /dev/shm/torch_*")

        import psutil
        shm = psutil.disk_usage('/dev/shm')
        self.main_logger.info(f"Shared memory usage before sampling: {shm.percent}%")
        if shm.percent > 80:
            os.system("rm -rf /dev/shm/*")
            self.main_logger.warning("Shared memory usage was high, cleared /dev/shm")

        self.main_logger.info(f"Starting sampling with {self.num_threads} threads")
        monitor_resources(self.main_logger, 'main', 'SAMPLE_START')
        
        # Check system resources before starting
        mem = psutil.virtual_memory()
        if mem.percent > 80:
            self.main_logger.warning(f"High memory usage before sampling: {mem.percent}%")
        
        t_start = time.time()
        to_test(*self.sample_modules)

        with to_cpu(*self.sample_modules):
            with torch.no_grad():
                thread_batch_size = int(math.floor(min_batch_size / self.num_threads))
                queue = multiprocessing.Queue()
                error_queue = multiprocessing.Queue()
                memories = [None] * self.num_threads
                loggers = [None] * self.num_threads
                workers = []

                # Start monitoring process
                monitor_proc = multiprocessing.Process(
                    target=self._monitor_workers,
                    args=(error_queue,)
                )
                monitor_proc.start()

                # Spawn workers
                for i in range(self.num_threads - 1):
                    worker_args = (i + 1, queue, thread_batch_size, error_queue)
                    worker = multiprocessing.Process(
                        target=self.sample_worker_wrapper, args=worker_args
                    )
                    worker.start()
                    workers.append((i + 1, worker))
                    # self.main_logger.info(f"Started worker {i + 1}, PID: {worker.pid}")

                # Sample in main process
                main_logger = setup_worker_logger(0, self.debug_dir)
                result = self.sample_worker(0, None, thread_batch_size, main_logger, error_queue)
                if result is None:
                    raise RuntimeError("Main process sampling failed")
                memories[0], loggers[0] = result

                # Monitor and collect results
                self._collect_worker_results(
                    queue, error_queue, workers, memories, loggers
                )

                # Stop monitoring
                error_queue.put(('STOP', None, None))
                monitor_proc.join(timeout=5)

                # Merge results
                if hasattr(self.policy_net, "experts"):
                    traj_batch = self.traj_cls(memories, has_expert_idx=True)
                else:
                    traj_batch = self.traj_cls(memories)
                logger = self.logger_rl_cls.merge(loggers)

        logger.sample_time = time.time() - t_start
        self.main_logger.info(f"Sampling completed in {logger.sample_time:.2f}s")
        return traj_batch, logger

    def _monitor_workers(self, error_queue: multiprocessing.Queue):
        """Separate process to monitor for errors"""
        logger = setup_worker_logger('monitor', self.debug_dir)
        # logger.info("Error monitor started")
        
        while True:
            try:
                item = error_queue.get(timeout=1)
                if item[0] == 'STOP':
                    break
                    
                pid, error, trace = item
                logger.critical(f"Worker {pid} error: {error}\n{trace}")
                
            except:
                continue
                
        # logger.info("Error monitor stopped")

    def _collect_worker_results(
        self, queue, error_queue, workers, memories, loggers
    ):
        """Collect results with detailed monitoring"""
        workers_completed = 0
        timeout_per_worker = 300
        check_interval = 50
        
        # self.main_logger.info(f"Waiting for {self.num_threads - 1} workers")
        
        while workers_completed < self.num_threads - 1:
            # Check worker status
            for pid, worker in workers:
                if not worker.is_alive() and memories[pid] is None:
                    self.main_logger.error(f"Worker {pid} (PID: {worker.pid}) died! "
                                         f"Exit code: {worker.exitcode}")
                    
                    # Check for errors
                    try:
                        error = error_queue.get(block=False)
                        if error[0] == pid:
                            self.main_logger.error(f"Worker {pid} error details: {error[1]}")
                    except:
                        pass
            
            try:
                result = queue.get(timeout=check_interval)
                pid, worker_memory, worker_logger, error = result
                
                if error is not None:
                    self.main_logger.error(f"Worker {pid} error: {error}")
                    raise RuntimeError(f"Worker {pid} failed")
                
                memories[pid] = worker_memory
                loggers[pid] = worker_logger
                workers_completed += 1
                # self.main_logger.info(f"Worker {pid} completed ({workers_completed}/{self.num_threads - 1})")
                
            except multiprocessing.Queue.Empty:
                # Log current status
                monitor_resources(self.main_logger, 'main', f'WAITING_{workers_completed}')
                continue

        # Cleanup workers
        for pid, worker in workers:
            worker.join(timeout=10)
            if worker.is_alive():
                self.main_logger.warning(f"Force terminating worker {pid}")
                worker.terminate()
                worker.join()

    def preprocess_obs(self, obs: Any) -> np.ndarray:
        obs = obs.reshape(1, -1)
        if self.clip_obs and self.obs_low is not None and self.obs_high is not None:
            return np.clip(obs, self.obs_low, self.obs_high)
        return obs

    def preprocess_actions(self, actions: np.ndarray) -> np.ndarray:
        actions = (
            int(actions)
            if self.policy_net.type == "discrete"
            else actions.astype(self.np_dtype)
        )
        if self.clip_actions:
            actions = rescale_actions(
                self.actions_low,
                self.actions_high,
                np.clip(actions, self.actions_low, self.actions_high),
            )
        return actions

    def push_memory(
        self,
        memory: Memory,
        state: np.ndarray,
        action: np.ndarray,
        mask: int,
        next_state: np.ndarray,
        reward: float,
        exploration_flag: float,
        expert_idx: Optional[int] = None
    ) -> None:
        if expert_idx is not None:
            memory.push(state, action, mask, next_state, reward, exploration_flag, expert_idx)
        else:
            memory.push(state, action, mask, next_state, reward, exploration_flag)

    def pre_sample(self):
        pass

    def set_noise_rate(self, noise_rate):
        self.noise_rate = noise_rate