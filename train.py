"""Headless A3C training script."""
import os
import torch
import torch.multiprocessing as mp
import time

from model import A3C_Labyrinth_Net
from shared_optim import SharedRMSprop
from worker import worker

# M2 MacBook has 8 performance cores - match workers to CPU count
NUM_WORKERS = min(os.cpu_count() or 8, 8)
MAX_FRAMES = 1_000_000  # 1M for faster iteration; increase to 10M for full training


def main():
    mp.set_start_method('spawn', force=True)
    
    try:
        import gymnasium as gym
        import miniworld
        temp_env = gym.make("MiniWorld-Maze-v0")
        action_space = temp_env.action_space.n
        del temp_env
    except Exception as e:
        print(f"MiniWorld initialization failed: {e}")
        raise SystemExit(1)
    
    global_model = A3C_Labyrinth_Net(action_space)
    global_model.share_memory()
    optimizer = SharedRMSprop(global_model.parameters(), lr=1e-4)
    
    global_counter = mp.Value('i', 0)
    global_episodes = mp.Value('i', 0)
    lock = mp.Lock()
    
    processes = []
    print(f"Starting {NUM_WORKERS} workers...")
    
    for rank in range(NUM_WORKERS):
        p = mp.Process(target=worker, args=(
            rank, global_model, optimizer, global_counter, global_episodes,
            MAX_FRAMES, lock, None
        ))
        p.start()
        processes.append(p)
    
    time.sleep(2)
    alive = sum(1 for p in processes if p.is_alive())
    print(f"Workers alive: {alive}/{NUM_WORKERS}")
    
    if alive == 0:
        print("No workers started.")
        return
    
    try:
        while any(p.is_alive() for p in processes):
            time.sleep(10)
            frames = global_counter.value
            episodes = global_episodes.value
            print(f"Frames: {frames:,} ({100*frames/MAX_FRAMES:.1f}%) | Episodes: {episodes:,}")
    except KeyboardInterrupt:
        print("\nStopping...")
        for p in processes:
            p.terminate()
    
    print("Done.")


if __name__ == '__main__':
    main()
