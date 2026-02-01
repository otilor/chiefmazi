"""A3C Training Dashboard with Gradio."""
import gradio as gr
import torch.multiprocessing as mp
import numpy as np
import pandas as pd
import time
import gymnasium as gym

from worker import worker
from model import A3C_Labyrinth_Net
from shared_optim import SharedRMSprop

NUM_WORKERS = 16
MAX_FRAMES = 10_000_000


class TrainingState:
    def __init__(self, manager):
        self.data = manager.dict({
            'frame': None, 'episode_score': 0.0, 'total_frames': 0,
            'episodes': 0, 'last_episode_score': 0, 'policy_loss': 0.0,
            'value_loss': 0.0, 'entropy': 0.0, 'action_probs': None,
        })
        self.episode_scores = manager.list()
        self.score_history = manager.list()
        self.lock = manager.Lock()
        
    def update(self, **kwargs):
        with self.lock:
            for k, v in kwargs.items():
                if k == 'episode_scores':
                    self.episode_scores.append(v)
                    while len(self.episode_scores) > 100:
                        self.episode_scores.pop(0)
                elif k == 'score_history':
                    self.score_history.append(v)
                    while len(self.score_history) > 500:
                        self.score_history.pop(0)
                else:
                    self.data[k] = v
                    
    def get(self, key, default=None):
        return self.data.get(key, default)
    
    def get_scores_df(self):
        scores = list(self.score_history)
        if not scores:
            return pd.DataFrame({'step': [0], 'score': [0]})
        return pd.DataFrame({'step': range(len(scores)), 'score': scores})
    
    def get_episode_df(self):
        scores = list(self.episode_scores)[-50:]
        if not scores:
            return pd.DataFrame({'episode': [0], 'score': [0]})
        return pd.DataFrame({'episode': range(len(scores)), 'score': scores})


def main():
    mp.freeze_support()
    mp.set_start_method('spawn', force=True)
    
    manager = mp.Manager()
    state = TrainingState(manager)
    
    miniworld_ok = True
    action_space = 3
    action_names = ['Left', 'Right', 'Forward']
    
    try:
        import miniworld
        temp_env = gym.make("MiniWorld-Maze-v0")
        action_space = temp_env.action_space.n
        action_names = ['Left', 'Right', 'Forward'][:action_space]
        del temp_env
    except Exception as e:
        miniworld_ok = False
        print(f"MiniWorld failed: {e}")
    
    global_model = A3C_Labyrinth_Net(action_space)
    global_model.share_memory()
    optimizer = SharedRMSprop(global_model.parameters(), lr=1e-4)
    
    global_counter = mp.Value('i', 0)
    global_episodes = mp.Value('i', 0)
    lock = mp.Lock()
    
    processes = []
    if miniworld_ok:
        print(f"Starting {NUM_WORKERS} workers...")
        for rank in range(NUM_WORKERS):
            p = mp.Process(target=worker, args=(
                rank, global_model, optimizer, global_counter, global_episodes,
                MAX_FRAMES, lock, state if rank == 0 else None
            ))
            p.start()
            processes.append(p)
    
    last_frames, last_time, fps = [0], [time.time()], [0.0]
    
    def calc_fps():
        frames = state.get('total_frames', 0)
        now = time.time()
        dt = now - last_time[0]
        if dt > 0.5:
            fps[0] = (frames - last_frames[0]) / dt
            last_frames[0], last_time[0] = frames, now
        return fps[0]

    def get_frame():
        frame = state.get('frame')
        return frame if frame is not None else np.zeros((84, 84, 3), dtype=np.uint8)
    
    def get_stats():
        return f"""| Metric | Value |
|--------|-------|
| Frames | {state.get('total_frames', 0):,} |
| Episodes | {state.get('episodes', 0):,} |
| FPS | {calc_fps():.0f} |
| Current Score | {state.get('episode_score', 0):.1f} |
| Last Episode | {state.get('last_episode_score', 0):.1f} |"""

    def get_losses():
        return f"""| Loss | Value |
|------|-------|
| Policy | {state.get('policy_loss', 0):.4f} |
| Value | {state.get('value_loss', 0):.4f} |
| Entropy | {state.get('entropy', 0):.4f} |"""
    
    def get_actions():
        probs = state.get('action_probs')
        if probs is not None and len(probs) > 0:
            bars = [f"{action_names[i]}: {'█' * int(p*20):<20} {p:.0%}" 
                    for i, p in enumerate(probs[:len(action_names)])]
            return "```\n" + "\n".join(bars) + "\n```"
        return "Waiting..."
    
    def get_progress():
        frames = state.get('total_frames', 0)
        return f"**Progress:** {frames:,} / {MAX_FRAMES:,} ({100*frames/MAX_FRAMES:.1f}%)"

    with gr.Blocks(title="A3C Dashboard") as demo:
        gr.Markdown("# A3C Labyrinth Training")
        
        if not miniworld_ok:
            gr.Markdown("⚠️ **MiniWorld unavailable.** No display detected.")
        
        with gr.Row():
            with gr.Column(scale=1):
                agent_view = gr.Image(label="Agent View", height=250)
                stats = gr.Markdown()
                losses = gr.Markdown()
                actions = gr.Markdown()
            
            with gr.Column(scale=2):
                score_plot = gr.LinePlot(x="step", y="score", title="Training Score", height=250)
                episode_plot = gr.BarPlot(x="episode", y="score", title="Episode Scores", height=200)
        
        progress = gr.Markdown()
        
        timer = gr.Timer(0.1)
        timer.tick(fn=get_frame, outputs=agent_view, show_progress="hidden")
        timer.tick(fn=get_stats, outputs=stats, show_progress="hidden")
        timer.tick(fn=get_actions, outputs=actions, show_progress="hidden")
        
        slow_timer = gr.Timer(0.5)
        slow_timer.tick(fn=get_losses, outputs=losses, show_progress="hidden")
        slow_timer.tick(fn=state.get_scores_df, outputs=score_plot, show_progress="hidden")
        slow_timer.tick(fn=state.get_episode_df, outputs=episode_plot, show_progress="hidden")
        slow_timer.tick(fn=get_progress, outputs=progress, show_progress="hidden")
    
    print("Dashboard: http://localhost:7860")
    try:
        demo.launch(server_name="0.0.0.0", server_port=7860, quiet=True, theme=gr.themes.Soft())
    finally:
        for p in processes:
            p.terminate()


if __name__ == '__main__':
    main()
