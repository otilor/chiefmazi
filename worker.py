"""A3C Worker implementing Algorithm S3 from Mnih et al. 2016."""
import torch
import torch.nn.functional as F
import gymnasium as gym
import numpy as np
from torchvision import transforms
from env_wrapper import LabyrinthWrapper
from model import A3C_Labyrinth_Net

GAMMA = 0.99
BETA = 0.01
T_MAX = 20  # Larger n-step reduces sync overhead (paper used 5, but 20 is faster)
GRAD_CLIP = 40.0


def worker(rank, global_model, optimizer, global_counter, global_episodes, 
           max_frames, lock, ui_state=None):
    try:
        _run_worker(rank, global_model, optimizer, global_counter, global_episodes,
                    max_frames, lock, ui_state)
    except Exception as e:
        import traceback
        print(f"[Worker {rank}] Error: {e}")
        traceback.print_exc()


def _run_worker(rank, global_model, optimizer, global_counter, global_episodes, 
                max_frames, lock, ui_state=None):
    import miniworld
    
    env = gym.make("MiniWorld-Maze-v0", view='agent', render_mode='rgb_array')
    env = LabyrinthWrapper(env)
    
    preprocess = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((84, 84)),
        transforms.ToTensor()
    ])

    local_model = A3C_Labyrinth_Net(env.action_space.n)
    hx = torch.zeros(1, 256)
    cx = torch.zeros(1, 256)

    state, _ = env.reset()
    state_tensor = preprocess(state).unsqueeze(0)
    episode_score = 0.0
    step_in_episode = 0

    while True:
        with lock:
            if global_counter.value >= max_frames:
                break
        
        local_model.load_state_dict(global_model.state_dict())
        hx, cx = hx.detach(), cx.detach()

        values, log_probs, rewards, entropies = [], [], [], []
        done = False
        
        for _ in range(T_MAX):
            value, logit, (hx, cx) = local_model(state_tensor, (hx, cx))
            
            prob = F.softmax(logit, dim=1)
            log_prob = F.log_softmax(logit, dim=1)
            entropy = -(log_prob * prob).sum(1, keepdim=True)
            
            action = prob.multinomial(num_samples=1).detach()
            next_state, reward, terminated, truncated, _ = env.step(action.item())
            done = terminated or truncated
            
            episode_score += reward
            step_in_episode += 1
            
            with lock:
                global_counter.value += 1
                current_frame = global_counter.value

            if ui_state is not None and step_in_episode % 3 == 0:
                _update_ui(ui_state, next_state, episode_score, current_frame, prob, entropy)

            values.append(value)
            log_probs.append(log_prob.gather(1, action))
            rewards.append(reward)
            entropies.append(entropy)
            
            state_tensor = preprocess(next_state).unsqueeze(0)
            
            if done:
                with lock:
                    global_episodes.value += 1
                    num_episodes = global_episodes.value
                
                if ui_state is not None:
                    _update_ui_episode(ui_state, num_episodes, episode_score)
                
                hx, cx = torch.zeros(1, 256), torch.zeros(1, 256)
                episode_score = 0.0
                step_in_episode = 0
                state, _ = env.reset()
                state_tensor = preprocess(state).unsqueeze(0)
                break

        R = torch.zeros(1, 1)
        if not done:
            with torch.no_grad():
                value, _, _ = local_model(state_tensor, (hx, cx))
                R = value

        policy_loss = torch.zeros(1, 1)
        value_loss = torch.zeros(1, 1)
        
        for i in reversed(range(len(rewards))):
            R = rewards[i] + GAMMA * R
            advantage = R - values[i]
            value_loss = value_loss + 0.5 * advantage.pow(2)
            policy_loss = policy_loss - (log_probs[i] * advantage.detach() + BETA * entropies[i])

        optimizer.zero_grad()
        (policy_loss + 0.5 * value_loss).backward()
        torch.nn.utils.clip_grad_norm_(local_model.parameters(), GRAD_CLIP)

        for lp, gp in zip(local_model.parameters(), global_model.parameters()):
            if gp.grad is None:
                gp.grad = lp.grad.clone()
            else:
                gp.grad = gp.grad + lp.grad

        optimizer.step()
        
        if ui_state is not None:
            _update_ui_losses(ui_state, policy_loss.item(), value_loss.item())

    env.close()


def _update_ui(ui_state, frame, score, frames, prob, entropy):
    try:
        if frame.dtype != np.uint8:
            frame = (frame * 255).astype(np.uint8) if frame.max() <= 1.0 else frame.astype(np.uint8)
        ui_state.update(
            frame=frame,
            episode_score=score,
            total_frames=frames,
            action_probs=prob.detach().cpu().numpy().flatten(),
            entropy=entropy.item(),
            score_history=score,
        )
    except Exception:
        pass


def _update_ui_episode(ui_state, episodes, score):
    try:
        ui_state.update(episodes=episodes, last_episode_score=score, episode_scores=score)
    except Exception:
        pass


def _update_ui_losses(ui_state, policy_loss, value_loss):
    try:
        ui_state.update(policy_loss=policy_loss, value_loss=value_loss)
    except Exception:
        pass
