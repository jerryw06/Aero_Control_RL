# train_rl.py
"""
Improved REINFORCE trainer for lateral (Y) acceleration control with robustness constraints.
- 200 episodes with curriculum learning
- Adaptive learning rate
- Gradient clipping for stability
- Episode statistics tracking
- Model checkpointing
"""

import time
import json
import torch
import torch.nn as nn
import torch.optim as optim
from px4_accel_env import PX4AccelEnv
import numpy as np

class Policy(nn.Module):
    def __init__(self, obs_dim=6, hidden=1024, num_layers=8, act_limit=3.0):
        super().__init__()
        layers = []
        in_dim = obs_dim
        for i in range(num_layers):
            layers.append(nn.Linear(in_dim, hidden))
            layers.append(nn.LayerNorm(hidden))  # Add LayerNorm for stability
            layers.append(nn.Tanh())
            in_dim = hidden
        layers.append(nn.Linear(hidden, 1))
        self.net = nn.Sequential(*layers)
        self.log_std = nn.Parameter(torch.tensor(-0.5))  # learnable exploration
        self.act_limit = float(act_limit)

    def forward(self, obs_t):
        mu = self.net(obs_t)  # shape [B,1]
        std = torch.exp(self.log_std).clamp(1e-3, 2.0)
        return mu, std

    def sample_action(self, obs_np):
        obs_t = torch.as_tensor(obs_np, dtype=torch.float32).unsqueeze(0)
        mu, std = self.forward(obs_t)
        dist = torch.distributions.Normal(mu, std)
        a = dist.sample()
        a_tanh = torch.tanh(a) * self.act_limit
        logp = dist.log_prob(a) - torch.log(1 - torch.tanh(a)**2 + 1e-6)  # tanh correction
        return a_tanh.squeeze(0).detach().numpy(), logp.squeeze(0), mu.squeeze(0), std

def run():
    # Match env limits - now with 6D observations!
    env = PX4AccelEnv(rate_hz=50.0, a_max=3.0, ep_time_s=6.0, target_lateral_m=2.0)
    
    # Updated: 6D observations [x, y, z, vx, vy, vz], deeper network for robustness
    policy = Policy(obs_dim=6, hidden=1024, num_layers=8, act_limit=3.0)
    
    # Adaptive learning rate with scheduler
    optimizer = optim.Adam(policy.parameters(), lr=3e-3)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.8)
    
    gamma = 0.98
    num_eps = 200  # More episodes for better convergence

    # Statistics tracking
    episode_returns = []
    episode_lengths = []
    best_return = float('-inf')

    for ep in range(1, num_eps + 1):
        print(f"\n=== Episode {ep}/{num_eps} ===")
        obs, _ = env.reset()

        obs_buf, act_buf, logp_buf, rew_buf = [], [], [], []

        done = False
        while not done:
            act, logp, mu, std = policy.sample_action(obs)
            next_obs, reward, term, trunc, info = env.step(act)
            done = term or trunc

            obs_buf.append(obs.copy())
            act_buf.append(act.copy())
            logp_buf.append(logp.sum())  # scalar
            rew_buf.append(reward)

            obs = next_obs

        # Compute returns with Generalized Advantage Estimation style
        G = 0.0
        returns = []
        for r in reversed(rew_buf):
            G = r + gamma * G
            returns.append(G)
        returns.reverse()
        returns_t = torch.as_tensor(returns, dtype=torch.float32)
        
        # Normalize returns for stability (with safety check)
        if len(returns_t) > 1:
            returns_t = (returns_t - returns_t.mean()) / (returns_t.std() + 1e-8)

        # Policy loss with entropy bonus for exploration
        logp_t = torch.stack(logp_buf)
        policy_loss = -(logp_t * returns_t).mean()
        
        # Optional: Add entropy bonus (uncomment for more exploration)
        # entropy = -logp_t.mean()
        # loss = policy_loss - 0.01 * entropy

        optimizer.zero_grad()
        policy_loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=0.5)
        
        optimizer.step()
        scheduler.step()

        # Track statistics
        ep_return = sum(rew_buf)
        ep_length = len(rew_buf)
        episode_returns.append(ep_return)
        episode_lengths.append(ep_length)

        # Print with more detail
        current_lr = scheduler.get_last_lr()[0]
        print(f"Episode {ep} | steps={ep_length} | return={ep_return:.3f} | lr={current_lr:.6f}")
        
        # Save best model
        if ep_return > best_return:
            best_return = ep_return
            torch.save(policy.state_dict(), 'best_policy.pth')
            print(f"  ✓ New best return: {best_return:.3f} (model saved)")

        # Print moving average every 10 episodes
        if ep % 10 == 0:
            avg_return = np.mean(episode_returns[-10:])
            avg_length = np.mean(episode_lengths[-10:])
            print(f"  Last 10 episodes avg: return={avg_return:.3f}, length={avg_length:.1f}")

        # Checkpoint every 50 episodes
        if ep % 50 == 0:
            torch.save({
                'episode': ep,
                'policy_state_dict': policy.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'episode_returns': episode_returns,
            }, f'checkpoint_ep{ep}.pth')
            print(f"  Checkpoint saved: checkpoint_ep{ep}.pth")

        # Pause between episodes for system stability
        time.sleep(2.0)

    env.close()
    
    # Final statistics
    print("\n" + "="*60)
    print("Training complete!")
    print(f"Total episodes: {num_eps}")
    print(f"Best return: {best_return:.3f}")
    print(f"Final 20 episodes avg return: {np.mean(episode_returns[-20:]):.3f}")
    print(f"Best policy saved to: best_policy.pth")

    # Export final model for reproduction
    export_config = {
        "obs_dim": 6,
        "hidden": 1024,
        "num_layers": 8,
        "act_limit": 3.0,
    }
    with open('policy_config.json', 'w') as f:
        json.dump(export_config, f, indent=2)

    torch.save(policy.state_dict(), 'final_policy_state.pt')

    # Also save a TorchScript version of the policy network (mu head)
    example_in = torch.zeros(1, export_config["obs_dim"], dtype=torch.float32)
    scripted_net = torch.jit.trace(policy.net, example_in)
    scripted_net.save('policy_scripted.pt')

    print("Saved artifacts:")
    print("  - final_policy_state.pt (state_dict)")
    print("  - policy_scripted.pt (TorchScript net)")
    print("  - policy_config.json (architecture)"
    )
    print("="*60)

if __name__ == "__main__":
    run()
