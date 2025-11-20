#include "px4_accel_env.hpp"
#include "policy_network.hpp"
#include <torch/torch.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <thread>
#include <iomanip>
#include <algorithm>
#include <nlohmann/json.hpp>

using json = nlohmann::json;
using namespace std::chrono_literals;

// PPO hyperparameters
struct PPOConfig {
    double gamma = 0.99;           // Discount factor
    double gae_lambda = 0.95;      // GAE lambda for advantage estimation
    double clip_epsilon = 0.2;     // PPO clipping parameter
    int update_epochs = 10;        // Number of epochs to update per batch
    int mini_batch_size = 64;      // Mini-batch size for updates
    double value_loss_coef = 0.5;  // Value loss coefficient
    double entropy_coef = 0.01;    // Entropy bonus coefficient
    double max_grad_norm = 0.5;    // Gradient clipping
    double learning_rate = 3e-4;   // Learning rate
};

// Compute Generalized Advantage Estimation (GAE)
std::pair<torch::Tensor, torch::Tensor> compute_gae(
    const std::vector<double>& rewards,
    const std::vector<torch::Tensor>& values,
    double gamma,
    double gae_lambda) {
    
    int T = rewards.size();
    std::vector<double> advantages(T, 0.0);
    std::vector<double> returns(T, 0.0);
    
    double gae = 0.0;
    for (int t = T - 1; t >= 0; --t) {
        double value_t = values[t].item<double>();
        double value_next = (t + 1 < T) ? values[t + 1].item<double>() : 0.0;
        
        // TD error: δ_t = r_t + γV(s_{t+1}) - V(s_t)
        double delta = rewards[t] + gamma * value_next - value_t;
        
        // GAE: A_t = δ_t + (γλ)A_{t+1}
        gae = delta + gamma * gae_lambda * gae;
        advantages[t] = gae;
        
        // Returns: R_t = A_t + V(s_t)
        returns[t] = gae + value_t;
    }
    
    auto adv_tensor = torch::tensor(advantages, torch::kFloat32);
    auto ret_tensor = torch::tensor(returns, torch::kFloat32);
    
    // Normalize advantages
    adv_tensor = (adv_tensor - adv_tensor.mean()) / (adv_tensor.std() + 1e-8);
    
    return {adv_tensor, ret_tensor};
}

void save_checkpoint(Policy& policy, 
                     ValueNetwork& value_net,
                     torch::optim::Adam& optimizer,
                     int episode,
                     const std::vector<double>& episode_returns,
                     const std::string& filename) {
    torch::serialize::OutputArchive archive;
    policy->save(archive);
    archive.save_to(filename);
    
    torch::serialize::OutputArchive value_archive;
    value_net->save(value_archive);
    value_archive.save_to(filename + ".value");
    
    std::cout << "  Checkpoint saved: " << filename << std::endl;
}

int main(int argc, char** argv) {
    try {
        std::cout << "=== PPO Training for PX4 Lateral Acceleration Control ===" << std::endl;
        
        // PPO configuration
        PPOConfig config;
        
        // Match Python env configuration
        PX4AccelEnv env(200.0, 3.0, 6.0, 2.0, 2.0);
        const int obs_dim = env.get_observation_dim();
        const int act_dim = env.get_action_dim();
        
        // Create policy network (actor)
        Policy policy(obs_dim, 1024, 8, 1.0, act_dim);
        
        // Create value network (critic)
        ValueNetwork value_net(obs_dim, 512, 4);
        
        // Create optimizers for actor and critic
        torch::optim::Adam policy_optimizer(policy->parameters(), 
                                             torch::optim::AdamOptions(config.learning_rate));
        torch::optim::Adam value_optimizer(value_net->parameters(), 
                                            torch::optim::AdamOptions(config.learning_rate));
        
        const int num_eps = 200;
        
        std::vector<double> episode_returns;
        std::vector<int> episode_lengths;
        double best_return = -std::numeric_limits<double>::infinity();
        
        for (int ep = 1; ep <= num_eps; ++ep) {
            std::cout << "\n=== Episode " << ep << "/" << num_eps << " ===" << std::endl;
            
            auto obs = env.reset();
            
            std::vector<torch::Tensor> obs_buf;
            std::vector<torch::Tensor> act_buf;
            std::vector<torch::Tensor> logp_buf;
            std::vector<torch::Tensor> val_buf;
            std::vector<double> rew_buf;
            
            // Collect episode data
            bool done = false;
            while (!done) {
                torch::NoGradGuard no_grad;  // Don't track gradients during rollout
                
                auto [act, logp, mu, std] = policy->sample_action(obs);
                auto obs_tensor = obs.to(torch::kFloat32).unsqueeze(0);
                auto value = value_net->forward(obs_tensor);
                
                auto result = env.step(act);
                done = result.terminated || result.truncated;
                
                obs_buf.push_back(obs.clone());
                act_buf.push_back(act.clone());
                logp_buf.push_back(logp.clone());
                val_buf.push_back(value.squeeze(0).clone());
                rew_buf.push_back(result.reward);
                
                obs = result.observation;
            }
            
            // Compute GAE advantages and returns
            auto [advantages, returns] = compute_gae(rew_buf, val_buf, config.gamma, config.gae_lambda);
            
            // Stack tensors for batch processing
            auto obs_batch = torch::stack(obs_buf);
            auto act_batch = torch::stack(act_buf);
            auto old_logp_batch = torch::stack(logp_buf).detach();
            
            int batch_size = obs_batch.size(0);
            
            // PPO update for multiple epochs
            double total_policy_loss = 0.0;
            double total_value_loss = 0.0;
            double total_entropy = 0.0;
            int num_updates = 0;
            
            for (int epoch = 0; epoch < config.update_epochs; ++epoch) {
                // Create mini-batches
                auto indices = torch::randperm(batch_size, torch::kLong);
                
                for (int start = 0; start < batch_size; start += config.mini_batch_size) {
                    int end = std::min(start + config.mini_batch_size, batch_size);
                    if (end - start < 2) continue;  // Skip tiny batches
                    
                    auto mb_indices = indices.slice(0, start, end);
                    auto mb_obs = obs_batch.index_select(0, mb_indices);
                    auto mb_act = act_batch.index_select(0, mb_indices);
                    auto mb_old_logp = old_logp_batch.index_select(0, mb_indices);
                    auto mb_adv = advantages.index_select(0, mb_indices);
                    auto mb_ret = returns.index_select(0, mb_indices);
                    
                    // Compute current log probabilities
                    auto new_logp = policy->compute_log_prob(mb_obs, mb_act);
                    
                    // Compute ratio for PPO clipping
                    auto ratio = torch::exp(new_logp - mb_old_logp);
                    
                    // Clipped surrogate objective
                    auto surr1 = ratio * mb_adv;
                    auto surr2 = torch::clamp(ratio, 1.0 - config.clip_epsilon, 1.0 + config.clip_epsilon) * mb_adv;
                    auto policy_loss = -torch::min(surr1, surr2).mean();
                    
                    // Value loss (MSE)
                    auto values = value_net->forward(mb_obs);
                    auto value_loss = torch::mse_loss(values, mb_ret);
                    
                    // Entropy bonus (encourage exploration)
                    auto [mu, std] = policy->forward(mb_obs);
                    auto entropy = (torch::log(std * std::sqrt(2 * M_PI * M_E))).mean();
                    
                    // Combined loss
                    auto loss = policy_loss + config.value_loss_coef * value_loss - config.entropy_coef * entropy;
                    
                    // Update policy
                    policy_optimizer.zero_grad();
                    value_optimizer.zero_grad();
                    loss.backward();
                    
                    torch::nn::utils::clip_grad_norm_(policy->parameters(), config.max_grad_norm);
                    torch::nn::utils::clip_grad_norm_(value_net->parameters(), config.max_grad_norm);
                    
                    policy_optimizer.step();
                    value_optimizer.step();
                    
                    total_policy_loss += policy_loss.item<double>();
                    total_value_loss += value_loss.item<double>();
                    total_entropy += entropy.item<double>();
                    num_updates++;
                }
            }
            
            // Track statistics
            double ep_return = 0.0;
            for (double r : rew_buf) ep_return += r;
            int ep_length = rew_buf.size();
            episode_returns.push_back(ep_return);
            episode_lengths.push_back(ep_length);
            
            // Print episode info with PPO metrics
            double avg_policy_loss = (num_updates > 0) ? total_policy_loss / num_updates : 0.0;
            double avg_value_loss = (num_updates > 0) ? total_value_loss / num_updates : 0.0;
            double avg_entropy = (num_updates > 0) ? total_entropy / num_updates : 0.0;
            
            std::cout << "Episode " << ep << " | steps=" << ep_length 
                      << " | return=" << std::fixed << std::setprecision(3) << ep_return 
                      << " | π_loss=" << std::setprecision(4) << avg_policy_loss
                      << " | v_loss=" << std::setprecision(4) << avg_value_loss
                      << " | entropy=" << std::setprecision(4) << avg_entropy << std::endl;
            
            // Save best model
            if (ep_return > best_return) {
                best_return = ep_return;
                torch::save(policy, "best_policy.pt");
                torch::save(value_net, "best_value.pt");
                std::cout << "  ✓ New best return: " << std::fixed << std::setprecision(3) 
                          << best_return << " (models saved)" << std::endl;
            }
            
            // Print moving average every 10 episodes
            if (ep % 10 == 0) {
                double avg_return = 0.0;
                double avg_length = 0.0;
                int start_idx = std::max(0, ep - 10);
                for (int i = start_idx; i < ep; ++i) {
                    avg_return += episode_returns[i];
                    avg_length += episode_lengths[i];
                }
                avg_return /= (ep - start_idx);
                avg_length /= (ep - start_idx);
                std::cout << "  Last 10 episodes avg: return=" << std::fixed << std::setprecision(3) 
                          << avg_return << ", length=" << std::setprecision(1) << avg_length << std::endl;
            }
            
            // Checkpoint every 50 episodes
            if (ep % 50 == 0) {
                std::string checkpoint_name = "checkpoint_ep" + std::to_string(ep) + ".pt";
                save_checkpoint(policy, value_net, policy_optimizer, ep, episode_returns, checkpoint_name);
            }
            
            // Pause between episodes for system stability
            std::this_thread::sleep_for(2s);
        }
        
        env.close();
        
        // Final statistics
        std::cout << "\n" << std::string(60, '=') << std::endl;
        std::cout << "PPO Training complete!" << std::endl;
        std::cout << "Total episodes: " << num_eps << std::endl;
        std::cout << "Best return: " << std::fixed << std::setprecision(3) << best_return << std::endl;
        
        double final_avg = 0.0;
        int final_count = std::min(20, static_cast<int>(episode_returns.size()));
        for (int i = episode_returns.size() - final_count; i < static_cast<int>(episode_returns.size()); ++i) {
            final_avg += episode_returns[i];
        }
        final_avg /= final_count;
        std::cout << "Final " << final_count << " episodes avg return: " 
                  << std::setprecision(3) << final_avg << std::endl;
        std::cout << "Best models saved to: best_policy.pt, best_value.pt" << std::endl;
        
        // Export configuration
        json ppo_config;
        ppo_config["algorithm"] = "PPO";
        ppo_config["obs_dim"] = 6;
        ppo_config["act_dim"] = 1;
        ppo_config["policy_hidden"] = 1024;
        ppo_config["policy_layers"] = 8;
        ppo_config["value_hidden"] = 512;
        ppo_config["value_layers"] = 4;
        ppo_config["act_limit"] = 3.0;
        ppo_config["gamma"] = config.gamma;
        ppo_config["gae_lambda"] = config.gae_lambda;
        ppo_config["clip_epsilon"] = config.clip_epsilon;
        ppo_config["learning_rate"] = config.learning_rate;
        
        std::ofstream config_file("policy_config.json");
        config_file << ppo_config.dump(2) << std::endl;
        config_file.close();
        
        // Save final models
        torch::save(policy, "final_policy.pt");
        torch::save(value_net, "final_value.pt");

        std::cout << "Saved artifacts:" << std::endl;
        std::cout << "  - final_policy.pt (actor network)" << std::endl;
        std::cout << "  - final_value.pt (critic network)" << std::endl;
        std::cout << "  - policy_config.json (PPO configuration)" << std::endl;
        std::cout << std::string(60, '=') << std::endl;
        
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}
