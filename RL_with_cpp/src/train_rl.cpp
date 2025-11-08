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

void save_checkpoint(Policy& policy, 
                     torch::optim::Adam& optimizer,
                     torch::optim::StepLR& scheduler,
                     int episode,
                     const std::vector<double>& episode_returns,
                     const std::string& filename) {
    torch::serialize::OutputArchive archive;
    policy->save(archive);
    archive.save_to(filename);
    std::cout << "  Checkpoint saved: " << filename << std::endl;
}

int main(int argc, char** argv) {
    try {
        std::cout << "=== C++ RL Training for PX4 Lateral Acceleration Control ===" << std::endl;
        
        // Match Python env configuration
        PX4AccelEnv env(200.0, 3.0, 6.0, 2.0, 2.0);
        const int obs_dim = env.get_observation_dim();
        const int act_dim = env.get_action_dim();
        
        // Create policy network (1024 hidden, 8 layers)
        Policy policy(obs_dim, 1024, 8, 1.0, act_dim);
        
        // Optimizer with adaptive learning rate
        torch::optim::Adam optimizer(policy->parameters(), torch::optim::AdamOptions(3e-3));
        torch::optim::StepLR scheduler(optimizer, /*step_size=*/50, /*gamma=*/0.8);
        
        const double gamma = 0.98;
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
            std::vector<double> rew_buf;
            
            bool done = false;
            while (!done) {
                auto [act, logp, mu, std] = policy->sample_action(obs);
                auto result = env.step(act);
                done = result.terminated || result.truncated;
                
                obs_buf.push_back(obs.clone());
                act_buf.push_back(act.clone());
                // Store full log probability (already per action scalar due to squash correction)
                logp_buf.push_back(logp.clone());
                rew_buf.push_back(result.reward);
                
                obs = result.observation;
            }
            
            // Compute returns (discounted rewards)
            std::vector<double> returns;
            double G = 0.0;
            for (int i = rew_buf.size() - 1; i >= 0; --i) {
                G = rew_buf[i] + gamma * G;
                returns.push_back(G);
            }
            std::reverse(returns.begin(), returns.end());
            
            auto returns_t = torch::tensor(returns, torch::kFloat32);
            
            // Normalize returns for stability
            if (returns_t.numel() > 1) {
                returns_t = (returns_t - returns_t.mean()) / (returns_t.std() + 1e-8);
            }
            
            // Policy loss
            // Stack log probs preserving gradient
            auto logp_t = torch::stack(logp_buf);
            if (!logp_t.requires_grad()) {
                std::cout << "[WARN] logp_t has no grad; ensure sample_action keeps autograd path." << std::endl;
            }
            auto policy_loss = -(logp_t * returns_t).mean();
            
            optimizer.zero_grad();
            policy_loss.backward();
            
            // Gradient clipping for stability
            torch::nn::utils::clip_grad_norm_(policy->parameters(), 0.5);
            
            optimizer.step();
            scheduler.step();
            
            // Track statistics
            double ep_return = 0.0;
            for (double r : rew_buf) ep_return += r;
            int ep_length = rew_buf.size();
            episode_returns.push_back(ep_return);
            episode_lengths.push_back(ep_length);
            
            // Print episode info (retrieve LR from optimizer options in C++ API)
            double current_lr = 0.0;
            if (!optimizer.param_groups().empty()) {
                auto &opts = static_cast<torch::optim::AdamOptions&>(optimizer.param_groups()[0].options());
                current_lr = opts.lr();
            }
            std::cout << "Episode " << ep << " | steps=" << ep_length 
                      << " | return=" << std::fixed << std::setprecision(3) << ep_return 
                      << " | lr=" << std::setprecision(6) << current_lr << std::endl;
            
            // Save best model
            if (ep_return > best_return) {
                best_return = ep_return;
                torch::save(policy, "best_policy.pt");
                std::cout << "  ✓ New best return: " << std::fixed << std::setprecision(3) 
                          << best_return << " (model saved)" << std::endl;
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
                save_checkpoint(policy, optimizer, scheduler, ep, episode_returns, checkpoint_name);
            }
            
            // Pause between episodes for system stability
            std::this_thread::sleep_for(2s);
        }
        
        env.close();
        
        // Final statistics
        std::cout << "\n" << std::string(60, '=') << std::endl;
        std::cout << "Training complete!" << std::endl;
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
        std::cout << "Best policy saved to: best_policy.pt" << std::endl;
        
        // Export configuration
        json config;
        config["obs_dim"] = 6;
        config["hidden"] = 1024;
        config["num_layers"] = 8;
        config["act_limit"] = 3.0;
        
        std::ofstream config_file("policy_config.json");
        config_file << config.dump(2) << std::endl;
        config_file.close();
        
    // Save final model
        torch::save(policy, "final_policy.pt");

        std::cout << "Saved artifacts:" << std::endl;
        std::cout << "  - final_policy.pt (full model)" << std::endl;
        std::cout << "  - policy_config.json (architecture)" << std::endl;
        std::cout << std::string(60, '=') << std::endl;
        
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}
