#include "policy_network.hpp"
#include <cmath>

PolicyImpl::PolicyImpl(int obs_dim, int hidden, int num_layers, double act_limit, int act_dim)
    : act_limit_(act_limit),
      act_dim_(act_dim)
{
    net_ = torch::nn::Sequential();
    int in_dim = obs_dim;
    for (int i = 0; i < num_layers; ++i) {
        net_->push_back(torch::nn::Linear(in_dim, hidden));
        net_->push_back(torch::nn::LayerNorm(torch::nn::LayerNormOptions({hidden})));
        net_->push_back(torch::nn::Tanh());
        in_dim = hidden;
    }
    net_->push_back(torch::nn::Linear(hidden, act_dim_));
    register_module("net", net_);

    // Learnable log_std parameter per action dimension
    log_std_ = register_parameter("log_std", torch::full({act_dim_}, -0.5f));
}

std::pair<torch::Tensor, torch::Tensor> PolicyImpl::forward(torch::Tensor obs_t) {
    auto mu = net_->forward(obs_t);
    auto std = torch::exp(log_std_).clamp(1e-3, 2.0);
    if (std.dim() == 1) {
        std = std.unsqueeze(0);
    }
    std = std.expand_as(mu);
    return {mu, std};
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> 
PolicyImpl::sample_action(torch::Tensor obs_np) {
    // Ensure input is float32 and has batch dimension
    auto obs_t = obs_np.to(torch::kFloat32).unsqueeze(0);
    
    auto [mu, std] = forward(obs_t);
    
    // Sample from Normal distribution: a ~ N(mu, std^2)
    auto a = torch::randn_like(mu) * std + mu;
    
    // Apply tanh and scale
    auto a_tanh = torch::tanh(a) * act_limit_;
    
    // Compute log probability with tanh correction
    auto log_prob = -0.5 * torch::pow((a - mu) / std, 2)
                    - 0.5 * std::log(2 * M_PI) - torch::log(std);
    log_prob = log_prob - torch::log(1 - torch::pow(torch::tanh(a), 2) + 1e-6);
    log_prob = log_prob.sum(-1, true);
    
    return {a_tanh.squeeze(0), log_prob.squeeze(0), mu.squeeze(0), std.squeeze(0)};
}

torch::Tensor PolicyImpl::compute_log_prob(torch::Tensor obs_t, torch::Tensor act_t) {
    // obs_t: [batch, obs_dim], act_t: [batch, act_dim]
    auto [mu, std] = forward(obs_t);
    
    // Reverse tanh scaling to get pre-tanh action
    auto act_scaled = act_t / act_limit_;
    auto a = torch::atanh(act_scaled.clamp(-0.999, 0.999));
    
    // Compute log probability
    auto log_prob = -0.5 * torch::pow((a - mu) / std, 2)
                    - 0.5 * std::log(2 * M_PI) - torch::log(std);
    log_prob = log_prob - torch::log(1 - torch::pow(act_scaled, 2) + 1e-6);
    log_prob = log_prob.sum(-1, true);
    
    return log_prob;
}

// Value Network Implementation
ValueNetworkImpl::ValueNetworkImpl(int obs_dim, int hidden, int num_layers) {
    net_ = torch::nn::Sequential();
    int in_dim = obs_dim;
    for (int i = 0; i < num_layers; ++i) {
        net_->push_back(torch::nn::Linear(in_dim, hidden));
        net_->push_back(torch::nn::LayerNorm(torch::nn::LayerNormOptions({hidden})));
        net_->push_back(torch::nn::Tanh());
        in_dim = hidden;
    }
    net_->push_back(torch::nn::Linear(hidden, 1));
    register_module("net", net_);
}

torch::Tensor ValueNetworkImpl::forward(torch::Tensor obs_t) {
    return net_->forward(obs_t).squeeze(-1);
}
