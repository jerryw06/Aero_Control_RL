#include "policy_network.hpp"
#include <cmath>

PolicyImpl::PolicyImpl(int obs_dim, int hidden, int num_layers, double act_limit)
    : act_limit_(act_limit)
{
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

    // Learnable log_std parameter (scalar)
    log_std_ = register_parameter("log_std", torch::tensor(-0.5, torch::kFloat32));
}

std::pair<torch::Tensor, torch::Tensor> PolicyImpl::forward(torch::Tensor obs_t) {
    auto mu = net_->forward(obs_t);
    auto std = torch::exp(log_std_).clamp(1e-3, 2.0);
    // Broadcast std to mu shape if needed
    if (std.sizes() != mu.sizes()) {
        std = std.expand_as(mu);
    }
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
    auto log_prob = -0.5 * torch::pow((a - mu) / std, 2) - 0.5 * std::log(2 * M_PI) - torch::log(std);
    log_prob = log_prob - torch::log(1 - torch::pow(torch::tanh(a), 2) + 1e-6);
    
    return {a_tanh.squeeze(0), log_prob.squeeze(0), mu.squeeze(0), std};
}
