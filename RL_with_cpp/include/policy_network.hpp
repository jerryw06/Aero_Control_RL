#ifndef POLICY_NETWORK_HPP
#define POLICY_NETWORK_HPP

#include <torch/torch.h>
#include <vector>

class PolicyImpl : public torch::nn::Module {
public:
    PolicyImpl(int obs_dim = 6, int hidden = 1024, int num_layers = 8, double act_limit = 3.0);

    std::pair<torch::Tensor, torch::Tensor> forward(torch::Tensor obs_t);
    
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> 
    sample_action(torch::Tensor obs_np);

    double get_act_limit() const { return act_limit_; }

private:
    torch::nn::Sequential net_{nullptr};
    torch::Tensor log_std_;
    double act_limit_;
};

TORCH_MODULE(Policy);

#endif // POLICY_NETWORK_HPP
