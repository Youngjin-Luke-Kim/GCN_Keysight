#include <torch/torch.h>
#include <iostream>
#include <random>
#include <map>

namespace F = torch::nn::functional;

// GCN struc. Declar.
struct GCNImpl : torch::nn::Module {

    torch::nn::Linear w0{nullptr}, w1{nullptr}; // 2 layer
    torch::OrderedDict<std::string, torch::nn::Linear> heads, adapters;
    int64_t fdim, hdim;

    GCNImpl(int64_t features, int64_t hidden) : fdim(features), hdim(hidden) {
        w0 = register_module("W0", torch::nn::Linear(features, hidden));
        w1 = register_module("W1", torch::nn::Linear(hidden, hidden));
    }

    torch::Tensor back_bone(torch::Tensor A, torch::Tensor X) { //Backbone A -> X -> Res
        X = torch::relu(A.mm(w0(X)));
        return torch::relu(A.mm(w1(X)));
    }

    //Dynamic head
    void add_head(std::string name, int64_t out_dim) {
        heads.insert(name, register_module("Head_" + name, torch::nn::Linear(hdim, out_dim))); 
    }
    
    void remove_head(std::string name) { heads.erase(name); } // remove head

    std::map<std::string, torch::Tensor> forward(torch::Tensor A, torch::Tensor X, const std::vector<std::string>& active_heads) {
        auto H = back_bone(A, X);
        std::map<std::string, torch::Tensor> outputs;
        for(auto& name : active_heads) { outputs[name] = heads[name](H); }
        return outputs;
    }

    void add_input_group(const std::string& name, int64_t input_dim) {
        adapters.insert(name, register_module("Adapter_" + name, torch::nn::Linear(input_dim, fdim)));
    }

    std::map<std::string, torch::Tensor> forward_groups (torch::Tensor A, const std::map<std::string, torch::Tensor>& inputs, const std::vector<std::string>& active) {

        auto N = inputs.begin()->second.size(0);
        auto X = torch::zeros({N, fdim});
        for (auto& [name, feat] : inputs){ X += adapters[name](feat);}
        X /= (double)inputs.size();
        return forward(A, X, active);
    }
};
TORCH_MODULE(GCN);

//Helpers
void save_checkpoint(const GCN& model, const std::string& path) {
    torch::serialize::OutputArchive arx;
    for (auto & p : model->named_parameters()) {arx.write(p.key(), p.value());}
    arx.save_to(path);
}

void load_checkpoint(GCN& model, const std::string& path) {
    torch::serialize::InputArchive arx;
    arx.load_from(path);
    for (auto & p : model->named_parameters()) {
        torch::Tensor loaded_params;
        if(arx.try_read(p.key(), loaded_params)) {
            torch::NoGradGuard g;
            p.value().copy_(loaded_params);
            std::cout << "Loaded " << p.key() << " with shape " << p.value().sizes() << std::endl;
        } else {
            std::cout << "No checkpoint found for " << p.key() << std::endl;
        }
    }
}

void show_params(const GCN& model) {
    for (auto& p : model->named_parameters()) {
        std::cout << p.key() << std::endl;
    }
}

int main() {
    
    //Part A  Graph info declar.
    torch::manual_seed(3407);
    std::mt19937 gen(3407);
    std::uniform_real_distribution<float> dist(0.05f, 0.1f);
    float density = dist(gen);
    const int N = 64, F = 8, H = 16, ITER = 30;
    auto noise = 0.01f * torch::randn({N, 3});
    auto A = (torch::rand({N,N}) < density).to(torch::kFloat);
    A = (A + A.t()).clamp_max(1.0f); //sym.
    auto A_tilde = A + torch::eye(N);
    auto D = A_tilde.sum(1).pow(-0.5).diag(); // D^-1/2
    auto A_hat = D.mm(A_tilde).mm(D);
    auto X = torch::randn({N, F});
    auto y = torch::argmax(X.slice(1,0,3) + noise, 1);
    //Check
    std::cout << "N is" << N << " and edges are " << (int)(A.sum().item<float>() / 2) << std::endl;

    // GCN Layer Declar.
    GCN gcn(F, H);
    auto PartB = gcn -> back_bone(A_hat, X);
    std::cout << "H2 Shape: " << PartB.sizes() << std::endl;

    //Dynamic head
    std::cout << "Before adding cls head:" << std::endl;
    show_params(gcn);
    gcn -> add_head("cls", 3);
    std::cout << "After adding cls head:" << std::endl;
    show_params(gcn);
    gcn -> add_head("cls2", 2);
    std::cout << "After adding cls2 head:" << std::endl;
    show_params(gcn);

    // Training
    GCN gcn_final(F, H);
    gcn_final -> add_head("cls", 3);
    torch::optim::Adam optimizer(gcn_final -> parameters(), 0.06);
    for (int iter = 0; iter < ITER; ++iter) {
        optimizer.zero_grad();
        auto loss = F::cross_entropy(gcn_final -> forward(A_hat, X, {"cls"})["cls"], y);
        loss.backward();
        optimizer.step();
        if(iter % 5 == 0) {
            std::cout << "Iteration " << iter << " loss= " << loss.item<float>() << std::endl;
        }
    }

    save_checkpoint(gcn_final, "gcn_final.pt"); //save N load
    GCN gcn_final_loaded(F, H);
    gcn_final_loaded -> add_head("cls", 3);
    gcn_final_loaded -> add_head("cls2", 2);
    auto norm_before = gcn_final_loaded -> heads["cls2"] -> weight.norm().item<float>(); //optional
    load_checkpoint(gcn_final_loaded, "gcn_final.pt");
    auto norm_after = gcn_final_loaded -> heads["cls2"] -> weight.norm().item<float>();
    std::cout << "Norm before: " << norm_before << " Norm after: " << norm_after << std::endl;

    auto outputs_for_both_heads = gcn_final_loaded -> forward(A_hat, X, {"cls", "cls2"});
    std::cout << "Outputs for both heads: " << outputs_for_both_heads["cls"].sizes() << " " << outputs_for_both_heads["cls2"].sizes() << std::endl;

    //stretch gcn declar.
    GCN stretch_gcn(F, H);
    stretch_gcn -> add_head("cls", 3);
    stretch_gcn -> add_input_group("feat_dim_A", 4);
    stretch_gcn -> add_input_group("feat_dim_B", 5);

    //fake data gen
    auto feat_dim_A = torch::randn({N, 4});
    auto feat_dim_B = torch::randn({N, 5});
    std::map<std::string, torch::Tensor> inputs;
    inputs.insert({"feat_dim_A", feat_dim_A});
    inputs.insert({"feat_dim_B", feat_dim_B});
    auto outputs = stretch_gcn -> forward_groups(A_hat, inputs, {"cls"});
    std::cout << "Outputs: " << outputs["cls"].sizes() << std::endl;
    

    
}












