#include <torch/torch.h>
#include <iostream>
#include <iomanip>
#include <map>

namespace F = torch::nn::functional;

// Graph Util

torch::Tensor normalize_adjacency(torch::Tensor A) {
    auto At = A + torch::eye(A.size(0), A.options());
    auto d_inv_sqrt = At.sum(1).pow(-0.5);
    return d_inv_sqrt.unsqueeze(1) * At * d_inv_sqrt.unsqueeze(0);
}

torch::Tensor scatter_mean(torch::Tensor src, torch::Tensor idx, int64_t N) {
    auto opts = src.options();
    auto counts = torch::zeros({N, 1}, opts);
    counts.index_add_(0, idx, torch::ones({idx.size(0), 1}, opts));
    return torch::zeros({N, src.size(1)}, opts).index_add(0, idx, src) / counts.clamp_min(1.0);
}

void print_accuracy(const std::string& label, torch::Tensor logits,
                    torch::Tensor y, torch::Tensor mask) {
    auto total = mask.sum().item<int64_t>();
    if (total == 0) {
        std::cout << "  " << label << ": 0/0 (N/A)" << std::endl;
        return;
    }
    auto pred = logits.index({mask}).argmax(1);
    auto correct = (pred == y.index({mask})).sum().item<int64_t>();
    std::cout << "  " << label << ": " << correct << "/" << total << " (" << std::fixed << std::setprecision(1) << 100.0 * correct / total << "%)" << std::endl;
}

// Grafenne
// phases=2:  Phase1(feat->obs) + Phase2(GCN)
// phases=3:  Phase1 + Phase2 + Phase3(obs->feat) + Repeat(feat->obs)
// Phase 3 layers are only registered when phases==3, so a 2-phase
// checkpoint loaded into a 3-phase model -> MISSING
// keys-> demonstrating partial checkpoint loading.

struct GrafenneImpl : torch::nn::Module {
    // Phase 1: feat -> obs
    torch::nn::Linear proj_feat{nullptr}, proj_edge{nullptr};
    torch::nn::Linear msg_fto{nullptr}, combine_fto{nullptr};
    torch::Tensor obs_embed;  // learnable initial obs embedding (replaces dead proj_obs)
    torch::Tensor feat_eye;   // cached identity matrix for feature embeddings

    // Phase 2: GCN on original topology  
    torch::nn::Linear gcn0{nullptr}, gcn1{nullptr};

    // Phase 3 + Repeat: only registered when phases==3
    torch::nn::Linear msg_otf{nullptr}, combine_otf{nullptr};
    torch::nn::Linear msg_fto2{nullptr}, combine_fto2{nullptr};

    torch::OrderedDict<std::string, torch::nn::Linear> heads;
    int64_t hdim, F_MAX, current_F, num_phases;

    GrafenneImpl(int64_t f_init, int64_t f_max, int64_t hidden, int64_t phases = 2)
        : hdim(hidden), F_MAX(f_max), current_F(f_init), num_phases(phases) {

        proj_feat    = register_module("proj_feat",    torch::nn::Linear(F_MAX, hdim));
        proj_edge    = register_module("proj_edge",    torch::nn::Linear(1, hdim));
        obs_embed    = register_parameter("obs_embed", torch::randn({1, hdim}) * 0.01);
        feat_eye     = register_buffer("feat_eye",     torch::eye(F_MAX));
        msg_fto      = register_module("msg_fto",      torch::nn::Linear(3 * hdim, hdim));
        combine_fto  = register_module("combine_fto",  torch::nn::Linear(2 * hdim, hdim));
        gcn0         = register_module("gcn0",          torch::nn::Linear(hdim, hdim));
        gcn1         = register_module("gcn1",          torch::nn::Linear(hdim, hdim));

        if (num_phases == 3) {
            msg_otf      = register_module("msg_otf",      torch::nn::Linear(3 * hdim, hdim));
            combine_otf  = register_module("combine_otf",  torch::nn::Linear(2 * hdim, hdim));
            msg_fto2     = register_module("msg_fto2",     torch::nn::Linear(3 * hdim, hdim));
            combine_fto2 = register_module("combine_fto2", torch::nn::Linear(2 * hdim, hdim));
        }
    }

    void add_head(const std::string& name, int64_t out_dim) {
        heads.insert(name, register_module("head_" + name, torch::nn::Linear(hdim, out_dim)));
    }

    void remove_head(const std::string& name) {
        TORCH_CHECK(heads.contains(name), "Head '", name, "' does not exist.");
        heads.erase(name);
        unregister_module("head_" + name);
    }

    struct BipartiteEdges { torch::Tensor obs, feat, vals; };

    BipartiteEdges build_edges(torch::Tensor X, torch::Tensor mask) {
        auto nz   = mask.nonzero();
        auto obs  = nz.select(1, 0);
        auto feat = nz.select(1, 1);
        return {obs, feat, X.index({obs, feat})};
    }

    torch::Tensor feat_to_obs(
            torch::Tensor h_obs, torch::Tensor h_feat,
            const BipartiteEdges& e, int64_t N,
            torch::nn::Linear& msg_l, torch::nn::Linear& comb_l) {
        auto msg = torch::relu(msg_l(torch::cat({
            h_feat.index_select(0, e.feat),
            h_obs.index_select(0, e.obs),
            proj_edge(e.vals.unsqueeze(1))
        }, 1)));
        return torch::relu(comb_l(torch::cat({h_obs, scatter_mean(msg, e.obs, N)}, 1)));
    }

    torch::Tensor obs_to_feat(
            torch::Tensor h_obs, torch::Tensor h_feat,
            const BipartiteEdges& e) {
        auto msg = torch::relu(msg_otf(torch::cat({
            h_obs.index_select(0, e.obs),
            h_feat.index_select(0, e.feat),
            proj_edge(e.vals.unsqueeze(1))
        }, 1)));
        return torch::relu(combine_otf(torch::cat({
            h_feat, scatter_mean(msg, e.feat, current_F)
        }, 1)));
    }

    std::map<std::string, torch::Tensor> forward(
            torch::Tensor A_hat, torch::Tensor X, torch::Tensor mask,
            const std::vector<std::string>& active) {
        auto N = X.size(0);
        auto edges = build_edges(X, mask);

        auto h_obs  = obs_embed.expand({N, hdim});
        auto h_feat = torch::relu(proj_feat(feat_eye.slice(0, 0, current_F)));

        // Phase 1: feat -> obs
        h_obs = feat_to_obs(h_obs, h_feat, edges, N, msg_fto, combine_fto);

        // Phase 2: GCN with residual
        h_obs = torch::relu(A_hat.mm(gcn0(h_obs)) + h_obs);
        h_obs = torch::relu(A_hat.mm(gcn1(h_obs)) + h_obs);

        // Phase 3 + Repeat (only in 3-phase mode)
        if (num_phases == 3) {
            h_feat = obs_to_feat(h_obs, h_feat, edges);
            h_obs  = feat_to_obs(h_obs, h_feat, edges, N, msg_fto2, combine_fto2);
        }

        std::map<std::string, torch::Tensor> out;
        for (auto& name : active) {
            TORCH_CHECK(heads.contains(name), "Head '", name, "' not registered.");
            out[name] = heads[name](h_obs);
        }
        return out;
    }

    void expand_features(int64_t new_F) {
        TORCH_CHECK(new_F <= F_MAX, "new_F exceeds pre-allocated F_MAX");
        current_F = new_F;
    }
};
TORCH_MODULE(Grafenne);

// Checkpoint

void save_checkpoint(const Grafenne& model, const std::string& path) {
    torch::serialize::OutputArchive ar;
    int count = 0;
    for (auto& p : model->named_parameters()) {
        ar.write(p.key(), p.value());
        count++;
    }
    ar.save_to(path);
    std::cout << "  saved " << count << " parameters to " << path << std::endl;
}

void load_checkpoint(Grafenne& model, const std::string& path) {
    torch::serialize::InputArchive ar;
    ar.load_from(path);
    int loaded = 0, missing = 0;
    for (auto& p : model->named_parameters()) {
        torch::Tensor val;
        if (ar.try_read(p.key(), val)) {
            torch::NoGradGuard g;
            p.value().copy_(val);
            std::cout << "  loaded:  " << p.key() << std::endl;
            loaded++;
        } else {
            std::cout << "  MISSING: " << p.key() << std::endl;
            missing++;
        }
    }
    std::cout << "  summary: " << loaded << " loaded, " << missing << " missing" << std::endl;
}

void show_params(const Grafenne& model) {
    int count = 0;
    for (auto& p : model->named_parameters()) {
        std::cout << p.key() << std::endl;
        count++;
    }
    std::cout << "  (" << count << " parameters total)" << std::endl;
}

// EWC 

// Approximate Fisher diagonal via single-pass squared gradients
std::map<std::string, torch::Tensor> compute_fisher(
        Grafenne& model, torch::Tensor A_hat, torch::Tensor X,
        torch::Tensor mask, torch::Tensor y, torch::Tensor train_mask) {
    model->zero_grad();
    auto out = model->forward(A_hat, X, mask, {"cls"});
    auto loss = F::cross_entropy(out["cls"].index({train_mask}), y.index({train_mask}));
    loss.backward();
    std::map<std::string, torch::Tensor> fisher;
    for (auto& p : model->named_parameters()) {
        if (p.value().grad().defined())
            fisher[p.key()] = p.value().grad().square().detach().clone();
    }
    model->zero_grad();
    return fisher;
}

torch::Tensor ewc_penalty(
        const Grafenne& model,
        const std::map<std::string, torch::Tensor>& fisher,
        const std::map<std::string, torch::Tensor>& old_params) {
    auto penalty = torch::tensor(0.0, model->parameters().front().options());
    for (auto& p : model->named_parameters()) {
        auto it_f = fisher.find(p.key());
        auto it_o = old_params.find(p.key());
        if (it_f != fisher.end() && it_o != old_params.end())
            penalty = penalty + (it_f->second * (p.value() - it_o->second).square()).sum();
    }
    return penalty;
}

// Synthetic heterogeneous feature data 

struct HeteroData {
    torch::Tensor A_hat, X, mask, y;
    torch::Tensor train_mask, val_mask, test_mask;

    void to(torch::Device dev) {
        A_hat = A_hat.to(dev); X = X.to(dev); mask = mask.to(dev); y = y.to(dev);
        train_mask = train_mask.to(dev); val_mask = val_mask.to(dev); test_mask = test_mask.to(dev);
    }
};

HeteroData generate_hetero_data(int64_t N, int64_t F_dim) {
    auto keep_rate = torch::rand({N, 1}) * 0.6 + 0.4;
    auto mask = (torch::rand({N, F_dim}) < keep_rate);
    if (F_dim >= 3)
        mask.slice(1, 0, 3).fill_(true);

    auto X = torch::randn({N, F_dim}) * mask.to(torch::kFloat);
    auto y = std::get<1>((X.slice(1, 0, 3) + 0.1 * torch::randn({N, 3})).max(1));

    auto A = (torch::rand({N, N}) < 0.03).to(torch::kFloat);
    A = (A + A.t()).clamp_max(1.0);
    A.fill_diagonal_(0);
    auto A_hat = normalize_adjacency(A);

    auto perm    = torch::randperm(N);
    int64_t n_tr = N * 60 / 100;
    int64_t n_va = N * 20 / 100;
    auto train_mask = torch::zeros({N}, torch::kBool);
    auto val_mask   = torch::zeros({N}, torch::kBool);
    auto test_mask  = torch::zeros({N}, torch::kBool);
    train_mask.index_put_({perm.slice(0, 0, n_tr)}, true);
    val_mask.index_put_({perm.slice(0, n_tr, n_tr + n_va)}, true);
    test_mask.index_put_({perm.slice(0, n_tr + n_va)}, true);

    return {A_hat, X, mask, y, train_mask, val_mask, test_mask};
}

int main() {
    torch::manual_seed(123);
    const int64_t N = 128, F_init = 12, F_max = 16, hdim = 32;

    auto device = torch::kCPU;
    if (torch::cuda::is_available()) {
        device = torch::kCUDA;
        std::cout << "CUDA available — using GPU" << std::endl;
    } else {
        std::cout << "CUDA not available — using CPU" << std::endl;
    }

    auto data = generate_hetero_data(N, F_init);
    data.to(device);
    std::cout << N << " nodes, " << F_init << " features (40-100% present per node)" << std::endl;

    //  2-phase 
    std::cout << std::endl << "=== Train 2-Phase Grafenne ===" << std::endl;
    Grafenne v1(F_init, F_max, hdim, /*phases=*/2);
    v1->add_head("cls", 3);
    v1->to(device);
    std::cout << "v1 parameters:" << std::endl;
    show_params(v1);

    torch::optim::Adam opt1(v1->parameters(), 0.01);
    for (int i = 0; i < 200; i++) {
        opt1.zero_grad();
        auto out = v1->forward(data.A_hat, data.X, data.mask, {"cls"});
        auto loss = F::cross_entropy(out["cls"].index({data.train_mask}), data.y.index({data.train_mask}));
        loss.backward();
        opt1.step();
        if (i % 40 == 0)
            std::cout << "  epoch " << i << "  loss=" << std::fixed << std::setprecision(2) << loss.item<float>() << std::endl;
    }

    {
        torch::NoGradGuard ng;
        auto logits = v1->forward(data.A_hat, data.X, data.mask, {"cls"})["cls"];
        print_accuracy("v1 test", logits, data.y, data.test_mask);
    }

    std::cout << std::endl << "  saving 2-phase checkpoint..." << std::endl;
    save_checkpoint(v1, "grafenne_v1.pt");

    //  3-phase + extra head, partial load 
    std::cout << std::endl << " Checkpoint -> 3-Phase Grafenne (+ head cls2) " << std::endl;
    Grafenne v2(F_init, F_max, hdim, /*phases=*/3);
    v2->add_head("cls", 3);
    v2->add_head("cls2", 2);
    v2->to(device);

    std::cout << "v2 parameters:" << std::endl;
    show_params(v2);

    auto norm_cls2_before = v2->heads["cls2"]->weight.norm().item<float>();

    std::cout << std::endl << "  loading v1 checkpoint into v2..." << std::endl;
    load_checkpoint(v2, "grafenne_v1.pt");

    auto norm_cls2_after = v2->heads["cls2"]->weight.norm().item<float>();
    std::cout << "  cls2 weight norm: " << std::setprecision(4)<< norm_cls2_before << " -> " << norm_cls2_after << " (unchanged = not loaded)" << std::endl;

    {
        torch::NoGradGuard ng;
        auto out = v2->forward(data.A_hat, data.X, data.mask, {"cls", "cls2"});
        std::cout << "  forward OK: cls=" << out["cls"].sizes() << " cls2=" << out["cls2"].sizes() << std::endl;
        print_accuracy("v2 before fine-tune", out["cls"], data.y, data.test_mask);
    }

    // EWC
    std::map<std::string, torch::Tensor> old_params;
    for (auto& p : v2->named_parameters())
        old_params[p.key()] = p.value().detach().clone();
    auto fisher = compute_fisher(v2, data.A_hat, data.X, data.mask, data.y, data.train_mask);
    float ewc_lambda = 100.0;

    std::cout << std::endl << "  fine-tuning 3-phase model with EWC (lambda=" << ewc_lambda << ", 100 epochs)..." << std::endl;
    torch::optim::Adam opt2(v2->parameters(), 0.005);
    for (int i = 0; i < 100; i++) {
        opt2.zero_grad();
        auto out = v2->forward(data.A_hat, data.X, data.mask, {"cls"});
        auto ce = F::cross_entropy(out["cls"].index({data.train_mask}), data.y.index({data.train_mask}));
        auto ewc = ewc_penalty(v2, fisher, old_params);
        auto loss = ce + ewc_lambda * ewc;
        loss.backward();
        opt2.step();
        if (i % 20 == 0)
            std::cout << "    epoch " << i << "  ce=" << std::fixed << std::setprecision(3) << ce.item<float>() << "  ewc=" << std::setprecision(5) << ewc.item<float>() << "  total=" << std::setprecision(3) << loss.item<float>() << std::endl;
    }

    {
        torch::NoGradGuard ng;
        auto logits = v2->forward(data.A_hat, data.X, data.mask, {"cls"})["cls"];
        print_accuracy("v2 test (after ft)", logits, data.y, data.test_mask);
    }

    //  Dynamic Feature Expansion 
    std::cout << std::endl << " Dynamic Feature Expansion " << std::endl;
    std::cout << "  expanding features: " << F_init << " -> " << F_max << std::endl;
    v2->expand_features(F_max);

    auto new_mask = (torch::rand({N, F_max - F_init}, torch::TensorOptions().device(device)) < 0.6);
    auto new_X = torch::randn({N, F_max - F_init}, torch::TensorOptions().device(device)) * new_mask.to(torch::kFloat);
    data.X    = torch::cat({data.X, new_X}, 1);
    data.mask = torch::cat({data.mask, new_mask}, 1);

    {
        torch::NoGradGuard ng;
        auto out = v2->forward(data.A_hat, data.X, data.mask, {"cls"});
        std::cout << "  post-expansion forward OK: " << out["cls"].sizes() << std::endl;
    }

    std::cout << "  fine-tuning with " << F_max << " features " << std::endl;
    torch::optim::Adam opt3(v2->parameters(), 0.005);
    for (int i = 0; i < 100; i++) {
        opt3.zero_grad();
        auto out = v2->forward(data.A_hat, data.X, data.mask, {"cls"});
        auto loss = F::cross_entropy(out["cls"].index({data.train_mask}), data.y.index({data.train_mask}));
        loss.backward();
        opt3.step();
        if (i % 20 == 0)
            std::cout << "    epoch " << i << "  loss=" << std::fixed << std::setprecision(2) << loss.item<float>() << std::endl;
    }

    {
        torch::NoGradGuard ng;
        auto logits = v2->forward(data.A_hat, data.X, data.mask, {"cls"})["cls"];
        print_accuracy("final test accuracy", logits, data.y, data.test_mask);
    }
}
