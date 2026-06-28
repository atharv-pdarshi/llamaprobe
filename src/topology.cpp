#include "topology.hpp"
#include <algorithm>
#include <cctype>
#include <map>

// Classify an activation tensor name into a LayerType.
// llama.cpp names activations like: Qcur-3, Kcur-3, ffn_gate-3, result_norm
static LayerType classify(const std::string& name) {
    auto has = [&](const char* s) { return name.find(s) != std::string::npos; };

    if (has("token_embd") || has("inp_embd"))   return LayerType::Embedding;

    // Normalization — must come before attn/ffn because "attn_norm" contains "attn".
    // Matches attn_norm, ffn_norm, result_norm, _norm, and bare "norm-N"
    if (has("norm"))                             return LayerType::LayerNorm;

    // Attention: Qcur, Kcur, Vcur, kq*, kqv*, inpSA, attn*, KV cache views
    if (has("Qcur")    || has("Kcur")    || has("Vcur") ||
        has("kqv")     || has("kq")      ||
        has("cache_k") || has("cache_v") ||
        has("inpSA")   || has("attn"))           return LayerType::Attention;

    // MLP/FFN: ffn_inp, ffn_gate, ffn_up, ffn_down, inpFF
    if (has("ffn") || has("mlp") || has("inpFF")) return LayerType::MLP;

    if (has("output") || has("lm_head"))         return LayerType::Output;
    return LayerType::Other;
}

// Parse "Kcur-5" → ("Kcur", 5).  Returns layer=-1 if no numeric suffix found.
static std::pair<std::string, int> parse_name(const std::string& name) {
    auto dash = name.rfind('-');
    if (dash != std::string::npos && dash + 1 < name.size()) {
        const std::string suffix = name.substr(dash + 1);
        if (!suffix.empty() &&
            std::all_of(suffix.begin(), suffix.end(), ::isdigit)) {
            return {name.substr(0, dash), std::stoi(suffix)};
        }
    }
    return {name, -1};
}

LayerNode build_topology(const std::vector<std::string>& layer_names,
                          const std::string& model_name) {
    LayerNode root;
    root.name     = model_name;
    root.type     = LayerType::Other;
    root.expanded = true;

    // Deduplicate
    std::vector<std::string> names = layer_names;
    std::sort(names.begin(), names.end());
    names.erase(std::unique(names.begin(), names.end()), names.end());

    // Buckets: global tensors (no layer number) and per-layer tensors
    std::vector<std::string>              global_tensors;
    std::map<int, std::vector<std::string>> per_layer;  // layer_idx → [names]

    for (const auto& n : names) {
        auto [base, layer] = parse_name(n);
        if (layer < 0) global_tensors.push_back(n);
        else            per_layer[layer].push_back(n);
    }

    // ── Global tensors → split into embedding / output / misc ────────────────
    std::vector<std::string> embd_list, output_list, misc_list;
    for (const auto& n : global_tensors) {
        LayerType t = classify(n);
        if (t == LayerType::Embedding) embd_list.push_back(n);
        else if (t == LayerType::Output || t == LayerType::LayerNorm)
            output_list.push_back(n);
        else misc_list.push_back(n);
    }

    auto make_leaf_group = [](const std::string& gname,
                               LayerType gtype,
                               const std::vector<std::string>& items) -> LayerNode {
        LayerNode g;
        g.name     = gname;
        g.type     = gtype;
        g.expanded = false;
        for (const auto& n : items) {
            LayerNode leaf;
            leaf.name = n;
            leaf.type = classify(n);
            g.children.push_back(std::move(leaf));
        }
        return g;
    };

    if (!embd_list.empty())
        root.children.push_back(make_leaf_group("embedding", LayerType::Embedding, embd_list));

    // ── Per-layer groups ──────────────────────────────────────────────────────
    for (auto& [idx, tensors] : per_layer) {
        LayerNode layer_node;
        layer_node.name     = "layer-" + std::to_string(idx);
        layer_node.type     = LayerType::Other;
        layer_node.expanded = false;

        // Sort tensors within the layer: Attention first, then MLP, then rest
        std::sort(tensors.begin(), tensors.end(), [](const std::string& a, const std::string& b) {
            int ra = (classify(a) == LayerType::Attention) ? 0 :
                     (classify(a) == LayerType::MLP)       ? 1 : 2;
            int rb = (classify(b) == LayerType::Attention) ? 0 :
                     (classify(b) == LayerType::MLP)       ? 1 : 2;
            if (ra != rb) return ra < rb;
            return a < b;
        });

        for (const auto& n : tensors) {
            LayerNode leaf;
            leaf.name = n;
            leaf.type = classify(n);
            layer_node.children.push_back(std::move(leaf));
        }
        root.children.push_back(std::move(layer_node));
    }

    if (!output_list.empty())
        root.children.push_back(make_leaf_group("output", LayerType::Output, output_list));
    if (!misc_list.empty())
        root.children.push_back(make_leaf_group("misc", LayerType::Other, misc_list));

    return root;
}
