#include "topology.hpp"
#include <algorithm>
#include <sstream>

// Classify a layer name into a LayerType
static LayerType classify(const std::string& name) {
    if (name.find("token_embd") != std::string::npos) return LayerType::Embedding;
    if (name.find("attn_norm")  != std::string::npos ||
        name.find("ffn_norm")   != std::string::npos ||
        name.find("result_norm")!= std::string::npos) return LayerType::LayerNorm;
    if (name.find("attn")       != std::string::npos) return LayerType::Attention;
    if (name.find("ffn")        != std::string::npos) return LayerType::MLP;
    if (name.find("output")     != std::string::npos) return LayerType::Output;
    return LayerType::Other;
}

// Parse "blk.3.attn_q" → prefix="blk.3", leaf="attn_q"
static std::pair<std::string, std::string> split_name(const std::string& name) {
    auto dot = name.rfind('.');
    if (dot == std::string::npos) return {"", name};
    return {name.substr(0, dot), name.substr(dot + 1)};
}

LayerNode build_topology(const std::vector<std::string>& layer_names,
                          const std::string& model_name) {
    LayerNode root;
    root.name     = model_name;
    root.type     = LayerType::Other;
    root.expanded = true;

    // Group layer names by their parent prefix
    // e.g. "blk.0.attn_q" → parent "blk.0", leaf "attn_q"
    // "blk.0" → parent "blk", leaf "0"

    // Collect unique prefixes first
    std::vector<std::string> unique_names = layer_names;
    std::sort(unique_names.begin(), unique_names.end());
    unique_names.erase(std::unique(unique_names.begin(), unique_names.end()),
                       unique_names.end());

    // Two-level grouping: group → block → leaf
    // group = "blk", "token_embd", "output"
    std::map<std::string, std::map<std::string, std::vector<std::string>>>
        groups;  // group → block → [leaves]

    for (const auto& n : unique_names) {
        auto [prefix, leaf] = split_name(n);
        if (prefix.empty()) {
            groups[""][n] = {};
        } else {
            auto [group, block] = split_name(prefix);
            if (group.empty()) groups[prefix][leaf].push_back(n);
            else               groups[group][block].push_back(n);
        }
    }

    for (auto& [grp_name, blocks] : groups) {
        LayerNode grp_node;
        grp_node.name     = grp_name.empty() ? "misc" : grp_name;
        grp_node.type     = LayerType::Other;
        grp_node.expanded = grp_name == "blk";  // expand blk by default

        for (auto& [blk_name, leaves] : blocks) {
            if (leaves.empty()) {
                // It's a leaf itself
                LayerNode leaf_node;
                leaf_node.name = blk_name;
                leaf_node.type = classify(blk_name);
                grp_node.children.push_back(std::move(leaf_node));
                continue;
            }
            LayerNode blk_node;
            blk_node.name     = blk_name;
            blk_node.type     = LayerType::Other;
            blk_node.expanded = false;

            for (const auto& lname : leaves) {
                LayerNode leaf;
                leaf.name = lname;
                leaf.type = classify(lname);
                blk_node.children.push_back(std::move(leaf));
            }
            grp_node.children.push_back(std::move(blk_node));
        }
        root.children.push_back(std::move(grp_node));
    }
    return root;
}
