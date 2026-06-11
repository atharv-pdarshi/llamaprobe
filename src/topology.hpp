#pragma once

#include "types.hpp"
#include <string>
#include <vector>

// A node in the model layer tree shown in Panel 1
struct LayerNode {
    std::string            name;
    LayerType              type      = LayerType::Other;
    bool                   expanded  = true;
    bool                   selected  = false;
    std::vector<LayerNode> children;
};

// Build a topology tree from a flat list of layer names captured during inference.
// Names follow llama.cpp convention: "blk.0.attn_q", "blk.0.ffn_up", etc.
LayerNode build_topology(const std::vector<std::string>& layer_names,
                          const std::string& model_name = "model");
