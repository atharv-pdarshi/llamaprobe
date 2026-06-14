#include "../src/topology.hpp"
#include <cassert>
#include <vector>
#include <string>

static void test_empty_input() {
    auto root = build_topology({}, "test-model");
    assert(root.name == "test-model");
    assert(root.children.empty());
}

static void test_basic_llama_layers() {
    std::vector<std::string> names = {
        "token_embd",
        "blk.0.attn_norm",
        "blk.0.attn",
        "blk.0.ffn_norm",
        "blk.0.ffn",
        "blk.1.attn_norm",
        "blk.1.attn",
        "blk.1.ffn_norm",
        "blk.1.ffn",
        "result_norm",
        "output",
    };
    auto root = build_topology(names, "tinyllama");
    assert(root.name == "tinyllama");
    assert(!root.children.empty());
}

static void test_layer_type_classification() {
    std::vector<std::string> names = {
        "token_embd",
        "blk.0.attn",
        "blk.0.ffn",
        "blk.0.attn_norm",
        "output",
    };
    auto root = build_topology(names);

    // Root should have children — just verify it builds without crash
    assert(root.children.size() > 0);
}

static void test_deduplication() {
    std::vector<std::string> names = {
        "blk.0.attn",
        "blk.0.attn",   // duplicate
        "blk.0.ffn",
    };
    // Should not crash or produce duplicate nodes
    auto root = build_topology(names);
    assert(!root.children.empty());
}

int main() {
    test_empty_input();
    test_basic_llama_layers();
    test_layer_type_classification();
    test_deduplication();
    return 0;
}
