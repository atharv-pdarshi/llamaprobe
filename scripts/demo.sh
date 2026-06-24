#!/usr/bin/env bash
set -e

MODEL=${1:-models/tinyllama.gguf}
SESSION=sessions/demo_$(date +%Y%m%d_%H%M%S).json

if [[ ! -f "$MODEL" ]]; then
    echo "Model not found: $MODEL"
    echo "Download it first:"
    echo "  wget https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf -O $MODEL"
    exit 1
fi

mkdir -p sessions

echo "=== Recording inference session ==="
./build/llamaprobe \
    --model "$MODEL" \
    --prompt "Explain how the transformer attention mechanism works in detail." \
    --n-predict 64 \
    --record "$SESSION"

echo ""
echo "=== Replaying session: $SESSION ==="
./build/llamaprobe --replay "$SESSION"