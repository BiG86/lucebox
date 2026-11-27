#!/usr/bin/env bash
# Explicit opt-in; see server/docs/INKLING2_DSPARK_SHORT.md.
set -euo pipefail

if [[ $# -ne 3 ]]; then
    echo "usage: $0 LUCE_SERVER TARGET_GGUF DSPARK_SAFETENSORS" >&2
    exit 2
fi

server=$1
target=$2
draft=$3
if [[ $server != */* ]]; then
    server=./$server
fi
if [[ ! -f $server || ! -x $server ]]; then
    echo "not an executable server file: $server" >&2
    exit 1
fi
for path in "$target" "$draft"; do
    if [[ ! -f $path || ! -r $path ]]; then
        echo "not a readable model file: $path" >&2
        exit 1
    fi
done
case "$draft" in
    0|off|none) draft=./$draft ;;
esac

exec env -u GGML_CUDA_DISABLE_GRAPHS -u GGML_CUDA_DISABLE_GRAPHS_DEVICES \
    -u LUCE_INKLING2_VERIFY_WIDTH -u LUCE_INKLING2_ADAPTIVE_VW \
    LUCE_INKLING2_WAVEFRONT_PREFILL=0 LUCE_INKLING2_DSPARK_Q4_HEAD=0 \
    "$server" "$target" --draft "$draft" --profile none --target-device hip:0 \
    --max-ctx 8192 --chunk 512 --cache-type-k q8_0 --cache-type-v q8_0 \
    --verify-width 0 --pflash off --max-concurrency 1 --default-max-tokens 128 \
    --prefix-cache-slots 0 --disk-prefix-cache off --shared-prefix off --no-agent-turn-cache \
    --model-name inkling2-mini --host 127.0.0.1 --port 8080
