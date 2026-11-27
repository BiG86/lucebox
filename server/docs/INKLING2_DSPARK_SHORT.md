# Inkling 2 Mini: explicit DSpark short-context profile on HIP

This opt-in profile is for one RX 7900 XTX (`gfx1100`, visible as `hip:0`) with ROCm 10.2.0. It uses an 8,192-token context and is intended for local evaluation. DSpark can produce different tokens from greedy autoregressive (AR) decoding, even when both answers satisfy the same task. This profile does not establish V1 qualification or change upstream model cards or draft auto-discovery. The launcher always takes an explicit local DSpark checkpoint. To run AR outside this profile, use `--draft off` explicitly; omission can auto-discover `mtp.safetensors` beside the GGUF.

**M4 gate:** native validation passed; hold publication or promotion as a serving profile because the HTTP semantic smoke did not pass. Use the launcher for local evaluation only.

## Build and launch

Use the [ROCm 10.2.0 toolchain](https://rocm.docs.amd.com/en/docs-10.2.0/install/rocm.html). From the repository root, set `ROCM_PATH` to its installation prefix and `TARGET_GGUF` and `DSPARK_SAFETENSORS` to readable local files. For the exact reference assets, verify SHA-256:

| Asset | SHA-256 |
|---|---|
| `Inkling-2-Mini-UD-Q4_K_M.gguf` | `78fcedcf93c5d1bfe83e225513b71392bb52259be0eaa5f8899c856a2e5834ae` |
| `mtp.safetensors` | `7c3b651317f01f73bcf4e1dceba66ef8b7e1a1eac6bc171e882e690a4b10a1d2` |

```bash
sha256sum "$TARGET_GGUF" "$DSPARK_SAFETENSORS"
build_dir=${BUILD_DIR:-build-inkling2-hip}
cmake -G Ninja -S server -B "$build_dir" -DCMAKE_BUILD_TYPE=Release \
  -DLUCE_GPU_BACKEND=hip -DLUCE_HIP_ARCHITECTURES=gfx1100 -DGGML_HIP_GRAPHS=ON \
  -DROCM_PATH="$ROCM_PATH" -DCMAKE_PREFIX_PATH="$ROCM_PATH"
cmake --build "$build_dir" --target luce_server
grep 'GGML_HIP_GRAPHS:BOOL=ON' "$build_dir/CMakeCache.txt"
# After checking GPU isolation and starting the VRAM monitor below:
bash server/scripts/serve_inkling2_dspark_short.sh \
  "$build_dir/luce_server" "$TARGET_GGUF" "$DSPARK_SAFETENSORS"
```

The [launcher](../scripts/serve_inkling2_dspark_short.sh) binds `127.0.0.1:8080`, uses `--profile none`, one concurrent request, chunk `512`, and Q8_0 K/V. It sets `--pflash off`, `LUCE_INKLING2_WAVEFRONT_PREFILL=0`, `LUCE_INKLING2_DSPARK_Q4_HEAD=0`, and disables prefix, disk, shared-prefix and agent-turn caches. It uses the default adaptive verify policy and clears graph-disable overrides; the build flag is still required. Start from an environment without other experimental Inkling or ggml tuning. Startup should show one `gfx1100` device, the expected loaded DSpark checkpoint, `max_ctx=8192`, `kv=q8_0`, no remote cache, `chunk = 512`, and `pflash = off`. A loaded checkpoint alone does not prove that a request used speculative decode; check positive accepted/proposed draft counts in the native decode log.

The launcher is not a GPU watchdog. Before evaluating, keep other GPU compute workloads stopped (including the benchmark Ollama container, if present). Identify the selected render node and its matching DRM `card` under `/sys/class/drm` rather than assuming an index on a different host. Read `mem_info_vram_total` and `mem_info_vram_used` on that card. In M4 the short preflight required ≥3 GiB projected free after subtracting the short work bound (19,300,000,000 bytes), the larger of measured idle and 1,588,957,184 bytes, and a further 1 GiB margin. This bound is specific to the listed model and hardware. Monitor free VRAM at 50 ms intervals and stop the evaluation below 2.5 GiB to protect the 2 GiB floor. Recheck that VRAM returns to its idle range after shutdown. For container validation, expose only the selected render node and `/dev/kfd`, mount assets read-only, and use an isolated loopback without publishing a port.

## Structured output smoke (loopback)

After the server becomes healthy, send one request from the same network namespace. The returned `message.content` should be a JSON object with `esito` equal to `confermato` and integer `totale` equal to `1`; look for `finish_reason: stop`. The request uses the [`response_format` API](API.md#supported-request-parameters).

In M4 the server started with the intended profile and returned schema-valid JSON with `finish_reason=stop`, but returned `totale=5` instead of `1`. The grammar path ran AR (`spec_decode_ran=false`) even though the draft was loaded. A separate control removing only `response_format` ran DSpark, reached the 128-token cap and returned empty visible content. The separate native structured prompt returned the correct values with real DSpark acceptance. Neither HTTP request passed its semantic check.

```bash
curl --noproxy '*' -fsS --max-time 60 http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' --data-binary @- <<'JSON'
{
  "model": "inkling2-mini",
  "messages": [{"role": "user", "content": "Scrivi solo un oggetto JSON. Se tre casse contengono due pezzi ciascuna e ne consegni cinque, imposta totale al numero intero rimasto ed esito alla parola italiana confermato."}],
  "temperature": 0,
  "seed": 42,
  "max_tokens": 128,
  "chat_template_kwargs": {"enable_thinking": false},
  "response_format": {
    "type": "json_schema",
    "json_schema": {
      "name": "inventario",
      "strict": true,
      "schema": {
        "type": "object",
        "properties": {"esito": {"type": "string"}, "totale": {"type": "integer"}},
        "required": ["esito", "totale"],
        "additionalProperties": false
      }
    }
  }
}
JSON
```

## Native comparison reference

The 52,224-token test is a separate, single-request in-process diagnostic; it does not qualify 52,224-token HTTP serving. It compares port runtime `0498d8c9904ce987b1448548d978aa169ed2a4bb` (fork base `2762166be103dd25c07999a5f38a4986bc8e7247`) to M2 runtime `f9ccfa0239d602a2c71479dd5fd077c5c877658c`, using the same clock, assets, target Q8_0, chunk 512, and explicit DSpark. The M2 comparison is **107.490 tok/s**, prefill **27,683.305 ms**, and **110/117** draft accepts. The historical 106.069 tok/s sample used a different, graphs-OFF build.

Regenerate the request below. The UTF-8 prompt SHA-256 must be `4c06c9d75a4b380d448dd9d74b1ba1049b078333bed408315e0d326007f4eede`, and the compact JSON array of 52,092 tokenizer IDs must hash to `fa1418c5e11fc460cf78576e056a816e6e838e6d7d785bb2d9c7bdb35bb6ab61`.

```bash
python3 - > request-52224.json <<'PY'
import hashlib
import json

instruction = 'Complete the sequence with consecutive integers separated by commas. Output at least 300 more integers, without explanation.'
prompt = instruction + '\n' + ', '.join(map(str, range(1, 13268))) + ', '
assert hashlib.sha256(prompt.encode()).hexdigest() == '4c06c9d75a4b380d448dd9d74b1ba1049b078333bed408315e0d326007f4eede'
print(json.dumps({'raw': True, 'experimental_capacity': True, 'measure': True, 'prompt': prompt,
                  'options': {'num_ctx': 52224, 'num_predict': 128, 'temperature': 0, 'seed': 42}}))
PY
```

The reference probe is `LLM/lucebox/trace_inkling2_dspark.cpp` from revision `9655a9a` of the benchmark evidence repository (source SHA-256 `291a8a69e5e72963081daa1b24806f61613b459f787ef018c9db89a994b02554`). Compile that probe separately with the flags of `smoke_inkling2_forward` from `compile_commands.json`, and replace the smoke object and output in Ninja's link command (`ninja -C "$build_dir" -t commands smoke_inkling2_forward`). Build the smoke target first so the same runtime libraries are available. Run with the assets as the first two arguments and `1` as the repetition count, redirecting `request-52224.json` to stdin. Apply the same graph, wavefront and draft-head environment as above; the probe directly invokes the backend without PFlash or prefix caching. At 52,224 the M4 preflight used the 19,864,915,968-byte M2 work bound plus the same 1 GiB safety margin, requiring ≥2.75 GiB projected free; the live stop and hard floor remained 2.5 and 2 GiB, respectively.

Accept a sample for this comparison only if the entire ID array matches, `cached_tokens=0`, 128 IDs were emitted at the cap (`length`), the numeric continuation is correct, and DSpark actually accepted draft tokens. Use `128 × 10⁹ / emit_window_ns` where `emit_window_ns` measures the first-to-last non-EOS callback in `steady_clock`; record native prefill and acceptance separately. A single valid sample is diagnostic rather than a statistical performance claim.

M4 measured **107.643 tok/s** (1,189,119,783 ns), **27,674.409 ms** native prefill and **110/117** draft accepts over 17 verify steps. Input and output IDs, stop, and verify decisions matched the M2 sample. The minimum free VRAM sampled was **3.921 GiB**. At 8,192 the native retrieval and structured fixtures passed; one cold AR/DSpark replay per representative prompt matched through EOS for Italian prose, Python and structured summarization. Earlier repeated replays diverged (the Italian prompt matched on warm-up, then failed exact parity in five out of five measured replays). M4 did not repeat that series and does not establish parity or general serving quality.
