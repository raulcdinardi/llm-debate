# llm_eval

Two small CLI tools for quick base-vs-finetuned comparisons in Tinker.

Assumptions:
- Models use a Qwen-style ChatML-ish template (`<|im_start|>...<|im_end|>`).
- The finetuned model is provided as a Tinker `model_path` (e.g. a `tinker://.../sampler_weights/...` URI).
- Output is JSON (intended to be easy for an LLM agent to parse).

## 1) Sample base vs finetuned

Notes:
- The tool now draws a distinct seed per sample to avoid repeated outputs from a single `num_samples` call.
- If `--seed` is omitted, a random base seed is chosen and printed in `sampling_params.seed` along with `sampling_params.sample_seeds` for reproducibility.

```bash
python scripts/llm_eval/sample_base_vs_ft.py \
  --base-model Qwen/Qwen3-4B-Instruct-2507 \
  --ft-model-path 'tinker://0437d4c0-ad18-5763-afc6-bf5b3f67f962:train:0/sampler_weights/secret_word_sft_256_ep5' \
  --user-message 'Write a 1-sentence story that includes the word "banana".' \
  --n 5 \
  --max-tokens 128 \
  --temperature 0.8 \
  --seed 0
```

## 2) Token-level NLL for provided responses (teacher-forced)

`--responses-json` expects a JSON list of strings.

```bash
python scripts/llm_eval/token_losses_base_vs_ft.py \
  --base-model Qwen/Qwen3-4B-Instruct-2507 \
  --ft-model-path 'tinker://0437d4c0-ad18-5763-afc6-bf5b3f67f962:train:0/sampler_weights/secret_word_sft_256_ep5' \
  --user-message 'Say exactly: banana.' \
  --responses-json '["banana.","Banana!","I cannot comply."]' \
  --concurrency 2 \
  --seed 0
```

## 3) Blind sample + reveal (shuffle then label)

Generate a blinded set plus a key (mapping), then reveal in a second step.

```bash
python scripts/llm_eval/blind_sample_base_vs_ft.py generate \
  --base-model Qwen/Qwen3-4B-Instruct-2507 \
  --ft-model-path 'tinker://0437d4c0-ad18-5763-afc6-bf5b3f67f962:train:0/sampler_weights/secret_word_sft_256_ep5' \
  --user-message 'Write a 1-sentence story that includes the word "banana".' \
  --n 5 \
  --max-tokens 128 \
  --temperature 0.8 \
  --seed 0 \
  --shuffle-seed 1 \
  --out-blind /tmp/blind_samples.json \
  --out-key /tmp/blind_key.json

python scripts/llm_eval/blind_sample_base_vs_ft.py reveal \
  --blind /tmp/blind_samples.json \
  --key /tmp/blind_key.json
```
