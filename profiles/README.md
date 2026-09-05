# Recorded experiment profiles

Profiles preserve the explicit arguments from six archived launch commands.
`source_launcher_sha256` identifies the original; `compatibility_mapping`
records any renamed mode needed to preserve its meaning. The CW baseline's
historical `soft_judge` is today's `soft_judge_raw`, not reliability weighting.

Training profiles retain 8×16 for CW baseline/full-gap and MMLU prompt-GRPO,
and the actual 64×2 used by the OpenBookQA direct-CE+JS / CW unlabeled-JS runs.
The latter geometry and the collapsed unlabeled-JS treatment are historical
reproduction settings, not recommendations for a new comparison.

Write a JSON bindings file supplying exactly the deployment names listed in
the profile's `bindings` array (model/tokenizer directory, input roots, output
directory and W&B run name). Adapter roots must hold the original verified
checkpoint assets, including the appropriate judge-harness manifest. Corpus
paths and the full-gap inference-bound judge are independent inputs.

```bash
python scripts/experiment_profile.py profiles/cw_fullgap_round3.json \
  --bindings /path/to/bindings.json
```

This prints the resolved command after parsing it. `--run` explicitly executes
it; no model/API work happens by default. Complete the usual experiment
preflight before launching. Do not use an existing output directory for a
fresh run. These profiles bind arguments, not model weights or a hardware
environment, and do not replace the original run manifests or checkpoints.
