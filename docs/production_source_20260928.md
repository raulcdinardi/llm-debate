# Production source lineage

This synchronization includes the source manifest consumed by the MBPP K1 step200→600 run (`e0598d83`), plus corresponding development tests/documentation. Frozen experiment files remain unchanged.

Source files independently checked against the manifest: 146.

Differences superseding the development checkout: src/llm_local_rl/config.py, src/llm_local_rl/debate_parity.py, src/llm_local_rl/debate_runtime.py, src/llm_local_rl/judge_harness.py, src/llm_local_rl/mixed_label_pairwise.py, src/llm_local_rl/python_jail.py, src/llm_local_rl/python_optimization.py, src/llm_local_rl/python_optimization_debate.py, src/llm_local_rl/python_sandbox.py, src/llm_local_rl/qwen35_instruct_format.py, src/llm_local_rl/registry.py.

No experiment outputs, adapters, credentials, signed URLs or provider controls are included.

Review corrections integrate the previously operational structure-only format adapter, derive dashboard categories from the real nested rollout config, and preserve historical exact-resume fingerprints/adapter inventory. The imported MBPP Python sandbox isolates the host, but its candidate and grader share an interpreter and stdout. Correctness can be forged by hostile interpreter/protocol manipulation; historical correctness and judge-gold metrics are not certified against that attack. Do not present this PR as hardening those measurements. A separate-process trusted grader is required before treating adversarial code correctness as authoritative.

Exact resume can replay steps logged after the latest durable checkpoint. W&B raw history preserves both attempts on the explicit training-step axis. The trailing-average publisher chooses the latest measurement per metric/step, so discarded attempts do not change the averages. Raw history remains auditable and may show both points.

The generic CLI now exposes the mixed-label and Python optimization tasks and their dataset config. Word-limit penalty opt-out fails early outside the Qwen three-point format.
