from __future__ import annotations

from dataclasses import dataclass

from tinker_debate.debate_types import assemble_training_data, assemble_training_data_grpo
from tinker_debate.paradigms.debate import DebateParadigm
from tinker_debate.paradigms.normal import NormalParadigm
from tinker_debate.tasks.confidence_task import ConfidenceTask
from tinker_debate.tasks.qa_task import QATask
from tinker_debate.tasks.summary_task import SummaryTask

from .driver_context import DriverContext
from .driver_types import RolloutOutput, RolloutDriver


@dataclass
class OrthogonalDriver(RolloutDriver):
    ctx: DriverContext

    def __post_init__(self) -> None:
        args = self.ctx.args

        if args.mode == "single_turn":
            if args.env is None:
                raise ValueError("--env is required when --mode=single_turn")
            task_name = args.env
        elif args.mode == "debate":
            task_name = "qa" if args.env is None else args.env
        else:
            raise ValueError(f"Unknown --mode={args.mode!r}")

        if task_name == "summary":
            if args.dataset not in (None, "cnn_dailymail"):
                raise ValueError("--dataset must be cnn_dailymail for summary task")
            n_samples = 1000
            self.task = SummaryTask.from_args(reward_fn=args.reward_fn, seed=args.seed, n_samples=n_samples)
            self.normal_max_tokens = 256
        elif task_name == "confidence":
            self.task = ConfidenceTask.from_file()
            self.normal_max_tokens = 128
        elif task_name == "qa":
            if args.dataset == "cnn_dailymail":
                raise ValueError("--dataset=cnn_dailymail is not a QA dataset")
            self.task = QATask.from_args(dataset_name=args.dataset, seed=args.seed)
            self.normal_max_tokens = 128
        else:
            raise ValueError(f"Unknown task {task_name!r}")

    async def rollout_step(self, *, step: int) -> RolloutOutput:
        args = self.ctx.args
        client = self.ctx.client
        console = self.ctx.console

        if args.mode == "single_turn":
            def save_record(record: dict) -> None:
                if self.task.name == "summary":
                    self.ctx.log_fns.save_summary_log(record=record, log_dir=self.ctx.log_dir, model_name=client.model_name)
                else:
                    self.ctx.log_fns.save_baseline_log(record=record, log_dir=self.ctx.log_dir, model_name=client.model_name)

            normal = NormalParadigm(
                task=self.task,
                tokenizer=client.tokenizer,
                sample_token_prompts=client.sample_token_prompts,
                save_record=save_record,
                accept_min_reward=float(args.accept_min_reward),
            )
            out = await normal.rollout(
                step=step,
                num_rollouts=int(args.num_rollouts),
                num_groups=int(args.num_groups),
                seed=args.seed,
                max_tokens=int(self.normal_max_tokens),
                temperature=0.7,
                tags=[self.task.name],
            )
            return RolloutOutput(
                training_data=out.training_data,
                rollout_time_s=float(out.rollout_time_s),
                num_rollouts=int(out.num_rollouts),
                info_lines=out.info_lines,
            )

        if args.mode != "debate":
            raise ValueError(f"Unknown --mode={args.mode!r}")

        from tinker_debate.debate_env import confidence_judge_r1, mock_judge_random
        from tinker_debate.debate_types import DebateConfig, compute_training_stats

        group_size = args.num_rollouts // args.num_groups
        debates_per_question = group_size // 2
        num_debates = args.num_groups * debates_per_question
        console.print(f"[dim]Running {num_debates} debates (task={self.task.name})[/dim]")

        if args.confidence_judge:
            judge_fn = confidence_judge_r1
        else:
            judge_fn = mock_judge_random if args.mock_judge else None

        debate = DebateParadigm(
            task=self.task,
            tokenizer=client.tokenizer,
            sample_token_prompts=client.sample_token_prompts,
            config=DebateConfig.cheap(),
            judge_fn=judge_fn,
        )
        out = await debate.rollout(step=step, num_groups=int(args.num_groups), group_size=int(group_size), seed=args.seed)

        for r in out.debates:
            self.ctx.log_fns.save_debate_log(r, self.ctx.log_dir, config=DebateConfig.cheap(), model_name=client.model_name)

        stats = compute_training_stats(out.debates)
        console.print(f"[dim]A:{stats['a_wins']} B:{stats['b_wins']} Invalid:{stats['invalid']}[/dim]")

        if args.debate_train == "rejection":
            training_data = assemble_training_data(out.debates)
            info_lines = [*out.info_lines, f"Training data: {len(training_data)} datums (winners) from {len(out.debates)} debates"]
        elif args.debate_train == "rs_grpo":
            if args.debate_grpo_reward != "task":
                raise ValueError("--debate-grpo-reward must be 'task' for rs_grpo")

            def reward_fn(traj, _debate):
                return float(traj.metrics["task_reward"])

            training_data = assemble_training_data_grpo(out.debates, reward_fn=reward_fn)
            info_lines = [*out.info_lines, f"Training data: {len(training_data)} datums (winners, GRPO) from {len(out.debates)} debates"]
        else:
            raise ValueError(f"Unknown --debate-train={args.debate_train!r}")

        if len(training_data) == 0:
            raise RuntimeError("Training data is empty (nothing to train on).")

        all_advs = [a for d in training_data for a in d.completion_advantages]
        nonzero_advs = [a for a in all_advs if a != 0.0]
        if len(nonzero_advs) == 0:
            raise RuntimeError("All token advantages are 0.0 (no learning signal).")
        info_lines.append(
            f"Advantage stats (nonzero): n={len(nonzero_advs)} min={min(nonzero_advs):.6g} "
            f"max={max(nonzero_advs):.6g} mean={sum(nonzero_advs)/len(nonzero_advs):.6g}"
        )

        return RolloutOutput(
            training_data=training_data,
            rollout_time_s=float(out.rollout_time_s),
            num_rollouts=int(args.num_rollouts),
            info_lines=info_lines,
        )
