from __future__ import annotations

from dataclasses import dataclass
import os

from tinker_debate.debate_types import assemble_training_data_grpo, assemble_training_data_r1_r23
from tinker_debate.paradigms.debate import DebateParadigm
from tinker_debate.paradigms.normal import NormalParadigm
from tinker_debate.summary.rewards import RewardConfig
from tinker_debate.tasks.coin_task import CoinTask
from tinker_debate.tasks.constrained_writing_task import ConstrainedWritingTask
from tinker_debate.tasks.confidence_task import ConfidenceTask
from tinker_debate.tasks.qa_task import QATask
from tinker_debate.tasks.graph_path_task import GraphPathTask
from tinker_debate.tasks.secret_word_debate_task import SecretWordDebateTask
from tinker_debate.tasks.summary_task import SummaryTask
from tinker_debate.local_renderers import infer_chat_preamble

from .driver_context import DriverContext
from .driver_types import RolloutOutput, RolloutDriver


@dataclass
class OrthogonalDriver(RolloutDriver):
    ctx: DriverContext
    chat_preamble: str = ""

    def __post_init__(self) -> None:
        args = self.ctx.args
        self.chat_preamble = infer_chat_preamble(self.ctx.client.tokenizer)

        prompt_style = os.environ.get("TINKER_PROMPT_STYLE", "").lower()
        if args.mode == "single_turn":
            if args.env is None:
                raise ValueError("--env is required when --mode=single_turn")
            task_name = args.env
        elif args.mode == "debate":
            task_name = "qa" if args.env is None else args.env
        else:
            raise ValueError(f"Unknown --mode={args.mode!r}")

        if prompt_style == "base":
            if args.mode != "single_turn" or task_name != "qa":
                raise ValueError("TINKER_PROMPT_STYLE=base is only supported for --mode single_turn --env qa.")

        if task_name == "summary":
            if args.dataset not in (None, "cnn_dailymail"):
                raise ValueError("--dataset must be cnn_dailymail for summary task")
            if args.replay_dir is not None:
                self.task = SummaryTask(
                    name="summary",
                    articles=[],
                    reward_config=RewardConfig.from_string(args.reward_fn),
                )
            else:
                n_samples = 1000
                self.task = SummaryTask.from_args(reward_fn=args.reward_fn, seed=args.seed, n_samples=n_samples)
            self.normal_max_tokens = int(args.max_tokens)
            self.normal_temperature = float(args.temperature)
        elif task_name == "confidence":
            self.task = ConfidenceTask.from_file()
            self.normal_max_tokens = int(args.max_tokens)
            self.normal_temperature = float(args.temperature)
        elif task_name == "coin":
            self.task = CoinTask.create(target_color=str(args.coin_target))
            self.normal_max_tokens = int(args.max_tokens)
            self.normal_temperature = float(args.temperature)
        elif task_name == "qa":
            if args.dataset == "cnn_dailymail":
                raise ValueError("--dataset=cnn_dailymail is not a QA dataset")
            self.task = QATask.from_args(dataset_name=args.dataset, seed=args.seed)
            self.normal_max_tokens = int(args.max_tokens)
            self.normal_temperature = float(args.temperature)
        elif task_name == "graph_path":
            self.task = GraphPathTask.from_args(args=args)
            self.normal_max_tokens = int(args.max_tokens)
            self.normal_temperature = float(args.temperature)
        elif task_name == "constrained_writing":
            self.task = ConstrainedWritingTask.from_args(
                rules_per_speaker=int(args.constraint_rules_per_speaker),
                reward_scope=str(args.constraint_reward_scope),
                sides=str(args.constraint_sides),
            )
            self.normal_max_tokens = int(args.max_tokens)
            self.normal_temperature = float(args.temperature)
        elif task_name == "secret_word":
            self.task = SecretWordDebateTask()
            self.normal_max_tokens = int(args.max_tokens)
            self.normal_temperature = float(args.temperature)
        else:
            raise ValueError(f"Unknown task {task_name!r}")

    async def rollout_step(self, *, step: int) -> RolloutOutput:
        args = self.ctx.args
        client = self.ctx.client
        console = self.ctx.console

        if args.mode == "single_turn":
            accept_min_reward = float(args.accept_min_reward)
            if self.task.name == "graph_path" and accept_min_reward == 0.0:
                accept_min_reward = -1.0e9

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
                accept_min_reward=accept_min_reward,
                accept_require_parse=bool(args.accept_require_parse),
                replay_dir=args.replay_dir,
            )
            out = await normal.rollout(
                step=step,
                num_rollouts=int(args.num_rollouts),
                num_groups=int(args.num_groups),
                seed=args.seed,
                max_tokens=int(self.normal_max_tokens),
                temperature=float(self.normal_temperature),
                min_p=float(args.min_p),
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
            config=DebateConfig.cheap(chat_preamble=self.chat_preamble),
            judge_fn=judge_fn,
        )
        out = await debate.rollout(step=step, num_groups=int(args.num_groups), group_size=int(group_size), seed=args.seed)

        for r in out.debates:
            self.ctx.log_fns.save_debate_log(
                r,
                self.ctx.log_dir,
                config=DebateConfig.cheap(chat_preamble=self.chat_preamble),
                model_name=client.model_name,
            )

        stats = compute_training_stats(out.debates)
        console.print(f"[dim]A:{stats['a_wins']} B:{stats['b_wins']} Invalid:{stats['invalid']}[/dim]")

        if args.debate_reward != "task":
            raise ValueError("--debate-reward must be 'task'")

        def reward_fn(traj, _debate):
            # Task reward cached in metrics by rollout: 1 if R1 matched ground truth else 0.
            return float(traj.metrics["task_reward"])

        r1_reward_fn = reward_fn if args.debate_r1_reward == "task" else (lambda _t, _d: 0.0)
        r23_reward = 0.0 if args.debate_r23_reward == "none" else float(args.debate_r23_constant)
        training_data = assemble_training_data_r1_r23(
            out.debates,
            r1_reward_fn=r1_reward_fn,
            r23_reward=r23_reward,
            r23_symmetric=(args.debate_r23_mode == "symmetric"),
        )
        r23_label = "none" if args.debate_r23_reward == "none" else f"{args.debate_r23_mode}:{r23_reward}"
        info_lines = [
            *out.info_lines,
            f"Training data: {len(training_data)} datums (R1 z-scored all solutions + R2/R3={r23_label}) from {len(out.debates)} debates",
        ]

        if len(training_data) == 0:
            raise RuntimeError("Training data is empty (nothing to train on).")

        all_advs = [a for d in training_data for a in d.completion_advantages]
        nonzero_advs = [a for a in all_advs if a != 0.0]
        if len(nonzero_advs) == 0:
            if args.dry_run:
                info_lines.append("All token advantages are 0.0 (no learning signal).")
                return RolloutOutput(
                    training_data=training_data,
                    rollout_time_s=float(out.rollout_time_s),
                    num_rollouts=int(args.num_rollouts),
                    info_lines=info_lines,
                )
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
