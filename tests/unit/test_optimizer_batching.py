from dataclasses import replace

import pytest

from llm_local_rl.checkpointing import config_fingerprint
from llm_local_rl.config import TrainRunConfig
from llm_local_rl.optimizer_batching import pack_optimizer_groups
from llm_local_rl.types import TrainExample
from scripts.run_train import parse_args


def row(group):
    return TrainExample("debate", [0], [0], [1], [1], [0.], [1.],
                        metadata={"optimizer_group_id": group})


@pytest.mark.parametrize("group_size", [2, 16])
def test_interleaved_groups_pack_into_four_complete_batches(group_size):
    rows = [row(str(group)) for _ in range(group_size) for group in range(128 // group_size)]
    batches = pack_optimizer_groups(rows, max_rows=32)
    assert [len(batch) for batch in batches] == [32] * 4
    assert [{r.metadata["optimizer_group_id"] for r in batch} for batch in batches] == [
        {str(g) for g in range(i * (32 // group_size), (i + 1) * (32 // group_size))} for i in range(4)]
    assert sorted(map(id, rows)) == sorted(id(r) for batch in batches for r in batch)


def test_unequal_groups_leave_short_batches_instead_of_splitting():
    rows = [row(str(g)) for g, size in enumerate([20, 20, 8]) for _ in range(size)]
    assert [len(b) for b in pack_optimizer_groups(rows, max_rows=32)] == [20, 28]
    with pytest.raises(ValueError, match="cannot split"):
        pack_optimizer_groups(rows, max_rows=16)
    with pytest.raises(ValueError, match="every row"):
        pack_optimizer_groups([replace(rows[0], metadata={})], max_rows=32)


def test_group_batch_flags_and_exact_resume_identity():
    args = parse_args(["--model-path", "/model", "--output-dir", "/out",
                       "--train-minibatch-size", "32", "--train-optimizer-batch-size", "32",
                       "--train-keep-groups-together"])
    config = TrainRunConfig(model_path=args.model_path, output_dir=args.output_dir,
                            adapter_layout="split", train_keep_groups_together=args.train_keep_groups_together,
                            train_minibatch_size=args.train_minibatch_size,
                            train_optimizer_batch_size=args.train_optimizer_batch_size)
    assert TrainRunConfig.from_dict(config.to_dict()) == config
    default = replace(config, train_keep_groups_together=False).to_dict()
    legacy = dict(default)
    del legacy["train_keep_groups_together"]
    assert config_fingerprint(default) == config_fingerprint(legacy)
    assert config_fingerprint(config.to_dict()) != config_fingerprint(legacy)
    assert TrainRunConfig.from_dict(legacy).train_keep_groups_together is False
