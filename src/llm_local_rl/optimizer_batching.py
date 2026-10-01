"""Pack complete rollout/advantage groups before physical length bucketing."""

from llm_local_rl.types import TrainExample


def pack_optimizer_groups(batch: list[TrainExample], *, max_rows: int) -> list[list[TrainExample]]:
    if max_rows <= 0:
        raise ValueError("Optimizer group capacity must be positive")
    groups: dict[str, list[TrainExample]] = {}
    for row in batch:
        group_id = row.metadata.get("optimizer_group_id")
        if not isinstance(group_id, str) or not group_id:
            raise ValueError("Group-preserving optimizer batches require optimizer_group_id on every row")
        groups.setdefault(group_id, []).append(row)
    result: list[list[TrainExample]] = []
    current: list[TrainExample] = []
    for group_id, rows in groups.items():
        if len(rows) > max_rows:
            raise ValueError(f"Optimizer group {group_id!r} has {len(rows)} rows, exceeding capacity {max_rows}; cannot split it")
        if current and len(current) + len(rows) > max_rows:
            result.append(current)
            current = []
        current.extend(rows)
    if current:
        result.append(current)
    return result
