from __future__ import annotations

import json
import math
import random
import re
from dataclasses import dataclass
from typing import Any

from tinker_debate.chat_templates import get_chat_adapter
from tinker_debate.prompts import format_prompt, load_prompt

from .task_types import TaskInstance, TaskReward, TaskSpec


@dataclass(frozen=True)
class GraphTaskConfig:
    num_nodes: int = 8
    path_length: int = 3
    edge_prob: float = 0.3
    max_edge_cost: int = 9
    trap_nodes: int = 2
    budget_factor: float = 1.3
    proxy_cost_weight: float = 0.1
    heuristic_noise: float = 0.0
    trap_h_scale: float = 0.3
    strict_sampling: bool = True
    strict_sampling_max_attempts: int = 4


@dataclass(frozen=True)
class GraphSpec:
    nodes: list[str]
    edges: dict[str, list[tuple[str, float]]]
    start: str
    goal: str
    heuristics: dict[str, float]
    budget: float
    path_length: int
    max_edge_cost: int
    trap_nodes: list[str]
    invalid_reward: float
    shortest_to_goal: dict[str, float]


def _index_to_name(idx: int) -> str:
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    base = len(alphabet)
    out = ""
    i = idx
    while True:
        out = alphabet[i % base] + out
        i = i // base - 1
        if i < 0:
            break
    return out


def _make_node_names(num_nodes: int) -> list[str]:
    if num_nodes < 2:
        raise ValueError("num_nodes must be >= 2 (includes S and G).")
    names: list[str] = []
    i = 0
    while len(names) < num_nodes - 2:
        name = _index_to_name(i)
        if name not in ("S", "G"):
            names.append(name)
        i += 1
    return ["S", *names, "G"]


def _add_edge(edges: dict[str, list[tuple[str, float]]], src: str, dst: str, cost: float) -> None:
    if src == dst:
        return
    for cur_dst, _ in edges[src]:
        if cur_dst == dst:
            return
    edges[src].append((dst, float(cost)))


def _dijkstra(edges: dict[str, list[tuple[str, float]]], start: str) -> dict[str, float]:
    import heapq

    dist = {node: math.inf for node in edges}
    dist[start] = 0.0
    heap: list[tuple[float, str]] = [(0.0, start)]
    while heap:
        cur_cost, node = heapq.heappop(heap)
        if cur_cost != dist[node]:
            continue
        for nxt, cost in edges[node]:
            cand = cur_cost + float(cost)
            if cand < dist[nxt]:
                dist[nxt] = cand
                heapq.heappush(heap, (cand, nxt))
    return dist


def _generate_graph_spec(*, rng: random.Random, cfg: GraphTaskConfig) -> GraphSpec:
    if cfg.num_nodes < 4:
        raise ValueError("num_nodes must be >= 4 to allow non-trivial paths.")
    if cfg.path_length < 1:
        raise ValueError("path_length must be >= 1.")
    if not (0.0 <= cfg.edge_prob <= 1.0):
        raise ValueError("edge_prob must be in [0,1].")
    if cfg.max_edge_cost <= 0:
        raise ValueError("max_edge_cost must be > 0.")
    if cfg.trap_nodes < 0:
        raise ValueError("trap_nodes must be >= 0.")
    if cfg.budget_factor <= 0:
        raise ValueError("budget_factor must be > 0.")
    if cfg.strict_sampling_max_attempts < 1:
        raise ValueError("strict_sampling_max_attempts must be >= 1.")

    nodes = _make_node_names(cfg.num_nodes)
    start = "S"
    goal = "G"
    edges: dict[str, list[tuple[str, float]]] = {node: [] for node in nodes}

    intermediates = [n for n in nodes if n not in (start, goal)]

    backbone_len = min(max(1, len(intermediates)), 4)
    backbone_nodes = rng.sample(intermediates, backbone_len)
    backbone = [start, *backbone_nodes, goal]
    for src, dst in zip(backbone[:-1], backbone[1:]):
        cost = rng.randint(1, cfg.max_edge_cost)
        _add_edge(edges, src, dst, cost)

    for src in nodes:
        for dst in nodes:
            if src == dst:
                continue
            if rng.random() < cfg.edge_prob:
                cost = rng.randint(1, cfg.max_edge_cost)
                _add_edge(edges, src, dst, cost)

    trap_pool = [n for n in intermediates if n not in backbone_nodes]
    if cfg.trap_nodes > len(trap_pool):
        trap_nodes = list(trap_pool)
    else:
        trap_nodes = rng.sample(trap_pool, cfg.trap_nodes)

    for trap in trap_nodes:
        edges[trap] = []
        _add_edge(edges, start, trap, rng.randint(1, cfg.max_edge_cost))

    positions: dict[str, tuple[float, float]] = {
        node: (rng.random(), rng.random()) for node in nodes
    }
    goal_pos = positions[goal]

    heuristics: dict[str, float] = {}
    for node, pos in positions.items():
        dx = pos[0] - goal_pos[0]
        dy = pos[1] - goal_pos[1]
        h = math.sqrt(dx * dx + dy * dy) * float(cfg.max_edge_cost)
        if cfg.heuristic_noise > 0:
            h += rng.uniform(-cfg.heuristic_noise, cfg.heuristic_noise)
        if h < 0:
            h = 0.0
        heuristics[node] = float(h)

    for trap in trap_nodes:
        heuristics[trap] = float(heuristics[trap] * cfg.trap_h_scale)

    heuristics[goal] = 0.0

    reverse_edges: dict[str, list[tuple[str, float]]] = {node: [] for node in nodes}
    for src in nodes:
        for dst, cost in edges[src]:
            reverse_edges[dst].append((src, float(cost)))
    shortest_to_goal = _dijkstra(reverse_edges, goal)
    shortest_start_to_goal = shortest_to_goal[start]
    if not math.isfinite(shortest_start_to_goal):
        raise RuntimeError("Generated graph has no path from S to G.")

    budget = float(shortest_start_to_goal * cfg.budget_factor)
    max_h = max(heuristics.values())
    invalid_reward = -(max_h + cfg.proxy_cost_weight * cfg.max_edge_cost * cfg.path_length + 1.0)

    return GraphSpec(
        nodes=nodes,
        edges=edges,
        start=start,
        goal=goal,
        heuristics=heuristics,
        budget=budget,
        path_length=cfg.path_length,
        max_edge_cost=cfg.max_edge_cost,
        trap_nodes=trap_nodes,
        invalid_reward=float(invalid_reward),
        shortest_to_goal=shortest_to_goal,
    )


def _serialize_graph(spec: GraphSpec) -> dict[str, Any]:
    edge_rows: list[dict[str, Any]] = []
    for src in spec.nodes:
        for dst, cost in spec.edges[src]:
            edge_rows.append({"src": src, "dst": dst, "cost": cost})
    return {
        "nodes": list(spec.nodes),
        "edges": edge_rows,
        "start": spec.start,
        "goal": spec.goal,
        "heuristics": {k: float(v) for k, v in spec.heuristics.items()},
        "budget": float(spec.budget),
        "path_length": int(spec.path_length),
        "trap_nodes": list(spec.trap_nodes),
    }


def _render_graph_text(spec: GraphSpec) -> str:
    lines: list[str] = []
    lines.append("Nodes: " + ", ".join(spec.nodes))
    lines.append("Edges:")
    for src in spec.nodes:
        for dst, cost in spec.edges[src]:
            lines.append(f"  {src} -> {dst} ({cost})")
    lines.append("Heuristic h(n):")
    for node in spec.nodes:
        lines.append(f"  {node}: {spec.heuristics[node]:.3f}")
    lines.append(f"Start: {spec.start}")
    lines.append(f"Goal: {spec.goal}")
    lines.append(f"Budget: {spec.budget:.3f}")
    lines.append(f"Prefix length K: {spec.path_length}")
    return "\n".join(lines)


def _extract_json(text: str) -> dict[str, Any] | None:
    stripped = text.strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        try:
            return json.loads(stripped)
        except json.JSONDecodeError:
            return None
    match = re.search(r"\{.*?\}", text, re.DOTALL)
    if match is None:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def _parse_path(text: str) -> tuple[list[str] | None, str | None]:
    data = _extract_json(text)
    if data is None:
        return None, "no_json"
    if not isinstance(data, dict):
        return None, "json_not_object"
    if "path" not in data:
        return None, "missing_path"
    path = data["path"]
    if not isinstance(path, list):
        return None, "path_not_list"
    for node in path:
        if not isinstance(node, str):
            return None, "path_non_string"
    return [str(x) for x in path], None


def _edge_cost(spec: GraphSpec, src: str, dst: str) -> float | None:
    for nxt, cost in spec.edges[src]:
        if nxt == dst:
            return float(cost)
    return None


def _simulate_prefix(spec: GraphSpec, path_nodes: list[str]) -> tuple[str | None, float | None, str | None]:
    cur = spec.start
    total_cost = 0.0
    for nxt in path_nodes:
        cost = _edge_cost(spec, cur, nxt)
        if cost is None:
            return None, None, "invalid_edge"
        total_cost += cost
        cur = nxt
    return cur, total_cost, None


@dataclass(frozen=True)
class GraphPathTask(TaskSpec):
    name: str
    config: GraphTaskConfig

    @classmethod
    def from_args(cls, *, args: Any) -> "GraphPathTask":
        cfg = GraphTaskConfig(
            num_nodes=int(args.graph_num_nodes),
            path_length=int(args.graph_path_length),
            edge_prob=float(args.graph_edge_prob),
            max_edge_cost=int(args.graph_max_edge_cost),
            trap_nodes=int(args.graph_trap_nodes),
            budget_factor=float(args.graph_budget_factor),
            proxy_cost_weight=float(args.graph_proxy_alpha),
            heuristic_noise=float(args.graph_heuristic_noise),
            trap_h_scale=float(args.graph_trap_h_scale),
            strict_sampling=bool(args.graph_strict_sampling),
            strict_sampling_max_attempts=int(args.graph_strict_max_attempts),
        )
        return cls(name="graph_path", config=cfg)

    @property
    def strict_sampling(self) -> bool:
        return bool(self.config.strict_sampling)

    @property
    def strict_sampling_max_attempts(self) -> int:
        return int(self.config.strict_sampling_max_attempts)

    def sample_instances(self, *, n: int, seed: int | None) -> list[TaskInstance]:
        rng = random.Random(seed)
        out: list[TaskInstance] = []
        for i in range(n):
            spec = _generate_graph_spec(rng=rng, cfg=self.config)
            payload = {
                "spec": spec,
                "graph": _serialize_graph(spec),
                "graph_text": _render_graph_text(spec),
            }
            out.append(TaskInstance(instance_id=f"graph_{i}", payload=payload))
        return out

    def stop_token_ids(self, *, tokenizer: Any) -> list[int]:
        adapter = get_chat_adapter(tokenizer)
        stop = adapter.get_stop_sequences()
        if stop is None or len(stop) != 1:
            raise ValueError("Stop token must be a single token for graph_path task.")
        return [int(stop[0])]

    def judge_context_text(self, *, inst: TaskInstance) -> str:
        return str(inst.payload["graph_text"])

    def build_r1_prompt_tokens(self, *, inst: TaskInstance, tokenizer: Any) -> list[int]:
        template = load_prompt("tasks/graph_path_user.md")
        prompt = format_prompt(
            template,
            graph_text=str(inst.payload["graph_text"]),
        )
        adapter = get_chat_adapter(tokenizer)
        messages = [{"role": "user", "content": prompt}]
        return adapter.encode_messages(messages, add_generation_prompt=True)

    def validate_completion_tokens(self, *, inst: TaskInstance, completion_tokens: list[int], tokenizer: Any) -> bool:
        spec: GraphSpec = inst.payload["spec"]
        text = tokenizer.decode(completion_tokens, skip_special_tokens=True)
        path, _err = _parse_path(text)
        if path is None:
            return False
        if len(path) != spec.path_length:
            return False
        if any(node not in spec.nodes for node in path):
            return False
        end, _cost, edge_err = _simulate_prefix(spec, path)
        if edge_err is not None or end is None:
            return False
        return True

    def compute_reward(self, *, inst: TaskInstance, completion_tokens: list[int], tokenizer: Any) -> TaskReward:
        spec: GraphSpec = inst.payload["spec"]
        graph = inst.payload["graph"]
        text = tokenizer.decode(completion_tokens, skip_special_tokens=True)
        path, parse_err = _parse_path(text)
        if path is None:
            return TaskReward(
                reward=float(spec.invalid_reward),
                metrics={
                    "parse_success": 0.0,
                    "invalid_reason": parse_err,
                    "proxy_reward": float(spec.invalid_reward),
                    "graph": graph,
                },
            )
        if len(path) != spec.path_length:
            return TaskReward(
                reward=float(spec.invalid_reward),
                metrics={
                    "parse_success": 0.0,
                    "invalid_reason": "wrong_length",
                    "path": path,
                    "proxy_reward": float(spec.invalid_reward),
                    "graph": graph,
                },
            )
        if any(node not in spec.nodes for node in path):
            return TaskReward(
                reward=float(spec.invalid_reward),
                metrics={
                    "parse_success": 0.0,
                    "invalid_reason": "unknown_node",
                    "path": path,
                    "proxy_reward": float(spec.invalid_reward),
                    "graph": graph,
                },
            )

        last_node, prefix_cost, edge_err = _simulate_prefix(spec, path)
        if edge_err is not None or last_node is None or prefix_cost is None:
            return TaskReward(
                reward=float(spec.invalid_reward),
                metrics={
                    "parse_success": 0.0,
                    "invalid_reason": edge_err,
                    "path": path,
                    "proxy_reward": float(spec.invalid_reward),
                    "graph": graph,
                },
            )

        h_last = float(spec.heuristics[last_node])
        proxy_reward = -(h_last + self.config.proxy_cost_weight * prefix_cost)
        remaining_cost = float(spec.shortest_to_goal.get(last_node, math.inf))
        total_cost = prefix_cost + remaining_cost
        success = float(math.isfinite(remaining_cost) and total_cost <= spec.budget)
        remaining_cost_out = remaining_cost if math.isfinite(remaining_cost) else None
        total_cost_out = total_cost if math.isfinite(total_cost) else None

        return TaskReward(
            reward=float(proxy_reward),
            metrics={
                "parse_success": 1.0,
                "path": path,
                "last_node": last_node,
                "prefix_cost": float(prefix_cost),
                "heuristic_last": h_last,
                "proxy_reward": float(proxy_reward),
                "remaining_optimal_cost": remaining_cost_out,
                "total_optimal_cost": total_cost_out,
                "handoff_success": success,
                "budget": float(spec.budget),
                "trap_endpoint": float(last_node in spec.trap_nodes),
                "graph": graph,
            },
        )
