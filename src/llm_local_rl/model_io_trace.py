from __future__ import annotations

from contextlib import contextmanager
import contextvars
from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime, timezone
from functools import wraps
import json
from pathlib import Path
import threading
import traceback
from typing import Any, Callable
import uuid


@dataclass(frozen=True)
class TraceConfig:
    enabled: bool = False
    output_dir: str | Path | None = None
    include_decoded: bool = True
    top_logprobs: int = 0
    jsonl_name: str = "model_io_trace.jsonl"
    metadata: dict[str, Any] | None = None


_TRACE_CONTEXT: contextvars.ContextVar[dict[str, Any]] = contextvars.ContextVar(
    "llm_local_rl_model_io_trace_context",
    default={},
)
_GLOBAL_TRACER: ModelIOTracer | None = None


@contextmanager
def trace_context(**metadata: Any):
    current = dict(_TRACE_CONTEXT.get())
    current.update({key: value for key, value in metadata.items() if value is not None})
    token = _TRACE_CONTEXT.set(current)
    try:
        yield
    finally:
        _TRACE_CONTEXT.reset(token)


def configure_model_io_tracing(
    *,
    enabled: bool,
    output_dir: str | Path | None,
    tokenizer: Any | None = None,
    top_logprobs: int = 0,
    include_decoded: bool = True,
    metadata: dict[str, Any] | None = None,
) -> "ModelIOTracer":
    global _GLOBAL_TRACER
    _GLOBAL_TRACER = ModelIOTracer(
        config=TraceConfig(
            enabled=enabled,
            output_dir=output_dir,
            include_decoded=include_decoded,
            top_logprobs=max(0, int(top_logprobs)),
            metadata=metadata,
        ),
        tokenizer=tokenizer,
    )
    return _GLOBAL_TRACER


def reset_model_io_tracing() -> None:
    global _GLOBAL_TRACER
    _GLOBAL_TRACER = None


def get_model_io_tracer() -> "ModelIOTracer":
    global _GLOBAL_TRACER
    if _GLOBAL_TRACER is None:
        _GLOBAL_TRACER = ModelIOTracer(config=TraceConfig(enabled=False), tokenizer=None)
    return _GLOBAL_TRACER


def is_model_io_tracing_enabled() -> bool:
    return get_model_io_tracer().enabled


def get_trace_top_logprobs() -> int:
    tracer = get_model_io_tracer()
    if not tracer.enabled:
        return 0
    return tracer.config.top_logprobs


def trace_model_io(
    *,
    phase: str,
    boundary: str | None = None,
    request_serializer: Callable[..., Any] | None = None,
    response_serializer: Callable[[Any], Any] | None = None,
):
    def decorate(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            tracer = get_model_io_tracer()
            if not tracer.enabled:
                return func(*args, **kwargs)
            resolved_boundary = boundary or f"{func.__module__}.{func.__qualname__}"
            request_payload = None
            if request_serializer is not None:
                request_payload = request_serializer(*args, **kwargs)
            else:
                request_payload = {"args": args, "kwargs": kwargs}
            try:
                response = func(*args, **kwargs)
            except Exception as exc:
                tracer.record_custom(
                    phase=phase,
                    boundary=resolved_boundary,
                    request=request_payload,
                    response=None,
                    error=exc,
                )
                raise
            tracer.record_custom(
                phase=phase,
                boundary=resolved_boundary,
                request=request_payload,
                response=response_serializer(response) if response_serializer is not None else response,
                error=None,
            )
            return response

        return wrapper

    return decorate


class ModelIOTracer:
    def __init__(self, *, config: TraceConfig, tokenizer: Any | None = None) -> None:
        self.config = config
        self.tokenizer = tokenizer
        self._lock = threading.Lock()
        self.output_dir = Path(config.output_dir) if config.output_dir is not None else None
        self.path = None
        if self.enabled:
            if self.output_dir is None:
                raise ValueError("Trace output_dir is required when model I/O tracing is enabled.")
            self.output_dir.mkdir(parents=True, exist_ok=True)
            self.path = self.output_dir / self.config.jsonl_name
            self.write_viewer()

    @property
    def enabled(self) -> bool:
        return bool(self.config.enabled)

    def set_tokenizer(self, tokenizer: Any | None) -> None:
        self.tokenizer = tokenizer

    def write_viewer(self) -> None:
        if self.output_dir is None:
            return
        (self.output_dir / "index.html").write_text(_VIEWER_HTML)

    def record_custom(
        self,
        *,
        phase: str,
        boundary: str,
        request: Any,
        response: Any,
        error: BaseException | None,
    ) -> None:
        if not self.enabled:
            return
        self._write_record(
            {
                "phase": phase,
                "boundary": boundary,
                "request": _json_safe(request),
                "response": _json_safe(response),
                "error": _error_payload(error),
            }
        )

    def record_generation(self, *, request: Any, result: Any, boundary: str) -> None:
        if not self.enabled:
            return
        prompt_ids = list(getattr(request, "prompt_token_ids", []))
        completion_ids = list(getattr(result, "completion_token_ids", []))
        raw = dict(getattr(result, "raw", {}) or {})
        provider = str(raw["sampler_backend"]) if "sampler_backend" in raw else "vllm"
        top_logprobs_by_pos = raw.pop("completion_top_logprobs", None)
        completion_logprobs = list(getattr(result, "completion_logprobs", []))
        completion_logprob_semantics = str(
            getattr(result, "completion_logprob_semantics", "unspecified")
        )

        completion_rows = []
        for idx, token_id in enumerate(completion_ids):
            row = {
                "index": idx,
                "token_id": int(token_id),
                "text": self._decode_token(token_id),
            }
            if idx < len(completion_logprobs):
                row["logprob"] = float(completion_logprobs[idx])
            if top_logprobs_by_pos is not None and idx < len(top_logprobs_by_pos):
                row["top_logprobs"] = self._top_logprob_rows(top_logprobs_by_pos[idx])
            completion_rows.append(row)

        payload = {
            "phase": "generation",
            "boundary": boundary,
            "model": {
                "provider": provider,
                "adapter_name": getattr(request, "adapter_name", None),
                **dict(self.config.metadata or {}),
            },
            "request": {
                "adapter_name": getattr(request, "adapter_name", None),
                "prompt_token_ids": prompt_ids,
                "prompt_tokens": self._token_rows(prompt_ids),
                "prompt_text": self._decode_tokens(prompt_ids),
                "stop_token_ids": list(getattr(request, "stop_token_ids", [])),
                "max_tokens": getattr(request, "max_tokens", None),
                "temperature": getattr(request, "temperature", None),
                "seed": getattr(request, "seed", None),
                "min_p": getattr(request, "min_p", None),
            },
            "response": {
                "completion_token_ids": completion_ids,
                "completion_logprobs": completion_logprobs,
                "completion_logprob_semantics": completion_logprob_semantics,
                "completion_tokens": completion_rows,
                "completion_text": getattr(result, "text", ""),
                "raw": _json_safe(raw),
            },
            "exactness": {
                "prompt_token_ids": "actual_model_input",
                "completion_token_ids": "actual_model_output",
                "completion_logprobs": completion_logprob_semantics,
                "top_logprobs": f"{provider}_generated_token_alternatives"
                if top_logprobs_by_pos is not None
                else "not_available",
                "decoded_text": "local_tokenizer_decode",
            },
        }
        self._write_record(payload)

    def record_external_judge(
        self,
        *,
        url: str,
        prompt_text: str,
        request_body: dict[str, Any],
        raw_text: str | None,
        verdict: str | None,
        harness_id: str,
        harness_fingerprint: str,
        error: BaseException | None = None,
    ) -> None:
        if not self.enabled:
            return
        prompt_token_ids = self._encode_text_for_visualization(prompt_text)
        self._write_record(
            {
                "phase": "external_judge",
                "boundary": "llm_local_rl.base_model_judge.remote_judge",
                "model": {
                    **dict(self.config.metadata or {}),
                    "provider": "external_http_judge",
                    "judge_harness_id": harness_id,
                    "judge_harness_fingerprint": harness_fingerprint,
                },
                "request": {
                    "url": url,
                    "body": request_body,
                    "prompt_text": prompt_text,
                    "prompt_token_ids": prompt_token_ids,
                    "prompt_tokens": self._token_rows(prompt_token_ids),
                },
                "response": {
                    "raw_text": raw_text,
                    "verdict": verdict,
                },
                "exactness": {
                    "prompt_text": "actual_http_request_body",
                    "response_text": "actual_http_response_body" if raw_text is not None else "not_available",
                    "token_ids": "local_visualization_only",
                },
                "error": _error_payload(error),
            }
        )

    def record_trainer_forward(
        self,
        *,
        phase: str,
        boundary: str,
        adapter_name: str,
        batch: list[Any],
        tensors: dict[str, Any],
        minibatch_start: int,
        token_logprobs: Any | None = None,
        top_token_ids: Any | None = None,
        top_logprobs: Any | None = None,
    ) -> None:
        if not self.enabled:
            return
        rows = []
        for row_idx, example in enumerate(batch):
            n = len(getattr(example, "input_ids", []))
            input_ids = _row_values(tensors.get("input_ids"), row_idx, n)
            target_ids = _row_values(tensors.get("target_ids"), row_idx, n)
            target_top_logprobs = None
            if top_token_ids is not None and top_logprobs is not None:
                target_top_logprobs = []
                top_ids_by_pos = _row_values(top_token_ids, row_idx, n)
                top_lps_by_pos = _row_values(top_logprobs, row_idx, n)
                for pos, (ids_at_pos, lps_at_pos) in enumerate(zip(top_ids_by_pos, top_lps_by_pos, strict=True)):
                    target_top_logprobs.append(self._ranked_token_rows(ids_at_pos, lps_at_pos, position=pos))
            rows.append(
                {
                    "example_index": minibatch_start + row_idx,
                    "metadata": _json_safe(getattr(example, "metadata", {})),
                    "input_ids": input_ids,
                    "input_tokens": self._token_rows(input_ids),
                    "target_ids": target_ids,
                    "target_tokens": self._token_rows(target_ids),
                    "attention_mask": _row_values(tensors.get("attention_mask"), row_idx, n),
                    "loss_mask": _row_values(tensors.get("loss_mask"), row_idx, n),
                    "behavior_logprob_mask": _row_values(
                        tensors.get("behavior_logprob_mask"),
                        row_idx,
                        n,
                    ),
                    "old_logprobs": _row_values(tensors.get("old_logprobs"), row_idx, n),
                    "advantages": _row_values(tensors.get("advantages"), row_idx, n),
                    "target_logprobs": _row_values(token_logprobs, row_idx, n) if token_logprobs is not None else None,
                    "target_top_logprobs": target_top_logprobs,
                }
            )
        self._write_record(
            {
                "phase": phase,
                "boundary": boundary,
                "model": {
                    "provider": "transformers",
                    "adapter_name": adapter_name,
                    **dict(self.config.metadata or {}),
                },
                "request": {
                    "adapter_name": adapter_name,
                    "minibatch_start": minibatch_start,
                    "num_examples": len(batch),
                    "rows": rows,
                },
                "response": {
                    "target_logprobs_recorded": token_logprobs is not None,
                    "top_logprobs_recorded": top_token_ids is not None and top_logprobs is not None,
                },
                "exactness": {
                    "input_ids": "actual_model_input_tensor",
                    "target_ids": "actual_gather_target_tensor",
                    "masks": "actual_training_tensors",
                    "top_logprobs": "transformers_forward_log_softmax_topk"
                    if top_token_ids is not None and top_logprobs is not None
                    else "not_recorded",
                },
            }
        )

    def _write_record(self, payload: dict[str, Any]) -> None:
        if not self.enabled:
            return
        if self.path is None:
            raise ValueError("Trace path is not configured.")
        record = {
            "schema_version": 1,
            "trace_id": uuid.uuid4().hex,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "context": _json_safe(_TRACE_CONTEXT.get()),
            **payload,
        }
        with self._lock:
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(_json_safe(record), ensure_ascii=False) + "\n")

    def _encode_text_for_visualization(self, text: str) -> list[int]:
        if self.tokenizer is None:
            return []
        try:
            return list(self.tokenizer.encode(text, add_special_tokens=False))
        except Exception:
            return []

    def _decode_token(self, token_id: int) -> str | None:
        if not self.config.include_decoded or self.tokenizer is None:
            return None
        try:
            return str(self.tokenizer.decode([int(token_id)], skip_special_tokens=False))
        except TypeError:
            try:
                return str(self.tokenizer.decode([int(token_id)]))
            except Exception:
                return None
        except Exception:
            return None

    def _decode_tokens(self, token_ids: list[int]) -> str | None:
        if not self.config.include_decoded or self.tokenizer is None:
            return None
        try:
            return str(self.tokenizer.decode(list(token_ids), skip_special_tokens=False))
        except TypeError:
            try:
                return str(self.tokenizer.decode(list(token_ids)))
            except Exception:
                return None
        except Exception:
            return None

    def _token_rows(self, token_ids: list[int] | None) -> list[dict[str, Any]]:
        if token_ids is None:
            return []
        return [
            {
                "index": idx,
                "token_id": int(token_id),
                "text": self._decode_token(int(token_id)),
            }
            for idx, token_id in enumerate(token_ids)
        ]

    def _top_logprob_rows(self, alternatives: Any) -> list[dict[str, Any]]:
        rows = []
        for rank, alt in enumerate(alternatives or [], start=1):
            token_id = int(alt["token_id"])
            rows.append(
                {
                    "rank": int(alt.get("rank", rank)),
                    "token_id": token_id,
                    "text": alt.get("text") if alt.get("text") is not None else self._decode_token(token_id),
                    "logprob": float(alt["logprob"]),
                    "chosen": bool(alt.get("chosen", False)),
                }
            )
        return rows

    def _ranked_token_rows(self, token_ids: Any, logprobs: Any, *, position: int) -> list[dict[str, Any]]:
        rows = []
        for rank, (token_id, logprob) in enumerate(zip(token_ids, logprobs, strict=True), start=1):
            token_id_int = int(token_id)
            rows.append(
                {
                    "position": position,
                    "rank": rank,
                    "token_id": token_id_int,
                    "text": self._decode_token(token_id_int),
                    "logprob": float(logprob),
                }
            )
        return rows


def _row_values(matrix: Any, row_idx: int, n: int) -> Any:
    if matrix is None:
        return None
    row = matrix[row_idx]
    if hasattr(row, "detach"):
        row = row.detach().cpu()
    row = row[:n]
    if hasattr(row, "tolist"):
        row = row.tolist()
    return _json_safe(row)


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value):
        return _json_safe(asdict(value))
    if isinstance(value, dict):
        return {str(key): _json_safe(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "detach"):
        return _json_safe(value.detach().cpu().tolist())
    if hasattr(value, "tolist"):
        return _json_safe(value.tolist())
    return repr(value)


def _error_payload(error: BaseException | None) -> dict[str, Any] | None:
    if error is None:
        return None
    return {
        "type": type(error).__name__,
        "message": str(error),
        "traceback": traceback.format_exception(type(error), error, error.__traceback__),
    }


_VIEWER_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Model I/O Trace Viewer</title>
  <style>
    body { font-family: system-ui, sans-serif; margin: 0; background: #f7f7f4; color: #1f2933; }
    header { padding: 16px 20px; border-bottom: 1px solid #d8d8d2; background: #ffffff; position: sticky; top: 0; z-index: 2; }
    main { display: grid; grid-template-columns: 320px 1fr; min-height: calc(100vh - 70px); }
    aside { border-right: 1px solid #d8d8d2; padding: 14px; overflow: auto; background: #fbfbf9; }
    section { padding: 16px; overflow: auto; }
    input, select { width: 100%; box-sizing: border-box; margin: 6px 0 12px; padding: 8px; border: 1px solid #c9c9c0; border-radius: 4px; background: white; }
    button { padding: 7px 9px; border: 1px solid #a7b0b8; border-radius: 4px; background: white; cursor: pointer; }
    .record { padding: 9px; border: 1px solid #dddcd5; border-radius: 6px; margin-bottom: 8px; background: white; cursor: pointer; }
    .record.active { border-color: #2f6f9f; box-shadow: inset 3px 0 #2f6f9f; }
    .muted { color: #6b7280; font-size: 12px; }
    table { border-collapse: collapse; width: 100%; background: white; margin: 12px 0; }
    th, td { border: 1px solid #deded7; padding: 5px 7px; text-align: left; vertical-align: top; font-size: 13px; }
    th { background: #f0f0ea; position: sticky; top: 0; }
    code, pre { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }
    pre { white-space: pre-wrap; background: #ffffff; border: 1px solid #deded7; padding: 10px; border-radius: 6px; }
    .lp-good { background: #e6f4ea; }
    .lp-mid { background: #fff7d6; }
    .lp-low { background: #fde8e8; }
    .toolbar { display: grid; grid-template-columns: repeat(4, minmax(110px, 1fr)); gap: 10px; align-items: end; }
  </style>
</head>
<body>
  <header>
    <strong>Model I/O Trace Viewer</strong>
    <span class="muted">Load model_io_trace.jsonl with the file picker.</span>
    <input id="file" type="file" accept=".jsonl,.json">
    <div class="toolbar">
      <label>Phase<input id="phase" placeholder="generation"></label>
      <label>Adapter<input id="adapter" placeholder="shared"></label>
      <label>Step<input id="step" placeholder="1"></label>
      <label>Search<input id="search" placeholder="trace id or text"></label>
    </div>
  </header>
  <main>
    <aside id="records"></aside>
    <section id="detail"><p class="muted">No trace loaded.</p></section>
  </main>
<script>
let records = [];
let selected = null;
const $ = id => document.getElementById(id);

$("file").addEventListener("change", async event => {
  const file = event.target.files[0];
  if (!file) return;
  const text = await file.text();
  records = text.split(/\\r?\\n/).filter(Boolean).map(line => JSON.parse(line));
  selected = records[0] || null;
  renderList();
  renderDetail();
});
["phase", "adapter", "step", "search"].forEach(id => $(id).addEventListener("input", renderList));

function filteredRecords() {
  const phase = $("phase").value.trim().toLowerCase();
  const adapter = $("adapter").value.trim().toLowerCase();
  const step = $("step").value.trim();
  const search = $("search").value.trim().toLowerCase();
  return records.filter(r => {
    const modelAdapter = String((r.model && r.model.adapter_name) || (r.request && r.request.adapter_name) || "").toLowerCase();
    const blob = JSON.stringify(r).toLowerCase();
    return (!phase || String(r.phase || "").toLowerCase() === phase)
      && (!adapter || modelAdapter === adapter)
      && (!step || String((r.context && r.context.step) || "") === step)
      && (!search || blob.includes(search));
  });
}

function renderList() {
  const root = $("records");
  const rows = filteredRecords();
  root.innerHTML = "";
  rows.forEach(r => {
    const div = document.createElement("div");
    div.className = "record" + (selected && selected.trace_id === r.trace_id ? " active" : "");
    div.innerHTML = `<strong>${escapeHtml(r.phase || "")}</strong><br>
      <span class="muted">${escapeHtml(r.boundary || "")}</span><br>
      <span class="muted">step=${escapeHtml(String((r.context && r.context.step) || ""))}
      adapter=${escapeHtml(String((r.model && r.model.adapter_name) || (r.request && r.request.adapter_name) || ""))}</span>`;
    div.onclick = () => { selected = r; renderList(); renderDetail(); };
    root.appendChild(div);
  });
}

function renderDetail() {
  const root = $("detail");
  if (!selected) { root.innerHTML = '<p class="muted">No record selected.</p>'; return; }
  const r = selected;
  let html = `<h2>${escapeHtml(r.phase || "")}</h2>
    <p class="muted">${escapeHtml(r.trace_id)} | ${escapeHtml(r.timestamp || "")}</p>
    <pre>${escapeHtml(JSON.stringify({context:r.context, exactness:r.exactness, error:r.error}, null, 2))}</pre>`;
  if (r.phase === "generation") {
    html += `<h3>Prompt</h3>${tokenTable((r.request && r.request.prompt_tokens) || [])}`;
    html += `<h3>Completion</h3>${tokenTable((r.response && r.response.completion_tokens) || [])}`;
  } else if ((r.phase || "").startsWith("trainer")) {
    html += trainerTables(r);
  } else {
    html += `<h3>Request</h3><pre>${escapeHtml(JSON.stringify(r.request, null, 2))}</pre>
      <h3>Response</h3><pre>${escapeHtml(JSON.stringify(r.response, null, 2))}</pre>`;
  }
  root.innerHTML = html;
}

function tokenTable(tokens) {
  if (!tokens.length) return '<p class="muted">No tokens.</p>';
  return `<table><thead><tr><th>#</th><th>ID</th><th>Text</th><th>Logprob</th><th>Top-k alternatives</th></tr></thead><tbody>`
    + tokens.map(t => `<tr class="${lpClass(t.logprob)}"><td>${t.index}</td><td>${t.token_id}</td><td><code>${escapeHtml(String(t.text ?? ""))}</code></td><td>${fmt(t.logprob)}</td><td>${alts(t.top_logprobs)}</td></tr>`).join("")
    + `</tbody></table>`;
}

function trainerTables(r) {
  const rows = (r.request && r.request.rows) || [];
  return rows.map(row => `<h3>Example ${row.example_index}</h3>
    <table><thead><tr><th>#</th><th>input</th><th>target</th><th>attention</th><th>loss</th><th>behavior lp</th><th>old lp</th><th>adv</th><th>target lp</th><th>top-k</th></tr></thead><tbody>`
    + row.input_ids.map((id, i) => `<tr><td>${i}</td><td>${id} <code>${escapeHtml(tokenText(row.input_tokens, i))}</code></td><td>${row.target_ids[i]} <code>${escapeHtml(tokenText(row.target_tokens, i))}</code></td><td>${row.attention_mask[i]}</td><td>${row.loss_mask[i]}</td><td>${(row.behavior_logprob_mask || [])[i]}</td><td>${fmt(row.old_logprobs[i])}</td><td>${fmt(row.advantages[i])}</td><td>${fmt((row.target_logprobs || [])[i])}</td><td>${alts((row.target_top_logprobs || [])[i])}</td></tr>`).join("")
    + `</tbody></table>`).join("");
}

function tokenText(rows, index) { const row = (rows || [])[index]; return row ? String(row.text ?? "") : ""; }
function alts(items) { return (items || []).map(a => `${a.rank}:${a.token_id} ${escapeHtml(String(a.text ?? ""))} (${fmt(a.logprob)})`).join("<br>"); }
function fmt(value) { return value === undefined || value === null ? "" : Number(value).toFixed(4); }
function lpClass(value) { if (value === undefined || value === null) return ""; if (value > -1) return "lp-good"; if (value > -5) return "lp-mid"; return "lp-low"; }
function escapeHtml(s) { return String(s).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c])); }
</script>
</body>
</html>
"""
