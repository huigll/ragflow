# Embedding Transient Retry Design

Date: 2026-07-05

## Problem

Document parsing tasks currently bind the embedding model by constructing an
`LLMBundle` and calling `encode(["ok"])` before chunk parsing continues. If the
embedding provider has a temporary network failure, the page task fails
immediately:

```text
Fail to bind embedding model: HTTPSConnectionPool(host='dashscope.aliyuncs.com', port=443): Max retries exceeded ...
Network is unreachable
```

The same document may later process successfully for another page range, which
shows that this class of failure is often transient rather than a permanent
model or configuration error.

The current retry coverage is uneven:

- `QWenEmbed.encode()` retries non-200 DashScope responses, but not SDK or
  requests exceptions raised before a response is available.
- The task queue's `retry_count` cap (3, in `TaskService.get_task`) only
  applies to *redelivered* messages, and redelivery only happens when a worker
  dies before acking (`get_unacked_iterator` at worker startup). A task that
  fails with an exception is acked and never retried by the queue.
- Other embedding paths such as Dataflow and retrieval also call
  `LLMBundle.encode()`, so provider-specific fixes would leave inconsistent
  behavior.

## Goals

- Apply the same transient retry policy to embedding model binding,
  chunk embedding, Dataflow embedding, and query embedding.
- Treat short network/API instability as recoverable without user action.
- Avoid retrying permanent configuration or request errors.
- Keep chat, vision, TTS, OCR, and rerank behavior unchanged.
- Preserve the existing task retry cap as the final guardrail.
- Make progress logs distinguish temporary embedding API failures from
  permanent failures.

## Non-Goals

- Add new frontend settings.
- Add new database fields or migrations.
- Rewrite provider-specific embedding implementations.
- Add fallback to another embedding model, because vector dimensions and
  dataset consistency would become unsafe.
- Change the task retry count semantics globally. (The targeted requeue in
  `handle_task` for retryable embedding errors reuses the existing
  per-delivery `retry_count` cap; it does not alter scheduling for any other
  error class.)

## Recommended Approach

Use a two-layer design:

1. Add short, embedding-only retries at the `LLMBundle.encode()` and
   `LLMBundle.encode_queries()` boundary.
2. On retry-exhausted transient embedding errors, have `handle_task` explicitly
   re-enqueue the task message instead of marking the task failed.

This combines immediate recovery for small API blips with queue-level recovery
for longer outages, while failing fast for permanent errors.

**Why layer 2 must be an explicit requeue.** There is no existing mechanism
that retries a cleanly-failed task. `handle_task`
(`rag/svr/task_executor.py`) catches every exception, writes `progress=-1`,
and then unconditionally calls `redis_msg.ack()` — an acked message is never
redelivered. The `retry_count >= 3` cap in `TaskService.get_task` increments
per *delivery*, and redelivery only happens through
`RedisDB.get_unacked_iterator`, which reclaims un-acked messages once at
worker startup. In other words, today's "task retry" only covers worker
crashes, not task failures. Simply re-raising a retryable error would mark
the task permanently failed and the "will retry automatically" progress
message would be false.

## Architecture

### Embedding Boundary

File: `api/db/services/llm_service.py`

Add a private helper on `LLMBundle`, for example:

```python
def _run_embedding_with_retry(self, operation_name: str, fn: Callable[[], tuple]):
    ...
```

`encode()` and `encode_queries()` keep their existing input normalization,
Langfuse observation, token logging, and return values. Only the call into the
underlying embedding model is wrapped:

- `self.mdl.encode(safe_texts)`
- `self.mdl.encode_queries(query)`

The helper owns:

- attempt count
- exponential backoff
- jitter
- transient/permanent error classification
- final conversion to `ModelException(retryable=True/False)`

This keeps the behavior provider-independent and avoids changing non-embedding
LLM operations.

### Task Boundary

Files:

- `rag/svr/task_executor.py`
- `rag/svr/task_executor_refactor/task_handler.py`

The task layer should continue to catch embedding binding and chunk embedding
errors. `handle_task` (the top-level catch in `rag/svr/task_executor.py`)
adds one distinction:

- If the exception is `ModelException` with `retryable=True` **and**
  `task["retry_count"] < 3` (the value fetched by `TaskService.get_task`,
  which increments it per delivery): re-produce the original message payload
  to its source queue via `REDIS_CONN`, ack the old message, and write a
  non-terminal progress message (progress stays `>= 0`) saying the embedding
  API error is temporary and the task has been requeued. The `retry_count`
  cap then genuinely abandons the task after 3 deliveries — and since PR
  #16381, the abandon path in `TaskService.get_task` also marks the Document
  `run=FAIL`, so the final failure state needs no extra handling here.
- Otherwise keep the current permanent failure behavior
  (`progress=-1`, ack).

Inner layers (`do_handle_task` binding probe, chunk embedding,
`task_handler.py` equivalents) must let the retryable `ModelException`
propagate to `handle_task` instead of masking it — today
`_bind_embedding_model` and the `do_handle_task` binding probe write
`progress_callback(-1, ...)` before re-raising, which would mark the task
failed even when it is about to be requeued. On `retryable=True` they must
skip the `progress=-1` write and only log. Since PR #16381 (commit
`985e3c1db`) this is mandatory, not cosmetic: `TaskService.update_progress`
now propagates any `progress=-1` immediately to the parent Document
(`run=FAIL`), and the task-progress update predicate
(`cls.model.progress != -1`) means a task that once reported `-1` can never
report intermediate progress again — a stray `-1` write would both fail the
document and silence progress for all subsequent requeued deliveries.

Requeue-at-binding is cheap because the probe runs before parse/OCR. A
chunk-embedding failure after parsing does repeat parse work on requeue —
acceptable, since it is bounded by the 3-delivery cap and beats failing the
task outright.

Dataflow embedding paths naturally benefit from the `LLMBundle.encode()` retry
wrapper even if their local error handling remains unchanged.

## Retry Configuration

Use environment variables first, with conservative defaults:

- `EMBEDDING_MAX_RETRIES=3`
- `EMBEDDING_RETRY_BASE_SECONDS=1.0`
- `EMBEDDING_RETRY_MAX_SECONDS=20.0`
- `EMBEDDING_QUERY_MAX_RETRIES=1` — smaller budget for `encode_queries()`,
  which sits on the synchronous user-facing retrieval path

Backoff:

```text
sleep = min(max_seconds, base_seconds * 2 ** attempt) * random.uniform(0.8, 1.3)
```

With the defaults, a single call waits roughly 1s, 2s, and 4s before giving up.
This is long enough for small provider hiccups and short enough to avoid tying
up a worker during longer outages.

`EMBEDDING_MAX_RETRIES=0` disables the short retry layer while preserving
classification.

## Error Classification

### Transient

Classification checks typed exceptions first; message-text rules are a
fallback for providers that raise generic exceptions.

An embedding error is transient when any of these apply, in order:

- The caught exception is `ModelException` and `retryable is True`.
- The exception is a typed connection/timeout/rate-limit error:
  `requests.exceptions.ConnectionError` / `Timeout`,
  `openai.APIConnectionError` / `APITimeoutError` / `RateLimitError`
  (except `insufficient_quota`, see below), or the `httpx` equivalents.
- The message contains one of:
  - `timeout`
  - `timed out`
  - `connection`
  - `connect`
  - `network`
  - `unreachable`
  - `dns`
  - `reset`
  - `temporarily unavailable`
  - `throttl` (DashScope rate limiting, e.g. `Throttling.RateQuota`)
- The message includes HTTP status `408`, `429`, `500`, `502`, `503`, or
  `504`. Match status codes structurally, not as bare substrings — extract
  from `status: <code>` / `status_code=<code>` patterns (the formats used by
  `rag/llm/embedding_model.py`), never `"500" in msg`, which false-matches
  token counts and model names.

### Permanent

An embedding error is permanent when any of these apply:

- The caught exception is `ModelException` and `retryable is False`.
- The message includes HTTP status `400`, `401`, `403`, or `404`.
- The message contains one of:
  - `invalid`
  - `bad request`
  - `api key`
  - `auth`
  - `permission`
  - `forbidden`
  - `model not found`
  - `insufficient quota` / `insufficient_quota`
  - `balance`
  - `billing`

Note: bare `quota` must NOT be a permanent keyword. DashScope rate-limit
errors carry codes like `Throttling.RateQuota` with HTTP 429 and are
transient; only quota-exhausted/billing wording (OpenAI `insufficient_quota`)
is permanent.

Permanent rules take precedence over text-only transient rules when both match,
except for an explicit `ModelException(retryable=True)` and a structural
`429` status, which stay transient.

Empty or whitespace-only embedding input is not an error because
`LLMBundle.encode()` already coerces it to `"None"`.

## Progress and Logging

Short retries should log warnings but avoid appending noisy progress updates on
every attempt.

Suggested warning log:

```text
LLMBundle.encode temporary embedding API error for model <name>; retrying in <seconds>s (attempt <n>/<max>): <error>
```

When short retries are exhausted and the task is requeued, task progress
should include one concise message:

```text
Temporary embedding API error. The task has been requeued (attempt <n>/3): <error>
```

Permanent failures should keep the existing user-facing message, optionally
including `non-retryable` for easier diagnosis.

## Data Flow

### Binding Probe

1. Task loads the tenant embedding model config.
2. Task constructs `LLMBundle`.
3. Task calls `encode(["ok"])`.
4. `LLMBundle` normalizes input and calls `_run_embedding_with_retry()`.
5. A transient provider error is retried locally.
6. If local retries succeed, task continues with `init_kb()`.
7. If local retries are exhausted, `ModelException(retryable=True)` bubbles to
   `handle_task`, which re-produces the message to its queue, acks the old
   one, and leaves progress non-terminal. `TaskService.get_task` abandons the
   task after 3 deliveries.
8. If the error is permanent, task fails immediately.

### Chunk Embedding

1. Chunk builder produces chunks.
2. `EmbeddingService.embed_chunks()` batches chunk text.
3. Each batch calls `LLMBundle.encode()` through the same retry wrapper.
4. Successful retries are transparent to indexing.
5. Retry-exhausted transient failures bubble up as recoverable task failures.

### Query Embedding

1. Retrieval code calls `LLMBundle.encode_queries()`.
2. The same retry wrapper handles temporary provider instability.
3. Since retrieval requests are synchronous user-facing operations, exhausted
   transient errors should surface as the current API error response, but with a
   clearer `retryable=True` classification for callers that inspect it.

## Edge Cases

- `TaskCanceledException` must not be retried.
- Keyboard interrupts, process termination, and asyncio cancellations must not
  be swallowed by the embedding retry helper.
- Provider-specific retry loops can remain in place. The outer helper is a
  safety net for failures they do not classify.
- The wrapper must not double-count tokens; token usage is logged and reported
  only after a successful attempt.
- Langfuse observations should end with error output when the operation finally
  fails.
- **Event-loop blocking**: chunk embedding already runs `encode()` through
  `thread_pool_exec`, so retry sleeps there land in worker threads. But the
  binding probe `encode(["ok"])` is called directly inside async coroutines
  (`do_handle_task` in `rag/svr/task_executor.py` and
  `_bind_embedding_model` in `task_handler.py`) — a blocking backoff sleep
  there stalls every concurrent task in the worker. Both probe call sites
  must be changed to `await thread_pool_exec(embedding_model.encode, ["ok"])`.
- **Timeout interaction**: the `batch_encode` wrappers are decorated with
  `@timeout(60)`, which now covers all local retries plus per-attempt network
  time. The default retry budget (~7s of sleep + up to 4 attempts) must fit
  well inside 60s; document that raising `EMBEDDING_MAX_RETRIES` or
  `EMBEDDING_RETRY_MAX_SECONDS` can silently hit this outer timeout instead.
- Query embedding (`encode_queries`) serves synchronous user-facing requests;
  a full 3-retry backoff adds ~7s of latency to a search. Use a smaller
  budget there (e.g. `EMBEDDING_QUERY_MAX_RETRIES=1`, default derived from
  but capped below the ingestion setting).

## Testing Plan

### Unit Tests

Add tests around `LLMBundle.encode()` and `encode_queries()` using a fake
underlying model:

- Succeeds after two transient connection failures.
- Raises `ModelException(retryable=True)` after retry exhaustion.
- Does not retry `ModelException(retryable=False)`.
- Does not retry permanent HTTP 401/403/400-style messages.
- Honors `EMBEDDING_MAX_RETRIES=0`.
- Does not retry `TaskCanceledException`.
- Preserves token counts and embeddings from the successful attempt.

Add task-layer tests for both executor paths:

- Retryable binding failure propagates without writing `progress=-1`.
- `handle_task` on `ModelException(retryable=True)`: re-produces the message
  to the source queue, acks the old message, writes the requeue progress
  message with non-terminal progress.
- `handle_task` on the same error at `retry_count >= 3`: falls through to the
  permanent failure path (no infinite requeue loop).
- Permanent binding failure keeps the existing failure message.
- Binding probe call sites use `thread_pool_exec` (no direct `encode` call on
  the event loop).

### Existing Test Commands

Focused commands:

```bash
uv run pytest test/unit_test/rag/llm/test_embedding_model.py
uv run pytest test/unit_test/rag/svr/task_executor_refactor/test_task_handler.py
uv run pytest test/unit_test/rag/svr/task_executor_refactor/test_embedding_service.py
```

Broader validation if time permits:

```bash
uv run pytest test/unit_test/rag/llm test/unit_test/rag/svr/task_executor_refactor
```

## Rollout

Ship in two phases; phase 1 alone resolves the reported failure mode.

**Phase 1 — layer 1 only (`LLMBundle` short retries).** Low risk, no change
to task/ack semantics, and it covers the actual reported incident (a
seconds-long network blip during the binding probe). Upstream PR #16381 now
fails documents cleanly and consistently when retries are exhausted, so the
cost of a residual failure is a manual re-run, not a stuck document.

**Phase 2 (optional) — layer 2 requeue in `handle_task`.** Only worth doing
if phase 1 monitoring shows outages that outlast the local retry budget
(~7s) but recover within minutes still fail tasks often enough to matter.
This is the invasive part of the design (ack semantics); defer until the
data justifies it.

1. Ship with conservative defaults.
2. Monitor task progress logs for reduced one-off embedding binding failures.
3. If workers spend too much time sleeping during provider outages, lower
   `EMBEDDING_MAX_RETRIES` or `EMBEDDING_RETRY_MAX_SECONDS`.
4. If short API blips still reach task-level retry too often, increase
   `EMBEDDING_MAX_RETRIES` gradually.

## Acceptance Criteria

- A temporary DashScope connection failure during `encode(["ok"])` is retried
  locally before the task is marked failed.
- If the temporary failure persists, the task is requeued (not marked failed),
  the progress message says it was requeued, and the task is abandoned only
  after 3 deliveries.
- Invalid API keys and malformed requests still fail without retry loops.
- The same retry behavior applies to binding, chunk embedding, Dataflow
  embedding, and query embedding.
- Existing successful embedding flows keep the same return values and token
  accounting.
