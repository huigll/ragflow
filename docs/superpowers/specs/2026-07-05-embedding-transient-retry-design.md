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
- The task queue has a coarse retry counter, but a full task retry may repeat
  expensive parse/OCR work.
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
- Change the task scheduler or task retry count semantics globally.

## Recommended Approach

Use a two-layer design:

1. Add short, embedding-only retries at the `LLMBundle.encode()` and
   `LLMBundle.encode_queries()` boundary.
2. Let task execution treat retry-exhausted transient embedding errors as
   recoverable task failures, so the existing task retry mechanism can schedule
   another attempt.

This combines immediate recovery for small API blips with queue-level recovery
for longer outages, while failing fast for permanent errors.

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
errors. It should add one distinction:

- If the exception is `ModelException` with `retryable=True`, write a progress
  message that says the embedding API error is temporary and the task will be
  retried automatically, then re-raise.
- Otherwise keep the current permanent failure behavior.

This applies to:

- binding probe: `embedding_model.encode(["ok"])`
- standard chunk embedding
- refactored task handler equivalents

Dataflow embedding paths naturally benefit from the `LLMBundle.encode()` retry
wrapper even if their local error handling remains unchanged.

## Retry Configuration

Use environment variables first, with conservative defaults:

- `EMBEDDING_MAX_RETRIES=3`
- `EMBEDDING_RETRY_BASE_SECONDS=1.0`
- `EMBEDDING_RETRY_MAX_SECONDS=20.0`

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

An embedding error is transient when any of these apply:

- The caught exception is `ModelException` and `retryable is True`.
- The exception is a requests/openai SDK connection or timeout error.
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
- The message includes HTTP status `408`, `429`, `500`, `502`, `503`, or `504`.

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
  - `quota`
  - `balance`
  - `billing`

Permanent rules take precedence over text-only transient rules when both match,
except for an explicit `ModelException(retryable=True)`.

Empty or whitespace-only embedding input is not an error because
`LLMBundle.encode()` already coerces it to `"None"`.

## Progress and Logging

Short retries should log warnings but avoid appending noisy progress updates on
every attempt.

Suggested warning log:

```text
LLMBundle.encode temporary embedding API error for model <name>; retrying in <seconds>s (attempt <n>/<max>): <error>
```

When short retries are exhausted, task progress should include one concise
message:

```text
Temporary embedding API error. The task will retry automatically: <error>
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
   the task layer and the task is retried by the existing queue mechanism.
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
- Retry sleeps happen inside worker threads for synchronous embedding calls, so
  the default retry budget must stay small.

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

- Retryable binding failure writes the temporary progress message and re-raises.
- Permanent binding failure keeps the existing failure message.

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

1. Ship with conservative defaults.
2. Monitor task progress logs for reduced one-off embedding binding failures.
3. If workers spend too much time sleeping during provider outages, lower
   `EMBEDDING_MAX_RETRIES` or `EMBEDDING_RETRY_MAX_SECONDS`.
4. If short API blips still reach task-level retry too often, increase
   `EMBEDDING_MAX_RETRIES` gradually.

## Acceptance Criteria

- A temporary DashScope connection failure during `encode(["ok"])` is retried
  locally before the task is marked failed.
- If the temporary failure persists, the task progress says it will retry
  automatically and the existing task retry mechanism gets control.
- Invalid API keys and malformed requests still fail without retry loops.
- The same retry behavior applies to binding, chunk embedding, Dataflow
  embedding, and query embedding.
- Existing successful embedding flows keep the same return values and token
  accounting.
