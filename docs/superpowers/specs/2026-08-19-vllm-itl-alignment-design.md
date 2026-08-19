# Align aisbench ITL Timing with vLLM Completions

## Goal

Make aisbench's VLLM streaming timing semantics match `vllm bench serve --backend vllm`: only nonempty `choices` responses count as output timing events, while usage-only and `[DONE]` messages do not affect ITL, TPOT, or E2EL.

## Scope

- Apply token-event filtering to `VLLMCustomAPI` and `VLLMCustomAPIChat`.
- Preserve existing streaming timing behavior for TGI, Triton, MindIE, and VITA adapters.
- Continue parsing usage-only responses so prompt and completion token counts remain available.
- Remove the current uncommitted `[ITL-DIAG]` instrumentation from `base_api.py`.
- Add regression tests before changing production behavior.
- Do not change ITL aggregation, percentiles, CSV columns, or the meaning of the reported `N` field in this change.

## Design

### Protocol-aware timing predicate

Add an overridable method to `BaseAPIModel`:

```python
def should_record_stream_time_point(self, data: dict) -> bool:
    return True
```

The base implementation preserves all current non-vLLM adapters. Both vLLM adapters override it with:

```python
def should_record_stream_time_point(self, data: dict) -> bool:
    return bool(data.get("choices"))
```

### Stream data flow

For each nonempty, non-comment, non-`[DONE]` stream message:

1. Decode and parse the JSON payload.
2. Call `should_record_stream_time_point(data)`.
3. Record a time point only when the predicate returns `True`.
4. Always call `parse_stream_response(data, output)` so usage-only responses still populate token counts.

The request-start timestamp remains unchanged. For 1024 choice-bearing output chunks, the final timing array contains the start plus 1024 output events, producing 1023 ITLs.

## Compatibility

- This matches the vLLM v0.26 Completions client, which timestamps inside its nonempty-`choices` branch.
- The chat adapter uses the same filtering so the currently configured aisbench Chat endpoint can be compared using the same timing semantics.
- Other adapters inherit the default predicate and retain unconditional JSON-chunk timing.
- JSON parse failures retain the existing error path and do not create timing points.

## Testing

Add focused async tests that feed a stream containing two choice-bearing chunks, one usage-only chunk, and `[DONE]`.

Required assertions:

- The VLLM adapter records three time points: request start plus two choice events.
- The usage-only chunk does not add a timing point.
- Completion and prompt token counts are populated from usage.
- The derived ITL count is one.
- A representative non-vLLM adapter continues recording each parsed data chunk.

The tests must fail against the current implementation before production code changes are made, then pass after the minimal implementation.

## Error Handling

No new error category is introduced. Existing HTTP, JSON parsing, retry, and empty-response behavior remains unchanged.

## Acceptance Criteria

- No sub-millisecond usage-tail ITL is produced by a VLLM usage-only response.
- VLLM usage token accounting remains correct.
- Non-vLLM streaming timing behavior does not change.
- Targeted tests and the relevant existing API-model test suite pass.
- `base_api.py` parses successfully after the temporary diagnostics are removed.
