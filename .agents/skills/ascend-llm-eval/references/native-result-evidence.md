# Native AISBench request evidence

native_results.py preserves the request-group validation semantics from baseline
aisbench_cli.validate_native_result at commit
74a7feffb76c32b23658d37ea681784b3a9b2474. The frozen helper source is unchanged;
the v2 evaluator now calls the new streaming validator.

Validation requires returncode=0, exactly the resolved model/dataset groups,
unique request IDs, success=true, no error_info, exact reviewed formal_requests,
and native worker logs. A validated.json artifact is written atomically only after
all groups pass. It lists operation-relative evidence paths.

Each JSONL row is read with a 1 MiB cap. Oversize rows fail explicitly; they are
not silently truncated. At most 128 result groups are accepted. Duplicate tracking
is bounded by the reviewed request count, and parsing stops if it is exceeded.
Duplicate JSON keys and non-finite numbers fail validation. Phase deadline and
cancellation are checked while reading. Symlink/out-of-operation detail paths are
rejected. These limits are execution safeguards, not changes to requested load.

Passing native requests does not establish a valid steady-state window or server
metric agreement. The result keeps performance_validity=UNVERIFIED until those
separate evidence checks are implemented and satisfied. Local tests use generated
native-format evidence, not a real AISBench run.
