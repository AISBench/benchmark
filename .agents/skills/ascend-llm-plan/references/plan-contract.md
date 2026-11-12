# Plan and bindings contract

The accompanying schemas define deploy-plan, deploy-bindings, external-eval-plan and eval-bindings. Review and freezing perform schema and semantic validation, compile topology and scripts, and bind all input and runtime digests. These local checks do not prove host readiness.

Owned deploy plans may contain evaluation {cases, test_binding}; deploy bindings then include testers. Each tester binds host, machine_id, work_root, python, aisbench, model_path and datasets. Native synthetic datasets use type=aisbench_synthetic. Explicit load values are preserved; defaults must be resolved before review.

Optional tester.environment is a map of explicit environment variable strings applied to preflight, AISBench and postflight subprocesses. No shell expansion occurs. Freeze actual absolute paths, not $VARIABLE expressions. Discover required CANN environment from the selected installed toolkit using read-only checks; validate imports under those values before deployment. Do not install or upgrade a user's environment to resolve missing libraries. Record chosen toolkit, environment values and import evidence. Never infer that an executable's presence means its imports work. Environment values must not contain secrets, since frozen inputs and evidence retain them.

Production preflight checks interpreter and console entrypoint ownership, tokenizer assets, config imports and dataset capability. Explicit authorized cache reset requires frozen per-instance targets; never infer these from the service URL. Execution and evidence/performance validity remain separate outcomes.
