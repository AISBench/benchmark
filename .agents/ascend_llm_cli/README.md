# Ascend LLM CLI

Execution package for the five sibling skills in ../skills/: ascend-llm-router, ascend-llm-plan, ascend-llm-deployment, ascend-llm-eval and ascend-llm-release.

- src/deploy_eval_v2/: installable Python implementation and schemas.
- tests/: contract, process and lifecycle tests.
- docs/: developer-facing contracts and design evidence.
- dist/: versioned wheel artifacts retained for reproducibility.

The source directory was renamed from runtime_v2. The package name deploy-eval-v2, command deploy-eval-v2 and module deploy_eval_v2 remain stable; this is a layout/skill-name migration, not a command migration. Python >=3.11 and jsonschema are required. Run tests from this directory with PYTHONPATH=src and python -m unittest discover -s tests -v.

Runtime and remote tester launch detached Python processes. Logs remain under the user work root. No systemd dependency. Linux socket acceptance of the direct Python launcher completed in a development environment; production model evaluation and fault-injection acceptance remain incomplete. Site-specific inputs and validation transcripts are retained locally and are not part of this source distribution.

Skill references are distributed by stage. CLI protocol references and cross-stage socket acceptance belong to router; plan owns plan/bindings/review schemas; deployment owns publication/start and host checks; eval owns benchmark/result evidence; release owns resource cleanup. Do not copy execution code into individual skills.
