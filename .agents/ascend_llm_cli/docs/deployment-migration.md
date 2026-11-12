# Deployment contract migration

The deployment item Schema is extracted from the immutable baseline commit
74a7feffb76c32b23658d37ea681784b3a9b2474. Exact provenance is recorded in
deployment-schema-source.json. No mutable stable module is imported. The test
fixture is the deployment item of that commit's job_spec.compact.json; it is
illustrative data, not authorization to access any server.

| Historical input | v2 deploy plan |
| --- | --- |
| job_id | task_id (v2 identity syntax) |
| deployments[i] | deployment (one reviewed deployment per stage plan) |
| deployments[i].instances[].launch | unchanged, including bootstrap/env/args/full KV JSON |
| execution | execution, with timing/gating/retry choices explicit before review |
| unresolved | unresolved |
| cleanup | explicit policy; no automatic equivalence mapping |
| cases/sub_tasks/stages | separate eval/routing inputs; not silently discarded by a converter |

This is a contract, not an automatic old-job converter. Multiple deployments must
be planned as separate stage operations; existing groups/stages have not yet been
migrated. There is no claim that old FULL lifecycle behavior has been verified.

Current validation covers Schema, unique instance identities, endpoint references,
KV/routing roles, proxy/model launch distinction and basic topology completeness.
It preserves extensible launch.args and complete KV payloads without mutation.
Bound compilation now validates binding Schema/IPv4, references, port spans,
environment-vs-network conflicts and device overlap through the frozen topology
helper. It expands full KV JSON, argv/env/bootstrap and mount profiles without
modifying inputs. Image-specific KV compatibility and container lifecycle scripts
remain incomplete. Provenance is in topology-migration-source.json. Input freezing
checks this plan but still reports production_ready=false; it is not deployment
approval. Existing review digest checks remain necessary.
