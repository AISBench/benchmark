"""离线替代 review_job 的现场探测：只验证「证据与绑定不一致即 BLOCKED」这一判定契约。

不访问网络、不 SSH：把 PROBE 的观测结果构造出来，直接调用 review_job.assess()。
"""
import json
import sys
from pathlib import Path

skill_dir, job_spec, bindings = sys.argv[1:4]
sys.path.insert(0, str(Path(skill_dir) / "scripts"))

from task_model import load_json  # noqa: E402
from review_job import assess  # noqa: E402


def evidence(ips):
    return {
        slot: {
            "models": {"dsv4": {"config": {"model_type": "demo", "quantization": None}}},
            "errors": [],
            "paths": {},
            "images": {},
            "network": [{"ifname": "eth0", "addr_info": [{"local": ip}]}],
        }
        for slot, ip in ips.items()
    }


spec = load_json(job_spec)
bindings_doc = load_json(bindings)

healthy = assess(spec, bindings_doc, evidence({"node_p": "10.0.0.21", "node_q": "10.0.0.22"}))
mismatch = assess(spec, bindings_doc, evidence({"node_p": "10.0.0.21", "node_q": "10.0.0.99"}))

print(json.dumps({
    "healthy": {"status": healthy["status"], "errors": healthy["errors"]},
    "mismatch": {"status": mismatch["status"], "errors": mismatch["errors"]},
}, ensure_ascii=False))
