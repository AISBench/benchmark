#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Unified agent dataset builder with pluggable source dataset types.

Usage:
    python build_agent_dataset.py --src-dataset-type Jiuwen --src <dir> [--output <file>] [--tokenizer <path>]
    python build_agent_dataset.py --src-dataset-type default --src <file> [--output <file>] \
        [--num-low 1] [--num-high 10]

Each dataset type has its own processor registered in DATASET_PROCESSORS.
To add a new type, implement a processor function and register it via @register("name").

  ┌─────────────┬────────────────────┬────────────────────────────────────┐
  │    简写     │        全称        │                说明                │
  ├─────────────┼────────────────────┼────────────────────────────────────┤
  │ -t          │ --src-dataset-type │ 数据集类型                         │
  ├─────────────┼────────────────────┼────────────────────────────────────┤
  │ -s          │ --src              │ 源路径                             │
  ├─────────────┼────────────────────┼────────────────────────────────────┤
  │ -o          │ --output           │ 输出文件                           │
  ├─────────────┼────────────────────┼────────────────────────────────────┤
  │ -c          │ --cut-count        │ session 内尾部截取条数             │
  ├─────────────┼────────────────────┼────────────────────────────────────┤
  │ -n          │ --session-count    │ 保留前 N 个 session                │
  ├─────────────┼────────────────────┼────────────────────────────────────┤
  │ -p          │ --add-pause        │ 暂停比例                           │
  ├─────────────┼────────────────────┼────────────────────────────────────┤
  │ -P          │ --preview          │ 预览模式（大写以区分 -p）          │
  ├─────────────┼────────────────────┼────────────────────────────────────┤
  │ --tokenizer │ 无简写             │ 保留全称（使用频率低，避免占字母） │
  ├─────────────┼────────────────────┼────────────────────────────────────┤
  │ --swe-format│ 无简写             │ SWE-bench 风格输出                 │
  └─────────────┴────────────────────┴────────────────────────────────────┘

使用实例：
# Jiuwen数据集（普通格式）
python ./build_agent_dataset.py --src-dataset-type Jiuwen --src ./c20_on_model_io/model_io
# Jiuwen数据集（SWE-bench 风格输出，透传 messages/tools 字段）
python ./build_agent_dataset.py --src-dataset-type Jiuwen --src ./c20_on_model_io/model_io --swe-format
# 截取尾部 cut-count 条数据
python ./build_agent_dataset.py --src-dataset-type Jiuwen --src ./c20_on_model_io/model_io --cut-count 5
# 截取前 --session-count 个 session
python ./build_agent_dataset.py --src-dataset-type Jiuwen --src ./c20_on_model_io/model_io --cut-count 5 --session-count 3
# 指定添加随机暂停 --add-pause 0.5
python ./build_agent_dataset.py --src-dataset-type Jiuwen --src ./c20_on_model_io/model_io --cut-count 4 --session-count 3 --add-pause 0.5
# 预览 --preview，把 question 字段置为 "preview text" 并打印数据集结构
python ./build_agent_dataset.py --src-dataset-type Jiuwen --src ./c20_on_model_io/model_io --cut-count 4 --session-count 3 --add-pause 0.5 --preview
# default 类型
python build_agent_dataset.py --src agent_dataset_pure.jsonl --output agent_dataset_pure2.jsonl --tokenizer D:/data/qwen3-06b/tokenizer.json --add-pause 0.5
"""

import argparse
import copy
import json
import os
import random
import re
import string
import sys
from pathlib import Path
from typing import Callable

# ---------------------------------------------------------------------------
# Dataset processors registry
# ---------------------------------------------------------------------------
# Each processor signature: (args: argparse.Namespace) -> None
# It is responsible for reading its own source and writing the output file.
DATASET_PROCESSORS: dict[str, Callable[[argparse.Namespace], None]] = {}


def register(name: str):
    def deco(fn):
        DATASET_PROCESSORS[name] = fn
        return fn
    return deco


# Resolve a path that may be relative. Relative paths are resolved against
# the script's own directory, so `--src ./c20_on_model_io` works from any cwd.
SCRIPT_DIR = Path(__file__).resolve().parent


def resolve_path(p: str) -> Path:
    path = Path(p)
    if path.is_absolute():
        return path
    return (SCRIPT_DIR / path).resolve()


# ---------------------------------------------------------------------------
# Output-format helpers
# ---------------------------------------------------------------------------
# When --swe-format is set, every record (data-plane AND manage-request)
# carries the SWE-bench-style fields ``messages``/``tools``/``tool_choice``.
# Manage requests use placeholder values (empty system message + dummy
# read_pdf tool) so the record shape is uniform across the dataset.
# When --swe-format is NOT set, records only carry ``question``/``answer``
# and manage requests leave those fields empty.

_SWE_PLACEHOLDER_MESSAGES = [{"content": "", "role": "system"}]
_SWE_PLACEHOLDER_TOOLS = [{
    "function": {
        "description": "",
        "name": "read_pdf",
        "parameters": {
            "additionalProperties": True,
            "properties": {
                "call_goal": {
                    "description": "call goal",
                    "type": "string"
                },
                "inputs": {
                    "additionalProperties": {},
                    "description": "inputs",
                    "type": "object"
                }
            },
            "required": ["inputs"],
            "title": "read pdf",
            "type": "object"
        }
    },
    "type": "function"
}]


def _base_record(swe_format: bool) -> dict:
    """Return a dict with the fields every record carries, in the requested
    output format. Data-plane records will fill ``question`` (non-swe) or
    ``messages``/``tools`` (swe); manage requests leave them as placeholders."""
    rec: dict = {"question": "", "answer": ""}
    if swe_format:
        rec["messages"] = list(_SWE_PLACEHOLDER_MESSAGES)
        rec["tools"] = [dict(t) for t in _SWE_PLACEHOLDER_TOOLS]
        rec["tool_choice"] = "auto"
    return rec


def _make_manage_request(
    session_id: str,
    parent_session_id: str,
    edit_type: str,
    stage: str = "",
    manage_request: bool = True,
    swe_format: bool = False,
) -> dict:
    """Build a control-plane record (offload/prefetch/evict/end).

    Args:
        edit_type: one of "offload"/"prefetch"/"evict"/"" (empty edits for end).
        stage: stage marker; "" for inline manage requests, "end" for end records.
        manage_request: True for control-plane; False for the trailing "end"
            data-plane marker.
        swe_format: when True, include placeholder messages/tools/tool_choice.

    Returns:
        A new record dict.
    """
    edits = [{"type": edit_type, "target": "session"}] if edit_type else []
    rec = _base_record(swe_format)
    rec["agent_hint"] = {
        "session_id": session_id,
        "parent_session_id": parent_session_id,
        "cache_control": {},
        "context_management": {
            "manage_request": manage_request,
            "edits": edits,
        },
        "stage": stage,
    }
    return rec


# ---------------------------------------------------------------------------
# Shared helpers used by both Jiuwen and default processors.
# ---------------------------------------------------------------------------
def add_pause_in_session(dataset, percent, swe_format=False):
    """Insert an offload+prefetch pair at a random position inside
    ``percent`` fraction of sessions.

    - Sessions are grouped by ``session_id`` (first-seen order).
    - Target session count = round(total_sessions * percent), clamped to
      [0, total_sessions]. Random sample of that many sessions is chosen.
    - For each chosen session with data_cnt >= 2 and no existing offload
      edit, a random insertion index rand_i is generated and an offload +
      prefetch pair is inserted at that position. Both records inherit the
      session's session_id and parent_session_id.
    - The insertion point is constrained to the trailing data-plane prefix
      of the session: trailing manage_request records (evict / end / offload
      / prefetch) are skipped, so the pause pair never lands right before
      a final evict/end marker. Sessions whose data-plane prefix has < 2
      records are skipped (no room for a pause).
    - Returns a new list with insertions applied; unchanged otherwise.
    """
    if percent is None or percent <= 0:
        return dataset

    # group by session_id (first-seen order)
    groups: dict[str, list[dict]] = {}
    order: list[str] = []
    for rec in dataset:
        sid = (rec.get("agent_hint") or {}).get("session_id", "")
        if sid not in groups:
            groups[sid] = []
            order.append(sid)
        groups[sid].append(rec)

    total = len(order)
    cnt = round(total * percent)
    cnt = max(0, min(cnt, total))
    if cnt == 0:
        return dataset

    chosen = set(random.sample(order, cnt))
    print(f"[Jiuwen] add_pause: {cnt}/{total} sessions get a pause")

    def _has_offload(items):
        for rec in items:
            cm = (rec.get("agent_hint") or {}).get("context_management", {})
            for edit in cm.get("edits", []) or []:
                if isinstance(edit, dict) and edit.get("type") == "offload":
                    return True
        return False

    out: list[dict] = []
    for sid in order:
        items = groups[sid]
        if sid not in chosen or len(items) < 2 or _has_offload(items):
            out.extend(items)
            continue
        data_cnt = len(items)

        # Find the trailing manage-request run (evict/end/offload/prefetch).
        # pause pair must NOT be inserted right before these control-plane
        # records — only between data-plane records. Walk back from the end
        # to find the last data-plane index.
        last_data_idx = data_cnt - 1
        while last_data_idx >= 0:
            ah = items[last_data_idx].get("agent_hint") or {}
            cm = ah.get("context_management", {}) or {}
            if cm.get("manage_request", False):
                last_data_idx -= 1
            else:
                break
        # Need at least 2 data-plane records to insert a pause between them.
        if last_data_idx < 1:
            out.extend(items)
            continue

        # Random insertion point in [1, last_data_idx] so the offload/prefetch
        # pair lands strictly between two data-plane records, never adjacent
        # to a trailing evict/end manage request.
        rand_i = random.randint(1, last_data_idx)
        last_ah = items[-1].get("agent_hint") or {}
        psid = last_ah.get("parent_session_id", "")
        offload_data = _make_manage_request(sid, psid, "offload", swe_format=swe_format)
        prefetch_data = _make_manage_request(sid, psid, "prefetch", swe_format=swe_format)
        print(f"[Jiuwen] add_pause: session {sid}, insert at index {rand_i}/{data_cnt} "
              f"(trailing manage-run starts at {last_data_idx + 1})")
        out.extend(items[:rand_i])
        out.append(offload_data)
        out.append(prefetch_data)
        out.extend(items[rand_i:])
    return out


def print_dataset_structure(dataset, title: str = "Dataset Structure",
                            show_agent_hint: bool = False) -> None:
    """Print the structure of a built dataset, grouped by ParentGroup (parent_session_id)
    and SessionGroup (session_id), showing each record's stage / manage_request / edit types.

    Uses ASCII-only labels to avoid mojibake on Windows GBK consoles. ParentGroup/SessionGroup
    correspond to 大组/小组 in the codebase's Chinese comments.

    Example output::

        ==================== Dataset Structure (12 records, 2 ParentGroups) ====================
        -- ParentGroup[0] parent_session_id=psid_A  (3 SessionGroups, 6 records)
           * SessionGroup[0] session_id=sid_A1  (2 records)
              [0] stage=start        manage=False  edits=[]
              [1] stage=stage-1      manage=False  edits=[]
           * SessionGroup[1] session_id=sid_A2  (2 records)
              [2] stage=             manage=True   edits=['offload']
              [3] stage=             manage=True   edits=['prefetch']
              [4] stage=start        manage=False  edits=[]
           * SessionGroup[2] session_id=sid_A1  (1 records)
              [5] stage=stage-2      manage=False  edits=[]
              [6] stage=stage-3      manage=False  edits=[]
              [7] stage=stage-3      manage=True   edits=['evict']
        -- ParentGroup[1] parent_session_id=psid_B  (1 SessionGroup, 2 records)
          ...

    Note: SessionGroup here is "maximal adjacent run of same session_id" — same as
    group_by_parent_sessions_evict's notion, NOT first-seen grouping. This
    matches the actual physical ordering of records in the output file.
    """
    if not dataset:
        print(f"===== {title}: empty =====")
        return

    # 1. ParentGroup by parent_session_id (first-seen order)
    parent_groups: dict[str, list[dict]] = {}
    parent_order: list[str] = []
    for rec in dataset:
        ah = rec.get("agent_hint") or {}
        psid = ah.get("parent_session_id", "") or ""
        if psid not in parent_groups:
            parent_groups[psid] = []
            parent_order.append(psid)
        parent_groups[psid].append(rec)

    total = len(dataset)
    print(f"========== {title} ({total} records, {len(parent_order)} ParentGroups) ==========")

    for bi, psid in enumerate(parent_order):
        recs = parent_groups[psid]

        # 2. SessionGroup = maximal adjacent runs of same session_id
        session_runs: list[tuple[str, list[dict]]] = []
        for rec in recs:
            sid = (rec.get("agent_hint") or {}).get("session_id", "") or ""
            if session_runs and session_runs[-1][0] == sid:
                session_runs[-1][1].append(rec)
            else:
                session_runs.append((sid, [rec]))

        print(f"-- ParentGroup[{bi}] parent_session_id={psid or '<empty>'}  "
              f"({len(session_runs)} SessionGroups, {len(recs)} records)")

        # 3. absolute index offset: sum of all records in prior ParentGroups
        abs_off = 0
        for bi_prev in range(bi):
            abs_off += len(parent_groups[parent_order[bi_prev]])

        local_idx = 0
        for si, (sid, run) in enumerate(session_runs):
            print(f"   * SessionGroup[{si}] session_id={sid or '<empty>'}  ({len(run)} records)")
            for rec in run:
                ah = rec.get("agent_hint") or {}
                cm = ah.get("context_management", {}) or {}
                edits = [e.get("type", "") for e in (cm.get("edits", []) or []) if isinstance(e, dict)]
                manage = bool(cm.get("manage_request", False))
                stage = ah.get("stage", "") or ""
                edit_str = edits if edits else "[]"
                print(f"      [{abs_off + local_idx}] stage={stage:<10} "
                      f"manage={manage!s:<5}  edits={edit_str}")
                if show_agent_hint:
                    ah_json = json.dumps(ah, ensure_ascii=False, sort_keys=True)
                    print(f"          agent_hint={ah_json}")
                local_idx += 1

    # 4. summary stats
    n_manage = sum(1 for r in dataset
                   if bool(((r.get("agent_hint") or {}).get("context_management", {}) or {})
                           .get("manage_request", False)))
    n_data = len(dataset) - n_manage
    print(f"----- Summary: {n_data} data-plane + {n_manage} manage-request = {len(dataset)} total -----")


# ---------------------------------------------------------------------------
# Jiuwen: data source is a directory of JSON files (one per request).
# Mirrors build_agent_dataset_no_end_c20.py
# ---------------------------------------------------------------------------
@register("Jiuwen")
def process_jiuwen(args: argparse.Namespace) -> None:
    src_dir = resolve_path(args.src)
    if not src_dir.is_dir():
        raise ValueError(f"Jiuwen expects a directory, got: {src_dir}")

    swe_format = getattr(args, "swe_format", False)

    tokenizer = None
    if args.tokenizer:
        try:
            from tokenizers import Tokenizer as HFTokenizer
            tokenizer = HFTokenizer.from_file(args.tokenizer)
        except Exception as e:
            print(f"[warn] failed to load tokenizer from {args.tokenizer}: {e}")

    def extract_s_r(filename: str):
        s_match = re.search(r'_S(\d+)', filename)
        r_match = re.search(r'_R(\d+)', filename)
        s_val = int(s_match.group(1)) if s_match else 99999
        r_val = int(r_match.group(1)) if r_match else 99999
        return (s_val, r_val)

    def get_files(d: Path):
        json_files = [
            f for f in d.iterdir()
            if f.is_file() and f.name.startswith("G") and f.name.endswith(".json")
        ]
        json_files.sort(key=lambda f: extract_s_r(f.name))
        return json_files

    def parse_file_name(file_name: str):
        s = file_name.split("_")
        session_idx = int(s[1][1:])
        request_idx = int(s[2][1:])
        is_manage = (
            (len(s) > 3 and s[3] == "management")
            or (len(s) > 4 and s[4] == "management")
            or (len(s) > 5 and s[5] == "management")
        )
        return (session_idx, request_idx, is_manage)

    def gen_dataset_line(filepath: Path, tokenizer=None):
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        request = data.get("request", {})
        agent_hint = request.get("agent_hint", {})

        if "context_management" in agent_hint and "manage_request" in agent_hint["context_management"]:
            if agent_hint["context_management"]["manage_request"] is True:
                pass
            else:
                agent_hint["context_management"]["edits"] = []

        _, request_idx, _ = parse_file_name(os.path.basename(filepath))
        agent_hint["stage"] = "start" if request_idx == 1 else f"stage-{request_idx-1}"

        # 用 usage 字段计算 max_tokens = total_tokens - prompt_tokens
        usage = data.get("usage") or {}
        total_tokens = usage.get("total_tokens", 0) or 0
        prompt_tokens = usage.get("prompt_tokens", 0) or 0
        max_tokens = total_tokens - prompt_tokens

        if max_tokens == 0:
            max_tokens = 1

        rec = _base_record(swe_format)
        rec["agent_hint"] = agent_hint
        rec["max_tokens"] = max_tokens
        if swe_format:
            # SWE-bench 风格：透传 messages/tools，question 留空
            rec["messages"] = request.get("messages", [])
            rec["tools"] = request.get("tools", [])
            rec["tool_choice"] = "auto"
        else:
            # 普通格式：拼 question 字段
            msg = ". ".join([m["content"] for m in request.get("messages", [])])
            rec["question"] = msg
        return rec

    def group_by_parent_sessions(dataset):
        """Re-group records by ``parent_session_id`` (大组), keeping the
        original read order within each 大组. Within a 大组, records are
        further grouped by ``session_id`` (小组, first-seen order); between
        consecutive 小组 an ``offload`` + ``prefetch`` pair of management
        requests is inserted. Returns a new list in the reordered sequence.
        """
        # 1. 大组 by parent_session_id (first-seen order)
        big_groups: dict[str, list[dict]] = {}
        big_order: list[str] = []
        for rec in dataset:
            psid = (rec.get("agent_hint") or {}).get("parent_session_id", "")
            if psid not in big_groups:
                big_groups[psid] = []
                big_order.append(psid)
            big_groups[psid].append(rec)

        out: list[dict] = []
        total_pairs = 0
        for psid in big_order:
            recs = big_groups[psid]

            # 2. 小组 by session_id within 大组 (first-seen order)
            sub_groups: dict[str, list[dict]] = {}
            sub_order: list[str] = []
            for rec in recs:
                sid = (rec.get("agent_hint") or {}).get("session_id", "")
                if sid not in sub_groups:
                    sub_groups[sid] = []
                    sub_order.append(sid)
                sub_groups[sid].append(rec)

            # 3. 在每两个相邻小组之间插入 offload + prefetch 管理请求
            big_out: list[dict] = []
            n_pairs = 0
            for i, sid in enumerate(sub_order):
                big_out.extend(sub_groups[sid])
                if i < len(sub_order) - 1:
                    offload_data = _make_manage_request(sid, psid, "offload", swe_format=swe_format)
                    prefetch_data = _make_manage_request(sub_order[i + 1], psid, "prefetch", swe_format=swe_format)
                    n_pairs += 1
                    big_out.append(offload_data)
                    big_out.append(prefetch_data)
            total_pairs += n_pairs

            out.extend(big_out)
        if total_pairs:
            print(f"[Jiuwen] group_by_parent_sessions: {total_pairs} offload+prefetch pairs "
                  f"across {len(big_order)} ParentGroups")
        return out

    def group_by_parent_sessions_evict(dataset):
        """Like ``group_by_parent_sessions`` but distinguishes 中断小组 and
        未中断小组 within each 大组 (parent_session_id).

        Steps:
        1. 大组 by ``parent_session_id`` (first-seen order); 大组内保持
           原始数据顺序。
        2. 大组内按 ``session_id`` 的相邻性切分小组（一段连续且相同
           session_id 的记录为一个小组）。
        3. 对每个 session_id 的多个小组两两成对：在前一个小组末尾追加
           offload，在后一个同 sid 小组最前边插入 prefetch（非相邻配对，
           中间可以穿插其他 session_id 的小组）。
           - 大组内某 session_id 的最后一个小组若是大组最后一个小组 ->
             仅追加 evict；若是未中断小组但非大组最后一个小组 ->
             追加 evict + end。
        4. 大组内所有记录的 ``session_id`` 赋值为大组的
           ``parent_session_id``。返回新列表。
        """
        # 1. 大组 by parent_session_id (first-seen order)
        big_groups: dict[str, list[dict]] = {}
        big_order: list[str] = []
        for rec in dataset:
            psid = (rec.get("agent_hint") or {}).get("parent_session_id", "")
            if psid not in big_groups:
                big_groups[psid] = []
                big_order.append(psid)
            big_groups[psid].append(rec)

        out: list[dict] = []
        t_prefetch = t_offload = t_evict = t_evict_end = 0
        for psid in big_order:
            recs = big_groups[psid]

            # 2. 切分小组：maximal adjacent runs of same session_id
            sub_runs: list[tuple[str, list[dict]]] = []
            for rec in recs:
                sid = (rec.get("agent_hint") or {}).get("session_id", "")
                if sub_runs and sub_runs[-1][0] == sid:
                    sub_runs[-1][1].append(rec)
                else:
                    sub_runs.append((sid, [rec]))

            # 3. 统计每个 session_id 在大组内的小组数量
            sid_run_count: dict[str, int] = {}
            for sid, _ in sub_runs:
                sid_run_count[sid] = sid_run_count.get(sid, 0) + 1

            big_out: list[dict] = []
            run_pos: dict[str, int] = {}
            total_runs = len(sub_runs)
            n_prefetch = n_offload = n_evict = n_evict_end = 0
            for run_idx, (sid, run) in enumerate(sub_runs):
                run_pos[sid] = run_pos.get(sid, 0) + 1
                is_first_run_of_sid = run_pos[sid] == 1
                is_last_run_of_sid = run_pos[sid] == sid_run_count[sid]
                is_last_run_in_big = run_idx == total_runs - 1

                # 非首段同 sid 小组：在小组最前边补一个 prefetch
                if not is_first_run_of_sid:
                    prefetch_data = _make_manage_request(sid, psid, "prefetch", swe_format=swe_format)
                    n_prefetch += 1
                    big_out.append(prefetch_data)

                big_out.extend(run)

                if not is_last_run_of_sid:
                    # 中断小组：末尾追加 offload
                    offload_data = _make_manage_request(sid, psid, "offload", swe_format=swe_format)
                    n_offload += 1
                    big_out.append(offload_data)
                elif is_last_run_in_big:
                    # 大组最后一个小组：仅追加 evict
                    evict_data = _make_manage_request(sid, psid, "evict", swe_format=swe_format)
                    n_evict += 1
                    big_out.append(evict_data)
                else:
                    # 未中断小组：evict + end（两条配合）
                    # evict: 携带 evict 信号，pass_agent_hint=True 时由 do_request 发送到推理服务
                    # end: stage="end" 触发 dispatcher split（break before do_request，不发送）
                    #      manage_request=True 标记为控制面（语义一致，虽然该记录本身不会被发送）
                    evict_data = _make_manage_request(sid, psid, "evict", swe_format=swe_format)
                    end_data = _make_manage_request(sid, psid, "", stage="end", manage_request=True, swe_format=swe_format)
                    n_evict_end += 1
                    big_out.append(evict_data)
                    big_out.append(end_data)
            t_prefetch += n_prefetch
            t_offload += n_offload
            t_evict += n_evict
            t_evict_end += n_evict_end

            out.extend(big_out)
        if t_prefetch or t_offload or t_evict or t_evict_end:
            print(f"[Jiuwen] group_by_parent_sessions_evict: across {len(big_order)} BigGroups "
                  f"prefetch={t_prefetch} offload={t_offload} evict={t_evict} evict+end={t_evict_end}")
        return out

    def add_evict_manage_request(dataset):
        """Append an evict management request at the end of each session
        (preserving first-seen session order). The evict record carries the
        same session_id as its group and a stage of ``stage-<n>`` where n is
        the group size.
        """
        groups: dict[str, list[dict]] = {}
        order: list[str] = []
        for rec in dataset:
            sid = rec["agent_hint"].get("session_id", "")
            if sid not in groups:
                groups[sid] = []
                order.append(sid)
            groups[sid].append(rec)

        out: list[dict] = []
        for sid in order:
            items = groups[sid]
            out.extend(items)
            n = len(items)
            last_ah = items[-1]["agent_hint"] if items else {}
            evict_rec = _make_manage_request(
                sid,
                last_ah.get("parent_session_id", ""),
                "evict",
                stage=f"stage-{n}",
                swe_format=swe_format,
            )
            out.append(evict_rec)
        return out

    def remove_manage_request(dataset):
        def _is_manage_request(rec) -> bool:
            ah = rec.get("agent_hint") or {}
            cm = ah.get("context_management", {})
            return cm.get("manage_request", False)

        return [rec for rec in dataset if not _is_manage_request(rec)]

    def renumber_stage_per_session(dataset):
        """Re-assign ``stage`` per session (grouped by ``session_id``) so each
        session has its own ``start → stage-1 → stage-2 → ...`` sequence,
        independent of the global request index from the source filename.

        This fixes the case where a sub-session's first request carries a
        large request_idx (e.g. R0018) and would otherwise get
        ``stage=stage-17`` instead of ``stage=start``.

        Operates in-place; returns the same list.
        """
        # group by session_id (first-seen order)
        groups: dict[str, list[dict]] = {}
        order: list[str] = []
        for rec in dataset:
            sid = (rec.get("agent_hint") or {}).get("session_id", "")
            if sid not in groups:
                groups[sid] = []
                order.append(sid)
            groups[sid].append(rec)
        for sid in order:
            for i, rec in enumerate(groups[sid]):
                ah = rec.get("agent_hint") or {}
                ah["stage"] = "start" if i == 0 else f"stage-{i}"
                rec["agent_hint"] = ah
        return dataset

    def filter_token_len(dataset, max_token_len):
        """Drop records whose ``question`` token count exceeds ``max_token_len``.

        Requires a tokenizer (loaded from --tokenizer). Records with an empty
        question (e.g. management requests) are kept. Returns a new list.
        """
        if tokenizer is None:
            raise ValueError("filter_token_len requires --tokenizer to be set")
        kept = []
        dropped = 0
        for rec in dataset:
            q = rec.get("question", "") or ""
            if q == "":
                kept.append(rec)
                continue
            n = len(tokenizer.encode(q).ids)
            if n > max_token_len:
                dropped += 1
                continue
            kept.append(rec)
        print(f"[Jiuwen] filter_token_len: dropped {dropped} records with token_len > {max_token_len}")
        return kept

    def process_preview(dataset):
        """Replace each record's prompt content with placeholder "preview text"
        in preview mode. No-op when --preview is not set. Returns the same list
        (in-place).
        """
        if not args.preview:
            return dataset
        for rec in dataset:
            if rec.get("question"):
                rec["question"] = "preview text"
            if swe_format:
                if rec.get("messages"):
                    rec["messages"] = [{"content": "preview text", "role": "system"}]
                # tools placeholder left as-is (structure needed for chat API)
        return dataset


    # cut_count：session内部截取数据条数
    def cut_session_inner(dataset, cut_count):
        """Keep only the last ``cut_count`` records of each session (grouped by
        session_id in first-seen order). ``cut_count`` is clamped to the
        actual session length; None keeps all. Kept records are re-staged as
        ``start, stage-1, stage-2, ...``. Returns (new_list, order).
        """
        if cut_count is not None and cut_count < 0:
            raise ValueError("cut_count must be >= 0")

        groups: dict[str, list[dict]] = {}
        order: list[str] = []
        for rec in dataset:
            sid = rec["agent_hint"].get("session_id", "")
            if sid not in groups:
                groups[sid] = []
                order.append(sid)
            groups[sid].append(rec)

        out: list[dict] = []
        cut_total = 0
        for sid in order:
            items = groups[sid]
            n = len(items)
            if cut_count is None:
                kept = items
            else:
                k = min(cut_count, n)
                lo = n - k
                kept = items[lo:]
                cut_total += lo
            # Re-number stage after cut: each session's kept records restart
            # at start → stage-1 → ... (cut changes the per-session length).
            for i, rec in enumerate(kept):
                rec["agent_hint"]["stage"] = "start" if i == 0 else f"stage-{i}"
            out.extend(kept)
        if cut_count is not None and cut_total > 0:
            print(f"[Jiuwen] cut_session_inner: dropped {cut_total} records (kept last {cut_count} per session, {len(order)} sessions)")
        return out, order

    # session_count：截取大组数量（按parent_session_id）
    def cut_session_outer(dataset, session_count):
        """Keep only the first ``session_count`` 大组 (grouped by
        parent_session_id in first-seen order). ``session_count`` is clamped
        to the actual number of 大组; None keeps all. Returns (new_list, order).
        """
        if session_count is not None and session_count < 0:
            raise ValueError("session_count must be >= 0")

        groups: dict[str, list[dict]] = {}
        order: list[str] = []
        for rec in dataset:
            psid = (rec.get("agent_hint") or {}).get("parent_session_id", "")
            if psid not in groups:
                groups[psid] = []
                order.append(psid)
            groups[psid].append(rec)

        if session_count is None:
            kept_order = order
        else:
            kept_order = order[:min(session_count, len(order))]

        out: list[dict] = []
        for psid in kept_order:
            out.extend(groups[psid])
        return out, kept_order

    def copy_big_groups(dataset, n):
        """按 parent_session_id 分大组，将每个大组复制 n 次。

        每个副本的所有记录的 parent_session_id 在原值后拼接一个 8 字符随机串，
        同时 session_id 也拼接同一个随机串，用于区分不同副本。
        n=0 时不复制（仅保留原数据），返回新列表。
        """
        if n is None or n <= 0:
            return dataset

        # 1. 大组 by parent_session_id (first-seen order)
        big_groups: dict[str, list[dict]] = {}
        big_order: list[str] = []
        for rec in dataset:
            psid = (rec.get("agent_hint") or {}).get("parent_session_id", "")
            if psid not in big_groups:
                big_groups[psid] = []
                big_order.append(psid)
            big_groups[psid].append(rec)

        out: list[dict] = []
        for psid in big_order:
            recs = big_groups[psid]
            # 原始大组
            out.extend(recs)
            # 复制 n 次
            for _ in range(n):
                suffix = ''.join(random.choices(string.ascii_letters + string.digits, k=8))
                for rec in recs:
                    new_rec = copy.deepcopy(rec)
                    ah = new_rec.get("agent_hint") or {}
                    orig_psid = ah.get("parent_session_id", "")
                    orig_sid = ah.get("session_id", "")
                    ah["parent_session_id"] = orig_psid + suffix
                    ah["session_id"] = orig_sid + suffix
                    new_rec["agent_hint"] = ah
                    out.append(new_rec)
        return out

    json_files = get_files(src_dir)
    all_records = []
    n_files = len(json_files)
    print(f"[Jiuwen] Reading {n_files} files from {src_dir}")
    for i, filepath in enumerate(json_files, 1):
        all_records.append(gen_dataset_line(filepath, tokenizer))
        if i % 100 == 0 or i == n_files:
            print(f"[Jiuwen]   {i}/{n_files} files parsed")

    # 2. 按max_token_len过滤超长question
    if getattr(args, "max_token_len", None) is not None:
        all_records = filter_token_len(all_records, args.max_token_len)

    # 3. --preview：把question字段替换为占位"preview text"
    all_records = process_preview(all_records)

    # 4. 删除自带的管理请求
    dataset = remove_manage_request(all_records)

    # 4.1 按session_id重新编号stage（每个session独立 start → stage-1 → ...）
    dataset = renumber_stage_per_session(dataset)

    # 5. 每个session内部截取尾部cut_count条数据
    if args.cut_count:
        dataset, _ = cut_session_inner(dataset, args.cut_count)

    # 6. 按parent_session_id分大组，内部按session_id分小组插入管理请求
    # 6.1 --evict：中断小组末尾加 offload、后一段同 sid 小组开头加 prefetch（非相邻配对）；
    #     未中断小组加 evict+end；大组最后小组仅加 evict。
    # 6.2 默认：每两个相邻小组之间插入 offload+prefetch，再在每个 session 末尾追加 evict。
    if getattr(args, "evict", False):
        dataset = group_by_parent_sessions_evict(dataset)
    else:
        dataset = group_by_parent_sessions(dataset)
        # 每个session末尾追加evict管理请求
        dataset = add_evict_manage_request(dataset)

    # 7. 保留前session_count个大组（按parent_session_id分组）
    dataset, order = cut_session_outer(dataset, args.session_count)

    # 8. 按比例在部分session内部随机插入offload+prefetch暂停对
    if args.add_pause is not None:
        dataset = add_pause_in_session(dataset, args.add_pause, swe_format=swe_format)

    # 9. 按parent_session_id分大组复制n次（副本的parent_session_id和session_id拼接随机后缀）
    if getattr(args, "copy", None):
        dataset = copy_big_groups(dataset, args.copy)

    # 10. --preview：打印数据集结构（大组/小组/stage/manage_request）
    if args.preview:
        print_dataset_structure(
            dataset,
            title=f"Jiuwen Preview ({args.output})",
            show_agent_hint=getattr(args, "show_agent_hint", False),
        )

    out_path = resolve_path(args.output)
    with open(out_path, "w", encoding="utf-8") as f:
        for record in dataset:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"[Jiuwen] Done: {len(dataset)} records (from {len(order)} sessions) -> {out_path}")


# ---------------------------------------------------------------------------
# default: data source is a single JSONL file.
# Mirrors add_agent_hint.py
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# default helpers: prefix-grouping, agent_hint injection, evict insertion
# ---------------------------------------------------------------------------
def group_by_token_prefix(token_seqs: list[list[int]], hit_rate: float = 0.0) -> list[list[int]]:
    """Group records into sessions by online prefix-cache hit rate.

    Walks records in order, maintaining a trie for the *current* session.
    For each record, computes its hit rate against the current trie
    (matched_tokens / total_tokens). If hit_rate <= ``hit_rate`` threshold,
    close the current session (start a new one with a fresh trie) and put
    the record into the new session; otherwise add it to the current
    session and update the trie. After the walk, each session is one group.

    Args:
        token_seqs: token-id sequences for each record, in original order.
        hit_rate: threshold in [0.0, 1.0]. A record whose hit rate against
            the current session's trie is <= this value starts a new session.

    Returns:
        List of groups (sessions), each a list of original record indices,
        in the order they were traversed.
    """
    groups: list[list[int]] = []
    cur_trie: dict = {}
    cur_group: list[int] = []

    def compute_hit_and_insert(trie: dict, ids: list[int]) -> float:
        """Return hit rate = matched_prefix_len / len(ids) against ``trie``,
        then insert ``ids`` into ``trie`` (mutates trie)."""
        if not ids:
            return 0.0
        node = trie
        matched = 0
        for tid in ids:
            if tid in node:
                node = node[tid]
                matched += 1
            else:
                break
        # insert full sequence
        node = trie
        for tid in ids:
            node = node.setdefault(tid, {})
        return matched / len(ids)

    for idx, ids in enumerate(token_seqs):
        hr = compute_hit_and_insert(cur_trie, ids)
        if cur_group and hr <= hit_rate:
            # start a new session: close current, reset trie
            groups.append(cur_group)
            cur_group = []
            cur_trie = {}
            # re-insert into the fresh trie (hit rate would be 0 in new trie)
            compute_hit_and_insert(cur_trie, ids)
        cur_group.append(idx)

    if cur_group:
        groups.append(cur_group)
    return groups


def add_agent_hint(records: list[dict], groups: list[list[int]],
                   token_seqs: list[list[int]], tokenizer,
                   swe_format: bool = False) -> None:
    """In-place: decode each record's token sequence back to text, then insert
    ``agent_hint`` with a per-group ``session_id`` (sid_00001, sid_00002, ...)
    and stage (start, stage-1, stage-2, ...). parent_session_id is empty.

    When swe_format is True, the decoded text is placed into the SWE-bench
    ``messages`` field (as a single user message) instead of ``question``.
    """
    for gi, rec_indices in enumerate(groups):
        sid = f"sid_{gi + 1:05d}"
        for i, idx in enumerate(rec_indices):
            rec = records[idx]
            text = tokenizer.decode(token_seqs[idx])
            stage = "start" if i == 0 else f"stage-{i}"
            if swe_format:
                rec["question"] = ""
                rec["messages"] = [{"role": "user", "content": text}]
                rec.setdefault("tools", [dict(t) for t in _SWE_PLACEHOLDER_TOOLS])
                rec["tool_choice"] = "auto"
            else:
                rec["question"] = text
            rec["agent_hint"] = {
                "context_management": {"edits": [], "manage_request": False},
                "parent_session_id": "",
                "session_id": sid,
                "stage": stage,
            }


def append_evict_between_groups(records: list[dict], groups: list[list[int]],
                                swe_format: bool = False) -> list[dict]:
    """Return a new list with an evict manage_request inserted after each
    group's last record. The evict record inherits the group's session_id
    and has stage = stage-<group_size>.
    """
    out: list[dict] = []
    for gi, rec_indices in enumerate(groups):
        sid = f"sid_{gi + 1:05d}"
        for idx in rec_indices:
            out.append(records[idx])
        evict_obj = _make_manage_request(
            sid, "", "evict",
            stage=f"stage-{len(rec_indices)}",
            swe_format=swe_format,
        )
        out.append(evict_obj)
    return out


# ---------------------------------------------------------------------------
# default: data source is a single JSONL file.
# Groups records by shared token-prefix, injects agent_hint, appends evict.
# ---------------------------------------------------------------------------
@register("default")
def process_default(args: argparse.Namespace) -> None:
    """Default processor: read a JSONL dataset, tokenize each record's
    question, group by shared token-prefix (trie), inject agent_hint per
    group, append evict after each group, write to the output file.
    """
    from tokenizers import Tokenizer as HFTokenizer

    swe_format = getattr(args, "swe_format", False)

    src_file = resolve_path(args.src)
    if not src_file.is_file():
        raise ValueError(f"default expects a file, got: {src_file}")

    if not args.tokenizer:
        raise ValueError("default type requires --tokenizer <path to tokenizer.json>")
    tok_path = resolve_path(args.tokenizer)
    if not tok_path.is_file():
        raise ValueError(f"tokenizer file not found: {tok_path}")
    tokenizer = HFTokenizer.from_file(str(tok_path))

    # 1. read all records
    with open(src_file, "r", encoding="utf-8") as f:
        records: list[dict] = []
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))

    if not records:
        print("input file is empty")
        return

    # 2. tokenize each record's "question"
    token_seqs: list[list[int]] = []
    for rec in records:
        text = rec.get("question", "") or ""
        ids = tokenizer.encode(text).ids
        token_seqs.append(ids)

    # 3. group by shared token-prefix (trie), sorted by length asc per group
    groups = group_by_token_prefix(token_seqs, args.hit_rate)

    # 4. decode tokens back to text + insert agent_hint (in-place)
    add_agent_hint(records, groups, token_seqs, tokenizer, swe_format=swe_format)

    # 5. insert evict manage_request after each group
    out_records = append_evict_between_groups(records, groups, swe_format=swe_format)

    # 6. insert offload+prefetch pause pair into a fraction of sessions
    if args.add_pause is not None:
        out_records = add_pause_in_session(out_records, args.add_pause, swe_format=swe_format)

    # 7. --preview: replace prompt text with placeholder, then print structure
    if args.preview:
        for rec in out_records:
            if rec.get("question"):
                rec["question"] = "preview text"
            if swe_format and rec.get("messages"):
                rec["messages"] = [{"content": "preview text", "role": "system"}]
        print_dataset_structure(
            out_records,
            title=f"default Preview ({args.output})",
            show_agent_hint=getattr(args, "show_agent_hint", False),
        )

    # 8. write output
    out_path = resolve_path(args.output)
    with open(out_path, "w", encoding="utf-8") as fout:
        for rec in out_records:
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"[default] Done: {len(out_records)} records in {len(groups)} prefix groups -> {out_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build agent dataset from a configurable source type."
    )
    parser.add_argument(
        "-t", "--src-dataset-type",
        default="default",
        choices=sorted(DATASET_PROCESSORS.keys()),
        help="Source dataset type. Available: " + ", ".join(sorted(DATASET_PROCESSORS.keys()))
             + ". Defaults to 'default'.",
    )
    parser.add_argument(
        "-s", "--src",
        required=True,
        help="Source path. [Jiuwen] a directory of Gxxxx_Sxxxx_Rxxxx_*.json "
             "trajectory files. [default] a JSONL file with a `question` field.",
    )
    parser.add_argument(
        "-o", "--output",
        default="agent_dataset_out.jsonl",
        help="Output JSONL file path. [shared]",
    )
    parser.add_argument(
        "--tokenizer",
        default=None,
        help="Path to tokenizer.json (HuggingFace tokenizers). "
             "[default] REQUIRED — drives token-level prefix grouping. "
             "[Jiuwen] optional — only used by --max-token-len filtering.",
    )
    parser.add_argument(
        "-c", "--cut-count", type=int, default=None,
        help="[Jiuwen only] Keep only the last N records of each session "
             "(grouped by session_id). Clamped to actual session length. "
             "No effect on 'default' (default has no original session_id).",
    )
    parser.add_argument(
        "-n", "--session-count", type=int, default=None,
        help="[Jiuwen only] Keep only the first N 大组 (grouped by "
             "parent_session_id, first-seen order). Clamped to actual count. "
             "No effect on 'default' (default has no parent_session_id).",
    )
    parser.add_argument(
        "-p", "--add-pause", type=float, default=None,
        help="[Jiuwen only] Insert an offload+prefetch pair at a random "
             "position inside this fraction of sessions. E.g. 0.5 means "
             "~50%% of sessions get a pause. Skipped if not set. "
             "No effect on 'default'.",
    )
    parser.add_argument(
        "-P", "--preview", action="store_true", default=False,
        help="[shared] Preview mode: replace each request's question text "
             "(or messages content) with placeholder \"preview text\", and "
             "print the dataset structure (ParentGroup/SessionGroup/stage/"
             "manage_request). Useful for quick smoke tests without sending "
             "real prompt content.",
    )
    parser.add_argument(
        "--show-agent-hint", action="store_true", default=False,
        help="[shared] When used with --preview, also print the full "
             "agent_hint JSON for each record (one extra indented line per "
             "record). Useful for debugging session_id / parent_session_id / "
             "context_management fields. No effect without --preview.",
    )
    parser.add_argument(
        "--evict", action="store_true", default=False,
        help="[Jiuwen only] Use group_by_parent_sessions_evict: within each "
             "大组, 中断小组 get offload+prefetch, 未中断小组 get evict+end "
             "(two records, both manage_request=True), 大组最后小组 gets only "
             "evict. Skips add_evict_manage_request when set. "
             "No effect on 'default' (default has no 大组/小组 structure).",
    )
    parser.add_argument(
        "-r", "--hit-rate", type=float, default=0.0,
        help="[default only] Hit-rate threshold in [0.0, 1.0] for starting a "
             "new session when grouping records by prefix cache. A record "
             "whose hit rate against the current session trie is <= this "
             "value starts a new session. 0.0 = only 0%% hit breaks; 1.0 = "
             "every record starts a new session. No effect on 'Jiuwen' "
             "(Jiuwen uses source-data session_id directly).",
    )
    parser.add_argument(
        "--max-token-len", type=int, default=None,
        help="[Jiuwen only] Drop records whose question field token count "
             "exceeds this value. Requires --tokenizer. Applied before "
             "remove_manage_request. Skipped if not set. "
             "No effect on 'default'.",
    )
    parser.add_argument(
        "--copy", type=int, default=None,
        help="[Jiuwen only] Copy each 大组 (grouped by parent_session_id) n "
             "times. Each copy's parent_session_id and session_id are "
             "suffixed with the same 8-char random string. E.g. --copy 1 "
             "makes 1 extra copy (so 2x total). No effect on 'default'.",
    )
    parser.add_argument(
        "--swe-format", action="store_true", default=False,
        help="[shared] Output SWE-bench-style records: data-plane records "
             "carry messages/tools/tool_choice instead of a flat question "
             "field; manage requests carry placeholder messages/tools for "
             "shape uniformity. [Jiuwen] transparently passes through source "
             "messages/tools. [default] wraps the decoded token text into a "
             "single user message + placeholder tools. Use this for "
             "chat-completions endpoints that expect the messages array.",
    )
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    processor = DATASET_PROCESSORS.get(args.src_dataset_type)
    if processor is None:
        print(f"Unknown --src-dataset-type: {args.src_dataset_type}", file=sys.stderr)
        print(f"Available: {sorted(DATASET_PROCESSORS.keys())}", file=sys.stderr)
        sys.exit(1)

    processor(args)


if __name__ == "__main__":
    main()
