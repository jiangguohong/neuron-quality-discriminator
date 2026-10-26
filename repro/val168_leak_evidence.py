#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""val168_leak_evidence.py — val168 档泄漏三数单独落凭据（LH R91 §6 / R92 挂账1）

背景：DSH 侧无 val168 本地副本（无 GPU、adapter/base 不在其可读域），
第三方视角该档「不可测」。本脚本把 val168 档三数（文本/pid/组合语义）
从 verify_v17_final.py 的 V4 合算中拆出单列，并附评测集来源 sha 前 16，
用于闭合第三视角。basis63 / fitcheck_bare 两档同口径附列对照。

判据（与 verify_v17_final.py V4 完全同口径）：
  文本维 = 剥模板归一后命中训练语料裸文本集合
  pid 维 = meta.pair_id 命中训练语料 pid 集合
  组合语义维 = xy_of（标签末段）命中训练语料组合集合
全部应为 0。
"""
import hashlib
import json
import os
import re
import sys

NQ = os.path.dirname(os.path.abspath(__file__))
PREFIX = "请审核神经元组合："


def norm(t):
    return "".join(str(t).split())


def bare(c):
    i = c.find(PREFIX)
    return c[i + len(PREFIX):] if i >= 0 else c


def ucontent(r):
    for m in r.get("messages", []):
        if m.get("role") == "user":
            return m.get("content", "")
    return r.get("text", "") or ""


def xy_of(c):
    m = re.search(r"标签:\s*(.+)", c)
    if not m:
        return ""
    parts = [p.strip() for p in m.group(1).split(",")]
    return parts[-1] if parts else ""


def sha16_of(path):
    raw = open(path, "rb").read()
    return hashlib.sha256(raw).hexdigest()[:16], hashlib.sha256(raw).hexdigest(), len(raw)


def load_eval_items(name):
    """返回 (items, source_path, sha16, sha_full, bytes, n_lines)"""
    if name == "basis63":
        p = os.path.join(NQ, "v04_eval63", "v04_eval_basis63.json")
        rows = json.load(open(p, encoding="utf-8"))["rows"]
        items = [{"pid": r.get("pair_id", ""), "content": PREFIX + (r.get("text", "") or "")} for r in rows]
        return items, p, *sha16_of(p), len(rows)
    rel = {"fitcheck_bare": os.path.join("_fitcheck", "v01_val.jsonl"),
           "val168": "v01_val.jsonl"}[name]
    p = os.path.join(NQ, rel)
    lines = [l for l in open(p, encoding="utf-8") if l.strip()]
    rows = [json.loads(l) for l in lines]
    items = [{"pid": (r.get("meta") or {}).get("pair_id", ""), "content": ucontent(r)} for r in rows]
    return items, p, *sha16_of(p), len(rows)


def main():
    train_path = os.path.join(NQ, "v17", "v01_train.jsonl")
    train_raw = open(train_path, "rb").read()
    train_sha = hashlib.sha256(train_raw).hexdigest()
    rows = [json.loads(l) for l in train_raw.decode("utf-8").splitlines() if l.strip()]
    pids = {(r.get("meta") or {}).get("pair_id", "") for r in rows}
    corp_bare = {norm(bare(ucontent(r))) for r in rows}
    corp_pid = {p for p in pids if p}
    corp_combo = {norm(xy_of(ucontent(r))) for r in rows if "标签:" in ucontent(r)} - {""}

    out = {
        "purpose": "val168 档泄漏三数单独凭据（闭合 DSH 侧不可测的第三视角）",
        "ruling": "LH R91 §6 / R92 挂账1",
        "train_corpus": {"path": "v17/v01_train.jsonl",
                          "sha256": train_sha, "sha16": train_sha[:16],
                          "bytes": len(train_raw), "lines": len(rows)},
        "gates": {},
    }
    all_zero = True
    for name in ("basis63", "fitcheck_bare", "val168"):
        items, src, s16, sfull, nbytes, nlines = load_eval_items(name)
        t = sum(1 for x in items if norm(bare(x["content"])) and norm(bare(x["content"])) in corp_bare)
        p = sum(1 for x in items if x["pid"] and x["pid"] in corp_pid)
        k = norm
        c = sum(1 for x in items if k(xy_of(x["content"])) and k(xy_of(x["content"])) in corp_combo)
        all_zero &= (t, p, c) == (0, 0, 0)
        out["gates"][name] = {
            "text_hits": t, "pid_hits": p, "combo_hits": c,
            "n_eval_items": len(items),
            "source": {"path": os.path.relpath(src, NQ), "sha16": s16,
                       "sha256": sfull, "bytes": nbytes, "lines": nlines},
        }
        print(f"[{name}] 文本={t} pid={p} 组合语义={c} ｜ n={len(items)} ｜ "
              f"源 {os.path.relpath(src, NQ)} sha16={s16} ({nbytes}B/{nlines}行)")
    out["all_zero"] = bool(all_zero)
    outp = os.path.join(NQ, "v17", "val168_leak_evidence.json")
    with open(outp, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n三档全 0: {all_zero} ｜ 凭据落盘: {outp}")
    sys.exit(0 if all_zero else 2)


if __name__ == "__main__":
    main()
