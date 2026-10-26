#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""v17 终版落盘独立复验（不读 prepare 内存态，全部从 v17/v01_train.jsonl 重算）
复验项（LH R87 §2 装配层冻结清单逐条对）：
  V1 行数 558 / 逐行 JSON 可解析 / keep 250 / reject 308
  V2 eval_hit 7 条 pid 全部不在终版
  V3 head 三口径带标签：head_final_strict=116（判据，六字段全）/ head_loose=118（参考）/ head_half_rows=2（LH R91 §2）
  V4 三维零交集：文本（剥模板归一）/ pid / 组合语义（xy_of，全评测集口径 basis63+val168+fitcheck_bare）全 0
  V5 LP-WD 实并 13（076 不在）；终版 sha256/bytes 落证
"""
import hashlib
import json
import os
import re
import sys

# LH R91 §3 通用纪律：输出含 ✓/✗/✅/✗ 的脚本在 Windows（SYSTEM 账户管道 locale=GBK）
# 下 print 即 UnicodeEncodeError 崩——stdout/stderr UTF-8 自保（与编排器 env_safe 双保险）
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

NQ = os.path.dirname(os.path.abspath(__file__))
PREFIX = "请审核神经元组合："
EVAL_HIT7 = {"LP-WD-001", "LP-WD-002", "LP-WD-005", "LP-WD-008",
             "LP-WD-009", "LP-WD-026", "LP-WD-029"}
KEEP13 = {"LP-WD-010", "LP-WD-011", "LP-WD-012", "LP-WD-013", "LP-WD-014",
          "LP-WD-021", "LP-WD-040", "LP-WD-053", "LP-WD-054", "LP-WD-055",
          "LP-WD-065", "LP-WD-073", "LP-WD-075"}


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


def has_head(text):
    return ("类型:" in text) and ("摘要:" in text)


def main():
    path = os.path.join(NQ, "v17", "v01_train.jsonl")
    raw = open(path, "rb").read()
    sha_full = hashlib.sha256(raw).hexdigest()
    rows = []
    for i, ln in enumerate(raw.decode("utf-8").splitlines(), 1):
        if not ln.strip():
            print(f"✗ V1 第 {i} 行为空行"); sys.exit(1)
        try:
            rows.append(json.loads(ln))
        except Exception as e:
            print(f"✗ V1 第 {i} 行 JSON 解析失败: {e}"); sys.exit(1)
    ok = True

    # V1
    keep = sum(1 for r in rows if (r.get("meta") or {}).get("quality") == "keep")
    rej = sum(1 for r in rows if (r.get("meta") or {}).get("quality") == "reject")
    v1 = (len(rows), keep, rej) == (558, 250, 308)
    ok &= v1
    print(f"V1 {'✅' if v1 else '✗'} 行数={len(rows)} keep={keep} reject={rej}（应 558/250/308）")

    # V2
    pids = {(r.get("meta") or {}).get("pair_id", "") for r in rows}
    leaked = sorted(pids & EVAL_HIT7)
    v2 = not leaked
    ok &= v2
    print(f"V2 {'✅' if v2 else '✗'} eval_hit 7 条在终版中出现: {leaked if leaked else '无'}")

    # V3 head 三口径带标签（LH R91 §2：唯一判据 = 严格六字段；禁裸数字）
    HEAD_FIELDS = ["类型:", "代数:", "标签:", "摘要:", "来源记忆:"]
    ucs = [ucontent(r) for r in rows]
    head_loose = sum(1 for c in ucs if ("类型:" in c) and ("摘要:" in c))
    head_strict = sum(1 for c in ucs if all(f in c for f in HEAD_FIELDS))
    half_rows = sorted((r.get("meta") or {}).get("pair_id", "")
                       for r, c in zip(rows, ucs)
                       if ("类型:" in c) and ("摘要:" in c) and not all(f in c for f in HEAD_FIELDS))
    v3 = (head_loose, head_strict, half_rows) == (118, 116, ["HYA-REJ-001", "HYA-REJ-008"])
    ok &= v3
    print(f"V3 {'✅' if v3 else '✗'} head 三口径：head_final_strict={head_strict}（判据，六字段全）｜ "
          f"head_loose={head_loose}（参考，类型+摘要）｜ head_half_rows={len(half_rows)}={half_rows}"
          f"（cluster_summary 缺「来源记忆」）｜ 151-33 派生式禁作登记值（LH R91 §2）")

    # V4 三维零交集（全评测集口径）
    ev = {}
    b = json.load(open(os.path.join(NQ, "v04_eval63", "v04_eval_basis63.json"), encoding="utf-8"))["rows"]
    ev["basis63"] = [{"pid": r.get("pair_id", ""), "content": PREFIX + (r.get("text", "") or "")} for r in b]
    for name, rel in [("fitcheck_bare", os.path.join("_fitcheck", "v01_val.jsonl")),
                      ("val168", "v01_val.jsonl")]:
        ev[name] = [{"pid": (r.get("meta") or {}).get("pair_id", ""), "content": ucontent(r)}
                    for r in (json.loads(l) for l in open(os.path.join(NQ, rel), encoding="utf-8") if l.strip())]
    corp_bare = {norm(bare(ucontent(r))) for r in rows}
    corp_pid = {p for p in pids if p}
    corp_combo = {norm(xy_of(ucontent(r))) for r in rows if "标签:" in ucontent(r)} - {""}
    t_hit = p_hit = c_hit = 0
    for name, items in ev.items():
        for x in items:
            if norm(bare(x["content"])) and norm(bare(x["content"])) in corp_bare:
                t_hit += 1
            if x["pid"] and x["pid"] in corp_pid:
                p_hit += 1
            k = norm(xy_of(x["content"]))
            if k and k in corp_combo:
                c_hit += 1
    v4 = (t_hit, p_hit, c_hit) == (0, 0, 0)
    ok &= v4
    print(f"V4 {'✅' if v4 else '✗'} 三维零交集：文本={t_hit} pid={p_hit} 组合语义={c_hit}（应全 0）")

    # V5
    lpwd_in = sorted(pids & (KEEP13 | {"LP-WD-076"}))
    v5 = lpwd_in == sorted(KEEP13)
    ok &= v5
    print(f"V5 {'✅' if v5 else '✗'} LP-WD 实并 {len(lpwd_in)} 条（076 在? {'076' in pids}）")

    print(f"\n终版凭据: {path}")
    print(f"  sha256 = {sha_full}")
    print(f"  bytes  = {len(raw)}  lines = {len(rows)}")
    print("\n复验总结: " + ("全部 PASS ✅" if ok else "存在 FAIL ✗"))
    sys.exit(0 if ok else 2)


if __name__ == "__main__":
    main()
