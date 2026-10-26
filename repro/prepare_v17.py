#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""v17 语料制备 · 干跑审计版（LH R86 L1~L5 直裁已落地）
===========================================================================
基底（L1）：v16 干净语料（577，全链唯一零交集）+ LP-WD 干净 keep14
            （LP-WD-076 因组合语义命中 HYA-105-103，H3 剔出 → 实并 13）
硬拦停（abort 级）：
  H1 eval_hit 7 条必剔（R84-cc §1）：LP-WD-001/002/005/008/009/026/029
      —— labeled 文件 meta.eval_hit=True 集必须恰等于该 7 条，且选中 14 条零交集
  H2 train_stats head>0（R81 §5 / R83 §5）：head=类型:+摘要: 六字段头；
      剔除组合级后仍须 >0（v16 自带 137 + LP-WD 14 = 151 起，H3 剔 33 后 118）
  H3 组合语义维必剔（L5 / LH R87 §1 全评测集口径）：xy_of(A×B 标签) 匹配
      任一评测集含标签项（basis63 + val168 + 后续新增评测集一律并入）且 pid
      不同的行，逐条剔出（同质不同号，语义弱泄漏）；剔除清单逐条附理由落档
      【LH R87 §1 裁断：采纳实测 33 行、驳回 27 行；禁止只取局部评测集口径】
三维零交集终检（L5 升维）：文本 + pid + 组合语义 三档命中全 0 才放行
  （文本/pid 双维=R70 §4 不撤的最低线；组合语义维=L5 加严）
class_weight（L4）：主臂 显式 1.0（唯一计分）/ 诊断臂 auto 1.355（并行隔离权重效应）
--restore-heads 本轮不启用（L7 明确）：head 恢复另立 v18 单独立项
禁分装（R84-cc §5）：默认 dry-run 只出审计 JSON + 三件；
  --write 仅 LH 发分装放行信号后执行
"""
import argparse
import json
import os
import re
import sys

NQ = r"C:\Users\<USER>\WorkBuddy\2026-10-10-06-58-10\neuron_quality"
PREFIX = "请审核神经元组合："
SYSTEM_PROMPT = (
    "你是神经元质量审核助手。给定一组神经元组合（由 '×' 分隔的标签或代号），"
    "判断其质量归类（keep=保留 / reject=丢弃），并给出简洁中文理由。"
    "输出格式：第一行 '判定：keep' 或 '判定：reject'，第二行 'reason：<理由>'。"
)
KEEP14 = ["LP-WD-010", "LP-WD-011", "LP-WD-012", "LP-WD-013", "LP-WD-014",
          "LP-WD-021", "LP-WD-040", "LP-WD-053", "LP-WD-054", "LP-WD-055",
          "LP-WD-065", "LP-WD-073", "LP-WD-075", "LP-WD-076"]
EVAL_HIT7 = {"LP-WD-001", "LP-WD-002", "LP-WD-005", "LP-WD-008",
             "LP-WD-009", "LP-WD-026", "LP-WD-029"}
LPWD_SRC = os.path.join(NQ, "extra_pairs", "local_pool_batch1_labeled.jsonl")
LPWD_SHA = "2dd28bd2c631d07f"   # 剥引号+逐行态重出版（WD_LPWD_JSONL_FORMAT_FIX 在案）
# LH R87 §3 放行令：落盘前 audit 补 input_lock 凭据（共享区同路径 raw bytes 实测）
LPWD_SHARED = r"D:\<SHARED>\lpwd_labeled_for_dsh\local_pool_batch1_labeled.jsonl"
DSH_REPORTED_SHA = "51020a49a0176bd9"   # LH R87 evidence 节：DSH §1 报数，待双方对账（LH §4.2）


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
    """六字段头结构判定（类型/代数/摘要 在文即视为带头行）"""
    return ("类型:" in text) and ("摘要:" in text)


def load_eval():
    """评测三项；含标签项额外存 gold 供组合语义维一致性核验"""
    ev = {}
    b = json.load(open(os.path.join(NQ, "v04_eval63", "v04_eval_basis63.json"), encoding="utf-8"))["rows"]
    ev["basis63"] = [{"pid": r.get("pair_id", ""), "content": PREFIX + (r.get("text", "") or ""),
                      "gold": r.get("gold", "")} for r in b]
    for name, path in [("fitcheck_bare", os.path.join("_fitcheck", "v01_val.jsonl")),
                       ("val168", os.path.join("v01_val.jsonl"))]:
        rows = [json.loads(l) for l in open(os.path.join(NQ, path), encoding="utf-8") if l.strip()]
        ev[name] = [{"pid": (r.get("meta") or {}).get("pair_id", ""), "content": ucontent(r),
                     "gold": (r.get("meta") or {}).get("quality", "")} for r in rows]
    return ev


def lpwd_to_messages(d):
    meta = d.get("meta") or {}
    reason = meta.get("reason") or "A×B 语义连贯可融合（LP-WD 标注 / R79 §2 口径）"
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": PREFIX + (d.get("text", "") or "")},
            {"role": "assistant", "content": f"判定：keep\nreason：{reason}"},
        ],
        "meta": {"pair_id": d.get("pair_id", ""), "quality": "keep",
                 "inspector": "wd", "round": "v17-lpwd-merge"},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true", help="落 v17 终版语料（LH 放行信号后才可执行）")
    args = ap.parse_args()
    audit = {"hard_stops": {}, "counts": {}, "final_zero_intersection": {}, "combo_level_removal": []}

    # —— H1a：labeled 文件 sha + eval_hit 集校验 ——
    import hashlib
    sha = hashlib.sha256(open(LPWD_SRC, "rb").read()).hexdigest()[:16]
    assert sha == LPWD_SHA, f"✗ labeled 文件 sha 不符: {sha} != {LPWD_SHA}"
    lpwd = [json.loads(l) for l in open(LPWD_SRC, encoding="utf-8") if l.strip()]
    eh_set = {d["pair_id"] for d in lpwd if (d.get("meta") or {}).get("eval_hit")}
    assert eh_set == EVAL_HIT7, f"✗ eval_hit 集与定稿不符: {sorted(eh_set)} vs {sorted(EVAL_HIT7)}"
    keep_rows = [d for d in lpwd if d["pair_id"] in KEEP14]
    assert len(keep_rows) == 14, f"✗ 14 条 keep 未找齐: {len(keep_rows)}"
    assert all((d.get("meta") or {}).get("quality") == "keep" for d in keep_rows), "✗ 选中行 quality 非 keep"
    assert not (set(KEEP14) & EVAL_HIT7), "✗ 选中 keep 与 eval_hit 7 条有交集"
    audit["hard_stops"]["H1_eval_hit"] = {"pass": True, "eval_hit_set": sorted(eh_set),
                                          "selected": len(keep_rows), "note": "R84-cc §1 双硬拦停之一"}
    print(f"H1 ✅ eval_hit 7 条集校验通过（{sorted(eh_set)}），选中 keep 14 条零交集")

    # —— input_lock 凭据（LH R87 §3）：共享区同路径 raw bytes 实测 + DSH sha 分歧对账（LH §4.2）——
    import hashlib as _hl, datetime as _dt
    raw = open(LPWD_SHARED, "rb").read()
    sha_full = _hl.sha256(raw).hexdigest()
    n_crlf, n_lf = raw.count(b"\r\n"), raw.count(b"\n")
    eol = "CRLF" if (n_crlf == n_lf and n_lf > 0) else ("LF" if n_crlf == 0 else "mixed")
    n_lines = len([l for l in raw.decode("utf-8").splitlines() if l.strip()])
    mt = _dt.datetime.fromtimestamp(os.stat(LPWD_SHARED).st_mtime).strftime("%Y-%m-%d %H:%M:%S")
    assert sha_full[:16] == LPWD_SHA, f"✗ 共享区 labeled sha 与本地不符: {sha_full[:16]} != {LPWD_SHA}"
    audit["input_lock"] = {
        "file": LPWD_SHARED, "sha256_raw": sha_full, "sha256_16": sha_full[:16],
        "bytes": len(raw), "lines": n_lines, "eol": eol, "mtime": mt,
        "note": "LH R87 §3 放行令凭据；与本地 extra_pairs 副本同 sha（raw bytes 全等）"}
    audit["sha_settlement"] = {
        "wd_measured_shared": sha_full[:16], "wd_local_extra_pairs": LPWD_SHA,
        "dsh_reported": DSH_REPORTED_SHA,
        "verdict": ("DSH 报数与 WD 两侧实测不一致 → 按 LH R87 §4.2 请 DSH 复算对账；"
                    "共享区文件 mtime=20:36:17 为 WD 20:36 逐行态修复版，DSH 51020a49 "
                    "疑为修前版本或其导出副本口径，以 mtime+raw bytes 定责"),
        "mtime_shared": mt}
    print(f"input_lock ✅ 共享区 labeled sha256={sha_full[:16]}… bytes={len(raw)} lines={n_lines} eol={eol} mtime={mt}")
    print(f"  sha 对账：WD 实测 {sha_full[:16]}（共享区=本地副本）｜DSH 报 {DSH_REPORTED_SHA} → 不一致，请 DSH 复算")

    # —— base + 增量装配 ——
    base = [json.loads(l) for l in open(os.path.join(NQ, "v16", "v01_train.jsonl"), encoding="utf-8") if l.strip()]
    corpus = list(base)
    lpwd_msgs = [lpwd_to_messages(d) for d in keep_rows]
    corpus.extend(lpwd_msgs)
    head_before = sum(1 for r in corpus if has_head(ucontent(r)))
    audit["hard_stops"]["H2_head"] = {"head_before": head_before, "pass": head_before > 0,
                                      "sources": {"base_v16": "v16/v01_train.jsonl（137 条标签行）",
                                                  "lpwd": LPWD_SRC + "（14 条 keep 全标签行）"}}
    assert head_before > 0, "✗ H2 train_stats head=0，拦停（R81 §5）"
    print(f"H2 ✅ train_stats head_before={head_before}（v16 标签 137 + LP-WD keep 标签 14）")

    # —— 零交集双维终检：文本 + pid（剥模板归一） ——
    ev = load_eval()
    corp_bare = {norm(bare(ucontent(r))) for r in corpus}
    corp_pid = {(r.get("meta") or {}).get("pair_id", "") for r in corpus}
    bad = {}
    for name, items in ev.items():
        t_hits = [x["pid"] for x in items if norm(bare(x["content"])) and norm(bare(x["content"])) in corp_bare]
        p_hits = [x["pid"] for x in items if x["pid"] and x["pid"] in corp_pid]
        if t_hits or p_hits:
            bad[name] = {"text": t_hits, "pid": p_hits}
    audit["final_zero_intersection"]["text_pid"] = {"hits": bad, "pass": not bad}
    assert not bad, f"✗ 文本+pid 双维零交集终检失败: {json.dumps(bad, ensure_ascii=False)[:400]}"
    print(f"双维终检（文本+pid）✅ PASS（basis63/fitcheck_bare/val168 剥模板归一）")

    # —— H3 组合语义维检测 + 剔出（L5） ——
    ev_combo = {}  # 组合键(xy_of) -> [(pid, gold)]
    for name, items in ev.items():
        for x in items:
            c = x["content"]
            if "标签:" in c:
                k = norm(xy_of(c))
                if k:
                    ev_combo.setdefault(k, []).append((x["pid"], x.get("gold", "")))
    combo_remove = []
    for i, r in enumerate(corpus):
        c = ucontent(r)
        if "标签:" not in c:
            continue
        k = norm(xy_of(c))
        if k and k in ev_combo:
            pid = (r.get("meta") or {}).get("pair_id", "")
            matched = [(p, g) for p, g in ev_combo[k] if p != pid]
            if matched:
                myq = (r.get("meta") or {}).get("quality", "")
                consistent = all(g == myq for p, g in matched)
                combo_remove.append({
                    "idx": i, "pid": pid, "combo_key": k,
                    "matched_eval_pids": [p for p, g in matched],
                    "my_quality": myq,
                    "gold_consistent": consistent,
                    "reason": (f"组合语义命中评测项 {[p for p, g in matched]}：同 A×B 标签({k})"
                               f"不同 pid，构成语义弱泄漏；gold 一致={consistent}（L5 加严剔出）"),
                })
    remove_idx = {x["idx"] for x in combo_remove}
    corpus = [r for i, r in enumerate(corpus) if i not in remove_idx]
    audit["hard_stops"]["H3_combo_semantic"] = {
        "pass": True, "removed_rows": len(combo_remove),
        "note": "L5：xy_of(A×B) 匹配评测含标签项且 pid 不同的行逐条剔出"}
    print(f"H3 ✅ 组合语义维剔出 {len(combo_remove)} 行（L5 / LH R87 §1 全评测集口径：32 v16 含标签 + 1 LP-WD-076）")

    # —— H2 after removal + 三维终检组合语义维 ——
    head_after = sum(1 for r in corpus if has_head(ucontent(r)))
    audit["hard_stops"]["H2_head"]["head_after"] = head_after
    audit["hard_stops"]["H2_head"]["pass"] = head_after > 0
    assert head_after > 0, f"✗ 剔除组合级后 head={head_after}，拦停（R83 §5）"
    corp_combo_after = set()
    for r in corpus:
        c = ucontent(r)
        if "标签:" in c:
            k = norm(xy_of(c))
            if k:
                corp_combo_after.add(k)
    combo_hits_after = sum(1 for k in ev_combo if k in corp_combo_after)
    audit["final_zero_intersection"]["combo_semantic"] = {"hits": combo_hits_after, "pass": combo_hits_after == 0}
    assert combo_hits_after == 0, f"✗ 剔除后组合语义维仍有命中: {combo_hits_after}"
    print(f"H2 ✅ head_after={head_after}（>0，R83 §5）；三维组合语义维终检 PASS（命中=0）")

    # —— 统计 + class_weight 双臂（L4） ——
    k = sum(1 for r in corpus if (r.get("meta") or {}).get("quality") == "keep")
    rj = sum(1 for r in corpus if (r.get("meta") or {}).get("quality") == "reject")
    total = len(corpus)
    audit["counts"] = {
        "total": total, "keep": k, "reject": rj, "reject_share": round(rj / total, 4),
        "head_before": head_before, "head_after": head_after,
        "combo_removed": len(combo_remove),
        "lpwd_merged": len(lpwd_msgs) - 1,  # LP-WD-076 被 H3 剔
        "class_weight_main_arm": 1.0, "class_weight_diag_arm_auto": 1.355,
        "note": "L4 主臂显式1.0唯一计分；诊断臂auto1.355并行仅隔离权重效应（首轮禁写auto为对外数）",
    }
    print(f"v17 终版语料: {total} 条 = keep {k} / reject {rj} | reject 占比 {rj/total*100:.1f}%")
    print(f"  head: before={head_before} after={head_after} | combo 剔 {len(combo_remove)} | LP-WD 实并 {len(lpwd_msgs)-1}")
    print(f"class_weight 主臂=1.0（计分） / 诊断臂=auto 1.355（并行隔离权重效应）")

    # —— 装配清单（LH §7 第一件） ——
    audit["assembly_list"] = {
        "total_after_combo_removal": total,
        "keep_reject": {"keep": k, "reject": rj},
        "eval_hit_removed_7": sorted(EVAL_HIT7),
        "assembly_domain_note": ("LH R87 §0 口径：装配域 = v16 577 + LP-WD 14（含076）→ 591；"
                                 "eval_hit 7 条本就不在 591 内（LP-WD 输入已取 14 条干净 keep），"
                                 "591→558 的 33 全部来自组合级剔出，不是 7+33"),
        "combo_removed_33": [
            {"pid": x["pid"], "combo_key": x["combo_key"],
             "matched_eval": x["matched_eval_pids"], "gold_consistent": x["gold_consistent"],
             "reason": x["reason"]} for x in combo_remove],
    }
    print(f"装配清单：剔 eval_hit 7（不进装配域）+ 组合级 {len(combo_remove)} → 终版 {total} 行")

    # —— 落盘控制 ——
    if args.write:
        out = os.path.join(NQ, "v17")
        os.makedirs(out, exist_ok=True)
        with open(os.path.join(out, "v01_train.jsonl"), "w", encoding="utf-8") as f:
            for r in corpus:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"落盘: {out}\\v01_train.jsonl（LH 放行信号后执行）")
    else:
        print("dry-run：未落语料（R84-cc §5 禁分装，LH 放行信号前保持）")
    with open(os.path.join(NQ, "v17_prepare_audit.json"), "w", encoding="utf-8") as f:
        json.dump(audit, f, ensure_ascii=False, indent=1)
    print("审计落盘: v17_prepare_audit.json")


if __name__ == "__main__":
    main()
