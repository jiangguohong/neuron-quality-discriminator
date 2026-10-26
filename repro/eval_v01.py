#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
判别器 v0.1 · 评估（LH #9 派单）
================================
在 v01_val.jsonl 上跑推理，解析「判定：keep/reject」与金标比对。
输出 per-sample 明细 + 总体/分类准确率 + 报告 md。
v0.1 不设准确率门槛——记录真实值即交付。

运行: python eval_v01.py [--adapter adapters_v01] [--base-model <path>] [--data <dir>]
"""
import argparse
import json
import os
import re
import sys
import time

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# LH R91 §3 通用纪律：输出含 ✓/✗ 或中文的脚本在 Windows（SYSTEM 账户管道 locale=GBK）
# 下 print 即 UnicodeEncodeError 崩（v17 gates 11:11 实证）——stdout/stderr UTF-8 自保，
# 与编排器 env_safe(PYTHONUTF8=1) 构成双保险
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_BASE = r"D:\amd_hip_sdk\models\Qwen2.5-0.5B-Instruct"


def parse_verdict(text: str):
    m = re.search(r"判定[:：]\s*(keep|reject)", text, re.I)
    if m:
        return m.group(1).lower()
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=HERE)
    ap.add_argument("--adapter", default=os.path.join(HERE, "adapters_v01"))
    ap.add_argument("--base-model", default=DEFAULT_BASE)
    ap.add_argument("--max-new-tokens", type=int, default=24)
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args()

    val_path = os.path.join(args.data, "v01_val.jsonl")
    if not os.path.exists(val_path):
        print(f"✗ 缺验证集 {val_path}，先跑 prepare_v01.py")
        raise SystemExit(1)
    samples = []
    with open(val_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                d = json.loads(line)
                samples.append((d["messages"], d.get("meta", {})))

    tok = AutoTokenizer.from_pretrained(args.adapter if os.path.exists(os.path.join(args.adapter, "tokenizer_config.json")) else args.base_model)
    model = AutoModelForCausalLM.from_pretrained(args.base_model, torch_dtype=torch.float32)
    if os.path.exists(os.path.join(args.adapter, "adapter_model.safetensors")):
        model = PeftModel.from_pretrained(model, args.adapter)
        tag = f"base+LoRA({os.path.basename(args.adapter)})"
    else:
        tag = "base-only(无适配器)"
    model.eval()
    torch.set_num_threads(args.threads)
    print(f"评估模型: {tag} | 验证集: {len(samples)} 条")

    rows = []
    t0 = time.time()
    for messages, meta in samples:
        prompt = tok.apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=True)
        inputs = tok(prompt, return_tensors="pt")
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                pad_token_id=tok.pad_token_id or tok.eos_token_id,
            )
        gen = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        pred = parse_verdict(gen)
        gold = meta.get("quality") or ""
        rows.append({
            "pair_id": meta.get("pair_id", ""),
            "text": messages[1]["content"].replace("请审核神经元组合：", ""),
            "gold": gold,
            "pred": pred,
            "correct": (pred == gold),
            "raw": gen.strip().replace("\n", " ")[:120],
        })
        print(f"  [{rows[-1]['pair_id']}] gold={gold} pred={pred} {'✓' if rows[-1]['correct'] else '✗'}")

    n = len(rows)
    n_ok = sum(1 for r in rows if r["correct"])
    n_parse_fail = sum(1 for r in rows if r["pred"] is None)
    for cls in ("keep", "reject"):
        sub = [r for r in rows if r["gold"] == cls]
        if sub:
            acc = sum(1 for r in sub if r["correct"]) / len(sub)
            print(f"  {cls}: {sum(1 for r in sub if r['correct'])}/{len(sub)} = {acc:.0%}")

    report = {
        "version": "v0.1",
        "model": tag,
        "n_val": n,
        "n_correct": n_ok,
        "accuracy": round(n_ok / n, 4) if n else None,
        "parse_fail": n_parse_fail,
        "note": "v0.1 管道验证：不设准确率门槛；小样本（10对量级）下准确率不具统计意义",
        "wall_time_s": round(time.time() - t0, 1),
        "rows": rows,
    }
    rp = os.path.join(args.data, "v01_eval_report.json")
    with open(rp, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    md = [f"# 判别器 v0.1 评估报告（{time.strftime('%Y-%m-%d %H:%M')}）", "",
          f"- 模型：{tag}", f"- 验证集：{n} 条（v0.1 冒烟样本，不设门槛）",
          f"- 总体准确率：**{n_ok}/{n} = {report['accuracy']:.0%}**" if n else "- 总体准确率：n/a",
          f"- 解析失败（未产出 keep/reject）：{n_parse_fail}",
          f"- 推理耗时：{report['wall_time_s']}s（CPU）", "",
          "| pair_id | 组合 | gold | pred | 结果 |", "|---|---|---|---|---|"]
    for r in rows:
        md.append(f"| {r['pair_id']} | {r['text']} | {r['gold']} | {r['pred'] or '(解析失败)'} | {'✓' if r['correct'] else '✗'} |")
    mp = os.path.join(args.data, "v01_eval_report.md")
    with open(mp, "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")
    print(f"\n===== EVAL_DONE → {rp} | {mp} =====")


if __name__ == "__main__":
    main()
