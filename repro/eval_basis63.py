#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval_basis63.py — 用指定 adapter 对固定 63 条基准复测，产出跨版本可比的准确率。

基准来源：v04_eval63/v04_eval_basis63.json 的 rows 字段（63 条固定样本，v0.1 冒烟集）。
v0.3 / v0.4 均已用同一 63 条复测，故本结果可与历史直接对比：
  v0.3 = 85.7%(54/63) / v0.4 = 82.5%(52/63)。

运行: python eval_basis63.py --adapter adapters_v05 [--out v05/v05_eval_basis63.json]
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

SYSTEM_PROMPT = (
    "你是神经元质量审核助手。给定一组神经元组合（由 '×' 分隔的标签或代号），"
    "判断其质量归类（keep=保留 / reject=丢弃），并给出简洁中文理由。"
    "输出格式：第一行 '判定：keep' 或 '判定：reject'，第二行 'reason：<理由>'。"
)


def parse_verdict(text: str):
    m = re.search(r"判定[:：]\s*(keep|reject)", text, re.I)
    return m.group(1).lower() if m else None


def default_out(adapter: str) -> str:
    """从 adapter 名推导输出路径（防覆盖历史版本结果）。
    教训：v0.9 管线忘传 --out，硬编码默认值把 v0.5 的 basis63 结果覆盖了（P025）。"""
    import re as _re
    base = os.path.basename(adapter.rstrip("/\\"))
    m = _re.match(r"adapters_(v\d+)", base)
    ver = m.group(1) if m else "misc"
    return os.path.join(HERE, ver, f"{ver}_eval_basis63.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--base-model", default=DEFAULT_BASE)
    ap.add_argument("--basis", default=os.path.join(HERE, "v04_eval63", "v04_eval_basis63.json"))
    ap.add_argument("--out", default=None, help="默认按 adapter 名推导 vNN/vNN_eval_basis63.json")
    ap.add_argument("--max-new-tokens", type=int, default=24)
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args()
    if not args.out:
        args.out = default_out(args.adapter)

    basis = json.load(open(args.basis, encoding="utf-8"))
    rows = basis["rows"]
    tok = AutoTokenizer.from_pretrained(
        args.adapter if os.path.exists(os.path.join(args.adapter, "tokenizer_config.json")) else args.base_model)
    model = AutoModelForCausalLM.from_pretrained(args.base_model, torch_dtype=torch.float32)
    if os.path.exists(os.path.join(args.adapter, "adapter_model.safetensors")):
        model = PeftModel.from_pretrained(model, args.adapter)
        tag = f"base+LoRA({os.path.basename(args.adapter)})"
    else:
        tag = "base-only(无适配器)"
    model.eval()
    torch.set_num_threads(args.threads)
    print(f"[eval_basis63] 模型={tag} | 固定基准 {len(rows)} 条")

    t0 = time.time()
    out_rows = []
    for r in rows:
        text, gold = r["text"], r["gold"]
        msgs = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"请审核神经元组合：{text}"},
        ]
        prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = tok(prompt, return_tensors="pt")
        with torch.no_grad():
            gen = model.generate(
                **inputs, max_new_tokens=args.max_new_tokens, do_sample=False,
                temperature=None, top_p=None,
                pad_token_id=tok.pad_token_id or tok.eos_token_id,
            )
        gen_text = tok.decode(gen[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        pred = parse_verdict(gen_text)
        ok = (pred == gold)
        out_rows.append({"pair_id": r["pair_id"], "text": text, "gold": gold,
                         "pred": pred, "correct": ok,
                         "raw": gen_text.strip().replace("\n", " ")[:120]})
        print(f"  [{r['pair_id']}] gold={gold} pred={pred} {'✓' if ok else '✗'}")

    n = len(out_rows)
    n_ok = sum(1 for x in out_rows if x["correct"])
    pf = sum(1 for x in out_rows if x["pred"] is None)
    acc = (n_ok / n) if n else 0.0
    rep = {"version": "v0.5", "model": tag, "n_val": n, "n_correct": n_ok,
           "accuracy": round(acc, 4), "parse_fail": pf,
           "wall_time_s": round(time.time() - t0, 1), "rows": out_rows}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(rep, open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    md = [f"# 判别器 v0.5 · 统一 63 条基准复测", "",
          f"- 模型：{tag}", f"- 基准：{n} 条（固定，跨版本可比）",
          f"- 总体准确率：**{n_ok}/{n} = {acc*100:.1f}%**",
          f"- 解析失败：{pf}", f"- 推理耗时：{rep['wall_time_s']}s（CPU）", "",
          "| pair_id | gold | pred | 结果 |", "|---|---|---|---|"]
    for x in out_rows:
        md.append(f"| {x['pair_id']} | {x['gold']} | {x['pred'] or '(解析失败)'} | {'✓' if x['correct'] else '✗'} |")
    open(args.out.replace(".json", ".md"), "w", encoding="utf-8").write("\n".join(md) + "\n")
    print(f"[eval_basis63] {n_ok}/{n} = {acc*100:.1f}% → {args.out}")
    print(f"BASIS63_ACC={acc*100:.1f} N={n} OK={n_ok}")


if __name__ == "__main__":
    main()
