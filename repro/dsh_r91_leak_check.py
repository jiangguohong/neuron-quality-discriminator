# -*- coding: utf-8 -*-
# DSH 侧独立实现，未复用 WD 侧脚本；用于 v17 训练后三维泄漏零交集复核
"""v17 训练后三维泄漏零交集复核（DSH 独立实现）

用法：
  python dsh_r91_leak_check.py --train <v17_train.jsonl> --basis <basis63.json> --fit <fitcheck79.jsonl> [--val168 <val168.jsonl>]

依赖：仅 Python 3.13 标准库（json/os/sys/re/argparse/collections），无第三方依赖、无 GPU。
期望输出：文本维 0 / pid 维 0 / 组合语义维 0（basis63 + fitcheck_bare 两档；val168 给了才三档，否则按不可测登记）。
"""
import json, os, re, sys, argparse
from collections import Counter

def load_jsonl(path):
    out = []
    if not os.path.exists(path): return out
    with open(path, encoding='utf-8', errors='replace') as f:
        for line in f:
            line = line.strip()
            if not line: continue
            try: out.append(json.loads(line))
            except Exception: pass
    return out

def load_json(path):
    with open(path, encoding='utf-8', errors='replace') as f:
        return json.load(f)

def norm(s):
    return re.sub(r'[\s\u3000\xa0]', '', s)

def user_content(r):
    for m in r.get('messages', []):
        if m.get('role') == 'user' and m.get('content'):
            t = m['content'].strip()
            t = re.sub(r'^请审核神经元组合[:：]?\s*', '', t)
            return t
    return None

def extract_text(r):
    if 'messages' in r and isinstance(r['messages'], list):
        last = r['messages'][-1]
        if isinstance(last, dict) and last.get('content'):
            return last['content'].strip()
    for k in ('text', 'content', 'prompt_text', 'source_text'):
        v = r.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return None

def main():
    ap = argparse.ArgumentParser(description='v17 训练后三维泄漏零交集复核（DSH 独立实现）')
    ap.add_argument('--train', required=True, help='训练语料 jsonl 路径')
    ap.add_argument('--basis', required=True, help='basis63 评测 json 路径（扁平 rows 结构）')
    ap.add_argument('--fit', required=True, help='fitcheck79 评测 jsonl 路径（messages 数组结构）')
    ap.add_argument('--val168', default=None, help='val168 评测路径（可选；不选则按不可测登记）')
    args = ap.parse_args()

    # 1. 训练语料
    train = load_jsonl(args.train)
    t_texts = set(); t_pairs = set()
    for r in train:
        t = user_content(r) or extract_text(r)
        if t: t_texts.add(t)
        pid = r.get('meta', {}).get('pair_id') or r.get('pair_id')
        if pid: t_pairs.add(str(pid))
    print(f"train: {len(train)} 行 / {len(t_texts)} 文本 / {len(t_pairs)} pid")

    # 2. 评测集
    # basis63：扁平 rows（text + pair_id）
    basis_texts = {}; basis_pairs = set()
    j = load_json(args.basis)
    for row in j.get('rows', []):
        basis_texts[row['text'].strip()] = row.get('pair_id', '')
        if row.get('pair_id'): basis_pairs.add(str(row['pair_id']))
    print(f"basis63: {len(basis_texts)} 文本 / {len(basis_pairs)} pid")

    # fitcheck：messages 数组
    fit_texts = {}; fit_pairs = set()
    for r in load_jsonl(args.fit):
        t = user_content(r) or extract_text(r)
        if t:
            fit_texts[t] = r.get('pair_id', '')
            if r.get('pair_id'): fit_pairs.add(str(r['pair_id']))
    print(f"fitcheck_bare: {len(fit_texts)} 文本 / {len(fit_pairs)} pid")

    # 3. 三维零交集
    ib = t_texts & set(basis_texts.keys())
    ifit = t_texts & set(fit_texts.keys())
    n_t = {norm(t) for t in t_texts}
    n_b = {norm(t) for t in basis_texts}
    n_f = {norm(t) for t in fit_texts}
    pb = t_pairs & basis_pairs
    pf = t_pairs & fit_pairs
    print(f"文本维: ∩basis63={len(ib)} ∩fit_bare={len(ifit)} union={len(ib|ifit)}")
    print(f"pid 维: ∩basis63={len(pb)} ∩fit_bare={len(pf)}")
    print(f"组合语义维: ∩basis63={len(n_t&n_b)} ∩fit_bare={len(n_t&n_f)}")

    # 4. val168（可选）
    if args.val168 and os.path.exists(args.val168):
        v168_texts = set(); v168_pairs = set()
        # val168 是 JSONL（messages+meta 每行一对象），需 load_jsonl + 提取
        for row in load_jsonl(args.val168):
            t = user_content(row) or extract_text(row)
            if t: v168_texts.add(t)
            pid = row.get('meta', {}).get('pair_id') or row.get('pair_id')
            if pid: v168_pairs.add(str(pid))
        iv = t_texts & v168_texts
        n_v = {norm(t) for t in v168_texts}
        print(f"val168: {len(v168_texts)} 文本 / {len(v168_pairs)} pid")
        print(f"val168 文本维: ∩={len(iv)} | pid 维: ∩={len(t_pairs & v168_pairs)} | 语义维: ∩={len(n_t & n_v)}")
    else:
        print("val168: 未提供 → 按「不可测」登记")

    # 5. 结论
    all_zero = len(ib|ifit) == 0 and len(pb|pf) == 0 and len((n_t&n_b)|(n_t&n_f)) == 0
    print(f"\n结论: {'PASS 三维全 0（basis63+fitcheck_bare+val168 三档口径）' if all_zero else 'FAIL 存在命中！'}")

if __name__ == '__main__':
    main()
