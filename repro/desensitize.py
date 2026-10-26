# -*- coding: utf-8 -*-
"""
D-15 确定性脱敏脚本（WD/飞影 制）
依据 LH R115 / 10-26 14:05 信第四节映射表，对 v17 仓 20 件资产（不含 MANIFEST 自身）做
确定性、幂等、可自验的字符串替换。MANIFEST 由 regen_manifest.py 单独重生。

映射（按序，特异优先）：凭据存放路径 -> <CREDENTIAL_STORE> / 总线端点 -> <BUS_HOST> /
隧道端点 -> <TUNNEL_HOST> / 主机账户名(WD与DSH两侧) -> <USER> / 共享区 -> <SHARED>
（具体被替换的源串见下方 REPLACEMENTS 正则；本段仅列占位目标，不回显被替换的标识值。）

P1 保留项（按 LH 裁定不改）：BUS_TOKEN / HETU_TOKEN / GITHUB_TOKEN 变量名。
不可复现风险：保持脚本可运行性，仅替换标识性路径/端点/账户名，不动代码结构。

用法：python desensitize.py  (在 repo_stage 根以仓库相对结构运行)
"""
import os, re, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # .../repo_stage

# 替换序：特异优先（源串为正则，匹配被替换的标识值；占位目标见上段）
REPLACEMENTS = [
    # 1) 凭据存放路径（最特异，必须在主机账户名通用替换之前吃掉）
    (re.compile(r"C:[\\/]Users[\\/]Jiang\.MACBOOK[\\/]\.hetu[\\/]bus_token\.txt", re.IGNORECASE), "<CREDENTIAL_STORE>"),
    # 2) 总线端点
    (re.compile(r"bus\.hetu\.dpdns\.org", re.IGNORECASE), "<BUS_HOST>"),
    # 3) 隧道端点（须在总线端点之后，避免被二次切割）
    (re.compile(r"hetu\.dpdns\.org", re.IGNORECASE), "<TUNNEL_HOST>"),
    # 4) 主机账户名（WD 侧）
    (re.compile(r"Jiang\.MACBOOK", re.IGNORECASE), "<USER>"),
    # 5) 主机账户名（DSH 侧）
    (re.compile(r"Administrator"), "<USER>"),
    # 6) 共享区
    (re.compile(r"agent_shared"), "<SHARED>"),
]

# 需要处理的资产文件（=MANIFEST 表 20 行 + README，排除 MANIFEST 自身）
ASSET_REL = [
    "corpus/v17_train.jsonl",
    "evidence/leak_recheck_output.txt",
    "evidence/v17_prepare_audit.json",
    "evidence/val168_leak_evidence.json",
    "report/WD_V17_GATES_20261020.md",
    "report/v17_发布载体方案表_20261021.md",
    "report/v17_对外终版报告_v1稿_20261021.md",
    "repro/dsh_r91_leak_check.manifest",
    "repro/dsh_r91_leak_check.py",
    "repro/eval_basis63.py",
    "repro/eval_sets/basis63.json",
    "repro/eval_sets/fitcheck_bare79.jsonl",
    "repro/eval_sets/val168.jsonl",
    "repro/eval_v01.py",
    "repro/prepare_v17.py",
    "repro/run_v17_gates.py",
    "repro/train_v01.py",
    "repro/val168_leak_evidence.py",
    "repro/verify_v17_final.py",
    "README.md",
]

def apply_to_text(text):
    total = 0
    for rx, rep in REPLACEMENTS:
        new, n = rx.subn(rep, text)
        total += n
        text = new
    return text, total

def main():
    summary = []
    for rel in ASSET_REL:
        fp = os.path.join(ROOT, rel)
        if not os.path.exists(fp):
            print(f"MISSING {rel}"); summary.append((rel, "MISSING", 0)); continue
        # 字节级安全：保留原 EOL/编码，唯一差异=映射替换（不用文本模式，避免 LF/CRLF 漂移）
        raw = open(fp, "rb").read()
        text = raw.decode("utf-8")
        new, n = apply_to_text(text)
        if n > 0:
            open(fp, "wb").write(new.encode("utf-8"))
        summary.append((rel, "changed" if n>0 else "clean", n))
        print(f"{'CHG' if n>0 else 'OK '} {rel}: {n} repl")
    # 汇总
    chg = sum(1 for _,s,_ in summary if s=="changed")
    print(f"\nDONE: {chg} files changed, {len(summary)-chg} already clean, {sum(n for _,_,n in summary)} total repl")
    return summary

if __name__ == "__main__":
    main()
