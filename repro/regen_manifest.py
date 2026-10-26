# -*- coding: utf-8 -*-
"""
D-15 步骤：重生 MANIFEST（脱敏后版本）
- 丢弃「二、原路径映射」（内部追溯，不随 public 出库）→ 移入私有台账
- 重算 20 件资产 sha256/字节/行数（脱敏后）
- 加脱敏声明
私有台账写在 repo 外：D:/agent_shared/v17_final_internal/_D15_INTERNAL_LEDGER.md
"""
import hashlib, re, os, datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PRI  = ROOT + "_pristine_baseline"
LEDGER = os.path.join(os.path.dirname(ROOT), "_D15_INTERNAL_LEDGER.md")

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

def sha256_file(fp):
    return hashlib.sha256(open(fp, "rb").read()).hexdigest()

def lines_file(fp):
    return len(open(fp, encoding="utf-8", errors="replace").read().splitlines())

# 读 pristine MANIFEST 的原始 sha + 原路径映射节
pri_man = open(os.path.join(PRI, "MANIFEST.md"), encoding="utf-8").read()
orig_sha = dict(re.findall(r"`([^`]+\.(?:md|json|jsonl|txt|py|manifest))` \| `([0-9a-f]{64})`", pri_man))
# 原路径映射节
m = re.search(r"## 二、原路径映射.*", pri_man, re.S)
mapping_section = m.group(0) if m else ""

# 计算脱敏后新值
rows = []
for i, rel in enumerate(ASSET_REL, 1):
    fp = os.path.join(ROOT, rel)
    raw = open(fp, "rb").read()
    rows.append((i, rel, sha256_file(fp), len(raw), lines_file(fp)))

# 写新 MANIFEST
now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S +08")
out = []
out.append("# MANIFEST — v17 frozen release assets (DESENSITIZED / 确定性脱敏版)")
out.append("")
out.append("> 本文件为 **确定性脱敏版** 资产清单。脱敏映射唯一来源为 `repro/desensitize.py`，")
out.append("> 复现方用同脚本对原始仓跑一遍，应得到与本表逐件一致的差异（仅标识性路径/端点/账户名被替换）。")
out.append("> 原始「仓内相对路径 → 原共享区绝对路径」映射节已移出本公开文件，留存于内部台账（不随 public 出库）。")
out.append("> MANIFEST 自身不入表；其 sha 以发布回执登记为准。脱敏再生时间：" + now)
out.append("")
out.append("## 一、资产清单（19 资产 + README，脱敏后 sha256 / 字节 / 行数）")
out.append("")
out.append("| # | 仓内相对路径 | sha256 全值 | 字节 | 行数 |")
out.append("|---|---|---|---|---|")
for num, path, sha, byt, ln in rows:
    out.append(f"| {num} | `{path}` | `{sha}` | {byt} | {ln} |")
out.append("")
out.append("## 二、脱敏声明（D-15）")
out.append("")
out.append("- 替换项：主机账户名（WD 侧与 DSH 侧）→ `C:/Users/<USER>`；")
out.append("  共享区 → `<SHARED>`；凭据存放路径 → `<CREDENTIAL_STORE>`；")
out.append("  总线端点 → `<BUS_HOST>`；隧道端点 → `<TUNNEL_HOST>`。")
out.append("- 保留项（按 LH 裁定，改名即不可复现）：`BUS_TOKEN` / `HETU_TOKEN` / `GITHUB_TOKEN` 变量名。")
out.append("- 内部标识扫描结论：主机账户名 / 真实端点 / 本机绝对路径三类 P0 经同正则复扫 **零命中**（见脱敏核验记录）。")
out.append("- 内部台账（含原路径映射 + 原 sha↔脱敏 sha 双表）留存于 `v17_final_internal/_D15_INTERNAL_LEDGER.md`，不随 public 出库。")
out.append("")
open(os.path.join(ROOT, "MANIFEST.md"), "w", encoding="utf-8").write("\n".join(out))

# 写私有内部台账
led = []
led.append("# D-15 内部台账（私有 · 不随 public 出库）")
led.append("")
led.append("生成时间：" + now)
led.append("")
led.append("## A. 原「仓内相对路径 → 原共享区绝对路径」映射（公开 MANIFEST 已剔除本节）")
led.append("")
led.append(mapping_section)
led.append("")
led.append("## B. 双表：原 sha256 ↔ 脱敏后 sha256（逐件）")
led.append("")
led.append("| # | 仓内相对路径 | 原 sha256 | 脱敏后 sha256 | 字节(原→新) | 行数(原→新) |")
led.append("|---|---|---|---|---|---|")
for num, path, nsha, nbyt, nln in rows:
    osha = orig_sha.get(path, "N/A")
    # 原始字节/行数
    opath = os.path.join(PRI, path)
    obyt = len(open(opath, "rb").read())
    oln = lines_file(opath)
    led.append(f"| {num} | `{path}` | `{osha}` | `{nsha}` | {obyt}→{nbyt} | {oln}→{nln} |")
open(LEDGER, "w", encoding="utf-8").write("\n".join(led))

print("MANIFEST regenerated:", os.path.join(ROOT, "MANIFEST.md"))
print("Ledger written   :", LEDGER)
print("assets:", len(rows))
