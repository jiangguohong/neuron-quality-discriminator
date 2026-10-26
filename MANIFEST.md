# MANIFEST — v17 frozen release assets (DESENSITIZED / 确定性脱敏版)

> 本文件为 **确定性脱敏版** 资产清单。脱敏映射唯一来源为 `repro/desensitize.py`，
> 复现方用同脚本对原始仓跑一遍，应得到与本表逐件一致的差异（仅标识性路径/端点/账户名被替换）。
> 原始「仓内相对路径 → 原共享区绝对路径」映射节已移出本公开文件，留存于内部台账（不随 public 出库）。
> MANIFEST 自身不入表；其 sha 以发布回执登记为准。脱敏再生时间：2026-10-26 16:11:34 +08

## 一、资产清单（19 资产 + README，脱敏后 sha256 / 字节 / 行数）

| # | 仓内相对路径 | sha256 全值 | 字节 | 行数 |
|---|---|---|---|---|
| 1 | `corpus/v17_train.jsonl` | `bddde99a0f6c1c308266883238ac40af07203e6bddfdbb74c8daf42be9f9f69e` | 436301 | 558 |
| 2 | `evidence/leak_recheck_output.txt` | `722908ca4470b7ad655690558deda6a6521c4241e15fc9e256546153438291b7` | 849 | 14 |
| 3 | `evidence/v17_prepare_audit.json` | `115005c484d70d3563f053a641c8fe8d1c2fee03e3101d1e178e20cd8e68a63b` | 16502 | 423 |
| 4 | `evidence/val168_leak_evidence.json` | `3c06c21d8f1e8ee5bb669b59077627b9f5cfa174edc5b7c3a5af26190660dfbe` | 1504 | 53 |
| 5 | `report/WD_V17_GATES_20261020.md` | `e1e5d4b2b2d4b59da4d9cbe77981a2f217d7fa96c4226649752bad025b89a752` | 3842 | 41 |
| 6 | `report/v17_发布载体方案表_20261021.md` | `27f5292fa6afb994ca97d70944829bf111af1d0f38cc941ab0a0009e0f4faecb` | 10466 | 81 |
| 7 | `report/v17_对外终版报告_v1稿_20261021.md` | `b6da32c1d93b68a6285b79377d2b5127598d431660c580e79b5bb7a15f7492c4` | 13900 | 136 |
| 8 | `repro/dsh_r91_leak_check.manifest` | `67cbfc735fdd97291ae5cc2d6bb4e794ecc6f37b078f9ab7820cd85322a4853c` | 1491 | 16 |
| 9 | `repro/dsh_r91_leak_check.py` | `8dd57d668615bbcbf2f37ac4365fa690859f0fe9ae3d6ed3a33f9f7856c4f112` | 5240 | 120 |
| 10 | `repro/eval_basis63.py` | `3e720fcbbeb509466f34ede4a46537c80bdd0facc8434f42be968dad618fc598` | 5832 | 129 |
| 11 | `repro/eval_sets/basis63.json` | `5fea69fecbf2b368faac8203bcb6249729349882ee1732ad4deab2691b1b1457` | 32043 | 516 |
| 12 | `repro/eval_sets/fitcheck_bare79.jsonl` | `cb1fc9e0a91f7d27e5a03144abe5ef44c9fdffb432aefcd001dc1d6dfc152205` | 53644 | 79 |
| 13 | `repro/eval_sets/val168.jsonl` | `b1fcdd1ffb86424aaf323750d397182712343661cb4f8f1b701dad75fa939d8b` | 135185 | 168 |
| 14 | `repro/eval_v01.py` | `d75e4a27fa2a9ea771c539b8739796b5d77540622aa05b22437d039c80a1ee94` | 5718 | 141 |
| 15 | `repro/prepare_v17.py` | `8ca9b0e7548e73cb696c1e72f8de0be991933be608ef84971bf76d72e6ef6083` | 15322 | 275 |
| 16 | `repro/run_v17_gates.py` | `284852f44d9514fb20f49a8c87d092c826409b1df6f82db620207dc8b38108fc` | 14150 | 236 |
| 17 | `repro/train_v01.py` | `d3d72e4582e007b58cffa2d0decb2958ebe3424717325d67d9a16f0800ee770b` | 21352 | 403 |
| 18 | `repro/val168_leak_evidence.py` | `dbf26083783adac70bd303060736e6cbd2693f0e9481acd3eb3cd3b3dc7155d6` | 4562 | 114 |
| 19 | `repro/verify_v17_final.py` | `f4207f13465d9eac057a3edb879b2285a9c0bcd713a9d1cf8914927b7677f71c` | 6063 | 143 |
| 20 | `README.md` | `213bf6c3482315649734d17132c33dd0c197ce89818a7cdacb3a6894a196967d` | 2986 | 55 |

## 二、脱敏声明（D-15）

- 替换项：主机账户名（WD 侧与 DSH 侧）→ `C:/Users/<USER>`；
  共享区 → `<SHARED>`；凭据存放路径 → `<CREDENTIAL_STORE>`；
  总线端点 → `<BUS_HOST>`；隧道端点 → `<TUNNEL_HOST>`。
- 保留项（按 LH 裁定，改名即不可复现）：`BUS_TOKEN` / `HETU_TOKEN` / `GITHUB_TOKEN` 变量名。
- 内部标识扫描结论：主机账户名 / 真实端点 / 本机绝对路径三类 P0 经同正则复扫 **零命中**（见脱敏核验记录）。
- 内部台账（含原路径映射 + 原 sha↔脱敏 sha 双表）留存于 `v17_final_internal/_D15_INTERNAL_LEDGER.md`，不随 public 出库。
