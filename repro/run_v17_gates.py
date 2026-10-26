#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_v17_gates.py — v17 三闸判读 + 三档泄漏复核编排（LH R86 L1~L5 / R87 §3 口径）

与 v16 版差异：
  1. ADAPTER → adapters_v17/ckpt_epoch5（base 冷起动产物，无 resume 链）
  2. 判读口径换轨（LH R87 §3）：裸闸分数**低于** v11 记忆主导水平
     （basis63 55/63、fitcheck 79/79、val168 120/168）= **预期结果**（学泛化非背题）；
     裸闸**异常高逼近记忆水平** = 泄漏信号 → 立即停并回报，不当成功推进
  3. **终版判据 = 三档泄漏复核 PASS**（basis63/fitcheck_bare/val168 对训练语料，
     文本+pid+组合语义三档全 0）；任一档 >0 → 回退重装配
  4. 不再做 v15/v16 跨版口径锁定校验（L3：裸闸唯一口径，跨闸比较禁止）
  5. fitcheck79 回归对照保留历史数（v13/v14/v15/v16），但只作参照不作闸
铁律：严禁总线发信/ack、严禁改注册表；仅落盘报告，由本人读后发 LH。
"""
import os
import sys
import time
import json
import shutil
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
PY = r"C:\Users\<USER>\.workbuddy\binaries\python\versions\3.13.12\python.exe"
BASE = r"D:\amd_hip_sdk\models\Qwen2.5-0.5B-Instruct"
ADAPTER = os.path.join(HERE, "adapters_v17", "ckpt_epoch5")
CKPT_MARKER = os.path.join(ADAPTER, "adapter_model.safetensors")

# v11 记忆主导水平（LH R87 §3 参照系；其语料含评测项 verbatim，分数=记忆）
MEM_BASIS63 = 55      # /63
MEM_FITCHECK = 79     # /79 = 100%
MEM_VAL168 = 120      # /168（文本级 verbatim 命中数）
LEAK_SUSPECT_FITCHECK = 0.90   # 异常高判据（v11=100%；泄漏期 v13/v14 也仅 24~77%）

LOG_PATH = os.path.join(HERE, "run_v17_gates.log")


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def wait_ckpt(timeout_s=18000):
    """等 ckpt_epoch5；v17 base 冷启跑满 5 epoch（1395 步 × ~10s/步 ≈ 3.9h），
    故窗口按 5h 设（历史速率实证：v15 10.2s/步、v16 8s/步，均只训 1 epoch）"""
    log(f"等待 ckpt 落盘：{CKPT_MARKER}")
    deadline = time.time() + timeout_s
    while not os.path.exists(CKPT_MARKER):
        if time.time() > deadline:
            raise TimeoutError(f"超时 {timeout_s}s 仍未产出 ckpt_epoch5")
        time.sleep(20)
    log("ckpt_epoch5 已产出，开始三闸判读")


def env_safe():
    e = os.environ.copy()
    e["OMP_NUM_THREADS"] = "2"; e["MKL_NUM_THREADS"] = "2"; e["OPENBLAS_NUM_THREADS"] = "2"
    e["MKL_DYNAMIC"] = "FALSE"; e["OMP_DYNAMIC"] = "FALSE"
    # Windows 管道下 Python 默认用 locale（GBK）编码 stdout，eval 脚本打印 ✓/✗
    # 即 UnicodeEncodeError 崩（11:11 实证）；PYTHONUTF8=1 强制 UTF-8 mode 根治
    e["PYTHONUTF8"] = "1"; e["PYTHONIOENCODING"] = "utf-8"
    return e


def run_eval(args, cwd=HERE):
    log(">>> " + " ".join(args))
    # SYSTEM 计划任务下 capture_output 管道静默失效（2026-10-21 三次实证：子进程
    # rc=0 但管道全空；用户账户下同一调用输出完整）——改进程级文件重定向保可观测性
    os.makedirs(os.path.join(HERE, "v17"), exist_ok=True)
    tmp = os.path.join(HERE, "v17", "_subproc_out.txt")
    with open(tmp, "w", encoding="utf-8") as fo:
        r = subprocess.run(args, cwd=cwd, env=env_safe(), stdout=fo, stderr=subprocess.STDOUT)
    out = open(tmp, encoding="utf-8").read()
    for ln in out.splitlines():
        if ln.strip():
            log("   " + ln)
    if r.returncode != 0:
        log("!!! 子进程非零退出 " + str(r.returncode))
        for ln in out.splitlines()[-20:]:
            log("   ERR " + ln)
        # 韧性：失败即抛，禁带病继续（原版 gate_a 失败后 json.load 抛
        # FileNotFoundError 掩盖真因，gates 静默死亡无判读产出）
        tail = " | ".join(out.splitlines()[-3:])
        raise RuntimeError(f"子进程失败 exit={r.returncode}: {' '.join(args[:2])} :: {tail}")
    return r


def gate_a():
    out = os.path.join(HERE, "v17", "v17_eval_basis63.json")
    os.makedirs(os.path.join(HERE, "v17"), exist_ok=True)
    run_eval([PY, "eval_basis63.py", "--adapter", ADAPTER,
              "--base-model", BASE, "--out", out, "--threads", "2"])
    rep = json.load(open(out, encoding="utf-8"))
    return {"n": rep["n_val"], "n_ok": rep["n_correct"], "acc": rep["accuracy"],
            "parse_fail": rep.get("parse_fail", 0),
            "mem_level": MEM_BASIS63, "mem_acc": MEM_BASIS63 / 63}


def gate_b_and_c():
    val_src = os.path.join(HERE, "v01_val.jsonl")
    val_dir = os.path.join(HERE, "v17", "val168")
    os.makedirs(val_dir, exist_ok=True)
    shutil.copy(val_src, os.path.join(val_dir, "v01_val.jsonl"))
    run_eval([PY, "eval_v01.py", "--adapter", ADAPTER,
              "--data", val_dir, "--base-model", BASE, "--threads", "2"])
    val_rep = json.load(open(os.path.join(val_dir, "v01_eval_report.json"), encoding="utf-8"))

    fc_src = os.path.join(HERE, "_fitcheck", "v01_val.jsonl")
    fc_dir = os.path.join(HERE, "v17", "fitcheck79")
    os.makedirs(fc_dir, exist_ok=True)
    shutil.copy(fc_src, os.path.join(fc_dir, "v01_val.jsonl"))
    run_eval([PY, "eval_v01.py", "--adapter", ADAPTER,
              "--data", fc_dir, "--base-model", BASE, "--threads", "2"])
    fc_rep = json.load(open(os.path.join(fc_dir, "v01_eval_report.json"), encoding="utf-8"))

    rows = val_rep["rows"]
    rej = [r for r in rows if r["gold"] == "reject"]
    rej_ok = sum(1 for r in rej if r["pred"] == "reject")
    rej_recall = (rej_ok / len(rej)) if rej else 0.0
    keep = [r for r in rows if r["gold"] == "keep"]
    keep_ok = sum(1 for r in keep if r["pred"] == "keep")
    keep_recall = (keep_ok / len(keep)) if keep else 0.0
    b = {"n_val": val_rep["n_val"], "reject_n": len(rej), "reject_ok": rej_ok,
         "reject_recall": rej_recall, "keep_n": len(keep), "keep_ok": keep_ok,
         "keep_recall": keep_recall, "parse_fail": val_rep.get("parse_fail", 0),
         "mem_text_hits": MEM_VAL168}
    c = {"n": fc_rep["n_val"], "n_ok": fc_rep["n_correct"], "acc": fc_rep["accuracy"],
         "parse_fail": fc_rep.get("parse_fail", 0), "mem_level": MEM_FITCHECK,
         "mem_acc": 1.0, "leak_suspect_threshold": LEAK_SUSPECT_FITCHECK,
         "leak_suspect": fc_rep["accuracy"] >= LEAK_SUSPECT_FITCHECK}
    return b, c


def leak_recheck():
    """终版判据：三档泄漏复核（verify_v17_final.py 同口径，对训练语料重跑）

    实现说明（SYSTEM 计划任务环境实测教训，2026-10-21）：
    管道捕获（capture_output=True）在 SYSTEM 账户计划任务下**静默失效**——
    子进程 rc=0 但 stdout/stderr 全空（12:07 gates / 17:08 gates / 17:17 diag 三次实证；
    同一调用在用户账户 bash/PowerShell 下输出完整 534B）。故改为**进程级文件重定向**
    （subprocess stdout=文件句柄 + -u 无缓冲），父进程再读文件。
    判定仍以 returncode 为准；落盘文件为 LH R91 §3 要求的显式凭据。
    """
    ev_dir = os.path.join(HERE, "v17")
    os.makedirs(ev_dir, exist_ok=True)
    ev_path = os.path.join(ev_dir, "leak_recheck_output.txt")
    with open(ev_path, "w", encoding="utf-8") as fo:
        fo.write(f"# verify_v17_final.py 输出（run_v17_gates leak_recheck 进程级重定向落盘）\n")
        fo.write(f"# 时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        fo.flush()
        r = subprocess.run([PY, "-u", "verify_v17_final.py"], cwd=HERE, env=env_safe(),
                           stdout=fo, stderr=subprocess.STDOUT)
    out = open(ev_path, encoding="utf-8").read()
    log("--- 三档泄漏复核对输出（已显式落盘: v17/leak_recheck_output.txt）---")
    for ln in out.splitlines():
        if ln.strip():
            log("   " + ln)
    return {"returncode": r.returncode, "pass": r.returncode == 0, "output": out,
            "evidence_file": ev_path}


def write_report(a, b, c, lk):
    suspect = c["leak_suspect"]
    if not lk["pass"]:
        verdict = "❌ 三档泄漏复核 FAIL → 按 LH R87 §3 回退重装配，立即回报 LH"
    elif suspect:
        verdict = (f"⚠️ fitcheck79={c['acc']*100:.1f}% 异常高（≥{LEAK_SUSPECT_FITCHECK*100:.0f}% 判据）"
                   "→ 泄漏信号嫌疑，停回报 LH，禁当成功推进")
    else:
        verdict = ("✅ 三档泄漏复核 PASS；裸闸数按 LH R87 §3 记入诊断参照"
                   "（低于 v11 记忆水平=预期，非失败）→ 报 LH 复核后发终版确认")
    L = []
    L.append("# WD→LH：v17 三闸判读 + 三档泄漏复核（LH R86 L1~L5 / R87 §3 口径）")
    L.append("")
    L.append(f"> 生成时间：{time.strftime('%Y-%m-%d %H:%M:%S')} ｜ 判别器：base 冷起动 + LoRA(adapters_v17/ckpt_epoch5)")
    L.append("> 配方：v17 语料 558 条（keep 250 / reject 308 = 55.2%）+ class_weight 显式 1.0（主臂唯一计分）+ v15 实证防崩配置")
    L.append("> 起步：base 冷起动（不 resume 任何历史 adapter，LH R86 L2）")
    L.append("> 判读口径（LH R87 §3）：裸闸低于 v11 记忆主导水平 = 预期；异常高逼近记忆水平 = 泄漏信号；终版判据 = 三档泄漏复核 PASS")
    L.append("")
    L.append("## 一、三闸结果（诊断参照，非通过性判据）")
    L.append("")
    L.append(f"- **闸 A basis63**：{a['n_ok']}/{a['n']} = {a['acc']*100:.1f}% ｜ v11 记忆水平 {a['mem_level']}/63 = {a['mem_acc']*100:.1f}% ｜ 解析失败 {a['parse_fail']}")
    L.append(f"- **闸 B val168**：reject 召回 {b['reject_ok']}/{b['reject_n']} = {b['reject_recall']*100:.1f}% ｜ keep 召回 {b['keep_ok']}/{b['keep_n']} = {b['keep_recall']*100:.1f}% ｜ v11 文本级记忆命中 {b['mem_text_hits']}/168 ｜ 解析失败 {b['parse_fail']}")
    L.append(f"- **闸 C fitcheck79**：{c['n_ok']}/{c['n']} = {c['acc']*100:.1f}% ｜ v11 记忆水平 {c['mem_level']}/79 = 100% ｜ 异常高判据 ≥{c['leak_suspect_threshold']*100:.0f}% → {'⚠️ 触发' if c['leak_suspect'] else '未触发'} ｜ 解析失败 {c['parse_fail']}")
    L.append("")
    L.append("## 二、三档泄漏复核（终版判据，LH R87 §3）")
    L.append("")
    L.append(f"- 对 v17 训练语料重跑 basis63/fitcheck_bare/val168 × 文本/pid/组合语义三档：{'**PASS（全 0）**' if lk['pass'] else '**FAIL**'}")
    L.append(f"- 复验脚本退出码：{lk['returncode']}；一手输出凭据文件：`{os.path.relpath(lk.get('evidence_file', ''), HERE) if lk.get('evidence_file') else '（本版未落盘）'}`（LH R91 §3：修好前该段禁作唯一凭据，落盘后以该文件为凭据）")
    L.append("")
    L.append("## 三、训练中断披露（LH R89 §2 硬要求：如实披露，禁隐去）")
    L.append("")
    L.append("- **首次 run（已作废）**：10-20 22:54:54 起跑 → **23:10:29 机器人工干净关机**（System 事件日志坐实：EventLog 6006 + Kernel-Power 109 + Kernel-General 13）→ 训练进程团灭于 **step 115/1395**（elapsed 907s，loss 已降至 ~1.7）")
    L.append("- **无 ckpt 致进度作废**：旧版 train_v01.py 仅在 epoch 末存 ckpt，团灭时 epoch 1 未完结 → step 115 的算力全部作废，现场存档 `train_v17_aborted_2.log`，`adapters_v17` 无残留")
    L.append("- **当前 run（唯一计分 run）**：10-21 07:56:56 同配置重启（base 冷起动 + cw 显式 1.0 + v17 语料 558 + seed 42 + 同超参，LH R89 §3 硬约束）；**重启前输入语料 sha 复核 = `fed6726ecf0f05ea…`，与终版凭据逐字节一致（语料未漂移）**")
    L.append("- **定性（LH R89 §2 拍板）**：人工关机导致的中断，**不是泄漏停机，也不是技术失败**；本披露段为记录纪律")
    L.append("- **ckpt 补丁（LH R89 §4，已落地）**：train_v01.py 增加 `--ckpt-every 50`（step 级定期 ckpt + manifest：step/seed/超参/输入语料 sha/输出目录）+ `--ckpt-share D:/<SHARED>/v17_train_ckpt`（DSH 可读域同步）+ step 级精确续跑（epoch 内批次偏移 + RNG 顺序复现）；当前 run 为旧代码起跑（07:56 进程已加载旧版，无法热插），epoch 末 ckpt 仍生效 → **最坏再中断损失 ≤ 1 epoch（279 步 ≈ 40 分钟算力）**；若再中断，续跑经 `V17_RESUME=1` 环境变量走新机制")
    L.append("- **本次 run ckpt 落点**：`adapters_v17/`（epoch 末 ckpt + 终版 adapter）；共享区副本（补丁生效后的 run）：`D:/<SHARED>/v17_train_ckpt/`（manifest + 最新 step ckpt）")
    L.append("")
    L.append("## 四、并列裁决")
    L.append("")
    L.append(f"**结论：{verdict}**")
    L.append("")
    L.append("## 五、fitcheck79 历史参照（同批 79 条；v13/v14 为泄漏期数，仅参照）")
    L.append("")
    L.append("- v11：79/79 = 100%（记忆主导，语料含评测项 verbatim）")
    L.append("- v13：19/79 = 24.1% ｜ v14：61/79 = 77.2%（均带头版格式，与裸闸不可比）")
    L.append("- v15：0/79 = 0.0%（配方自伤）｜ v16：10/79 = 12.7%（双重加权无效）")
    L.append(f"- v17：{c['n_ok']}/79 = {c['acc']*100:.1f}%（base 冷起动 + cw1.0 + 干净语料）")
    L.append("")
    L.append("## 六、发 LH 状态（铁律）")
    L.append("")
    L.append("⚠️ 本报告**仅落盘，未发送任何总线信号**。上级核验后由本人发 LH。")
    out = os.path.join(HERE, "WD_V17_GATES_20261020.md")
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    log("报告落盘: " + out)
    return verdict


def main():
    wait_ckpt()
    a = gate_a()
    log(f"闸 A basis63: {a['n_ok']}/{a['n']} = {a['acc']*100:.1f}%")
    b, c = gate_b_and_c()
    log(f"闸 B val168: reject 召回 {b['reject_recall']*100:.1f}% ｜ 闸 C fitcheck79: {c['n_ok']}/{c['n']} = {c['acc']*100:.1f}%")
    lk = leak_recheck()
    verdict = write_report(a, b, c, lk)
    log("判读完成: " + verdict)


if __name__ == "__main__":
    main()
