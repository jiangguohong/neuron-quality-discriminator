#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
判别器 v0.1 · CPU LoRA 训练（LH #9 派单 · 2026-10-15）
======================================================
基座: Qwen2.5-0.5B-Instruct（HF safetensors，CPU float32）
方法: peft LoRA（CPU 无 bitsandbytes，非量化；0.5B fp32≈2GB，16GB 机器可扛）
数据: v01_train.jsonl（prepare_v01.py 产出，messages 格式）
输出: adapters_v01/adapter_model.safetensors + adapter_config.json + train_log.json

训练侧不上 ROCm（PyTorch ROCm Windows 不稳，LH 明示）；推理评估另用 HIP 版 llama.cpp。

运行: python train_v01.py [--epochs 8] [--lr 1e-4] [--batch-size 4] [--max-len 512]
                       [--lora-r 16] [--threads 8] [--base-model <path>] [--data <dir>]
"""
import argparse
import base64
import datetime
import hashlib
import json
import os
import shutil
import time

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_BASE = r"D:\amd_hip_sdk\models\Qwen2.5-0.5B-Instruct"


def encode_example(tok, messages, max_len):
    """编码一条 messages；prompt 部分 labels=-100（只学 assistant 回复）。"""
    prompt_text = tok.apply_chat_template(
        messages[:-1], tokenize=False, add_generation_prompt=True
    )
    full_text = tok.apply_chat_template(messages, tokenize=False)
    prompt_ids = tok(prompt_text, add_special_tokens=False)["input_ids"]
    full_ids = tok(full_text, add_special_tokens=False, truncation=True, max_length=max_len)["input_ids"]
    if not full_ids:
        return None
    # full_ids 应以 prompt_ids 开头（chat template 一致）；截断保护
    n_prompt = len(prompt_ids)
    if n_prompt > len(full_ids) - 1:
        n_prompt = max(0, len(full_ids) // 2)
    labels = [-100] * n_prompt + full_ids[n_prompt:]
    return {"input_ids": full_ids, "labels": labels}


def _sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_ckpt_manifest(out_dir, *, run_tag, corpus_path, corpus_sha, corpus_lines, seed, hyper,
                        step_ckpts, epoch_ckpts, latest_step, extra=None):
    """LH R89 §4：ckpt 落盘即落 manifest（step/seed/超参/输入语料 sha/输出目录）。"""
    man = {
        "run_tag": run_tag,
        "out_dir": os.path.abspath(out_dir),
        "corpus_path": os.path.abspath(corpus_path),
        "corpus_sha256": corpus_sha,
        "corpus_lines": corpus_lines,
        "seed": seed,
        "hyper": hyper,
        "updated_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "step_ckpts": step_ckpts,
        "epoch_ckpts": epoch_ckpts,
        "latest_step": latest_step,
    }
    if extra:
        man.update(extra)
    p = os.path.join(out_dir, "ckpt_manifest.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(man, f, ensure_ascii=False, indent=2)
    return p


def sync_ckpt_to_share(share_dir, manifest_path, step_ckpt_dir=None):
    """manifest（+最新 step ckpt）同步共享区，DSH 可独立核「有无 ckpt」。失败不拖垮训练。"""
    if not share_dir:
        return
    try:
        os.makedirs(share_dir, exist_ok=True)
        shutil.copy2(manifest_path, os.path.join(share_dir, "ckpt_manifest.json"))
        if step_ckpt_dir and os.path.isdir(step_ckpt_dir):
            dst = os.path.join(share_dir, os.path.basename(step_ckpt_dir))
            shutil.copytree(step_ckpt_dir, dst, dirs_exist_ok=True)
        print(f"  [ckpt] manifest 已同步共享区 {share_dir}")
    except Exception as e:
        print(f"  [ckpt] ⚠ 共享区同步失败（不影响训练）: {e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=HERE)
    ap.add_argument("--base-model", default=DEFAULT_BASE)
    ap.add_argument("--out", default=os.path.join(HERE, "adapters_v01"))
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--class-weight", default="auto",
                    help="auto=按 keep/reject 样本数反比加权(上限10,治多数类塌缩); off=不加权")
    ap.add_argument("--resume", action="store_true",
                    help="从 <out>/ckpt_epochN 或 ckpt_stepM 恢复（取全局 step 最大者；LH P0/R89§4）")
    ap.add_argument("--ckpt-every", type=int, default=50,
                    help="每 N step 落 step 级 ckpt+manifest（LH R89 §4 硬派令；0=关，仅 epoch 末 ckpt）")
    ap.add_argument("--ckpt-share", default=None,
                    help="manifest+最新 step ckpt 的共享区同步目录（DSH 可读域，如 D:/<SHARED>/v17_train_ckpt）")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)

    train_path = os.path.join(args.data, "v01_train.jsonl")
    if not os.path.exists(train_path):
        print(f"✗ 缺训练数据 {train_path}，先跑 prepare_v01.py")
        raise SystemExit(1)
    samples = []
    with open(train_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line)["messages"])
    print(f"训练样本: {len(samples)} 条 | epochs={args.epochs} bs={args.batch_size} lr={args.lr} r={args.lora_r}")
    # 输入语料 sha（manifest 必备字段，LH R89 §4）
    corpus_sha = hashlib.sha256(open(train_path, "rb").read()).hexdigest()
    print(f"语料 sha256: {corpus_sha[:16]}…（{train_path}）")

    tok = AutoTokenizer.from_pretrained(args.base_model)
    model = AutoModelForCausalLM.from_pretrained(args.base_model, torch_dtype=torch.float32)
    model.config.use_cache = False  # 训练关 cache

    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora_cfg)
    model.print_parameter_summary() if hasattr(model, "print_parameter_summary") else None
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"可训练参数: {trainable/1e6:.2f}M / 总 {total/1e6:.1f}M ({100*trainable/total:.2f}%)")

    # ---- resume：加载全局 step 最大的 ckpt（epoch 级 or step 级），精确续跑 ----
    # LH P0（崩溃不白跑）+ R89 §4（≤50 step 定期 ckpt；step 级支持 epoch 内精确续跑）
    start_epoch = 0
    resume_step = 0            # 全局 step 连续性（lr warmup/cosine 对齐）
    resume_batch_idx = 0       # epoch 内续跑批量偏移（仅 step 级 ckpt；epoch 级=0 重跑整 epoch）
    if args.resume:
        spe_est = (len(samples) + args.batch_size - 1) // args.batch_size
        cands = []  # (abs_step, kind, name)
        if os.path.isdir(args.out):
            for name in os.listdir(args.out):
                if name.startswith("ckpt_epoch"):
                    try:
                        n = int(name[len("ckpt_epoch"):])
                        cands.append((n * spe_est, "epoch", name))
                    except ValueError:
                        pass
                elif name.startswith("ckpt_step"):
                    try:
                        cands.append((int(name[len("ckpt_step"):]), "step", name))
                    except ValueError:
                        pass
        if cands:
            cands.sort()
            abs_step, kind, ck_name = cands[-1]
            ck_dir = os.path.join(args.out, ck_name)
            print(f"[resume] 发现 checkpoint {ck_name}（全局 step {abs_step}，{kind} 级）")
            # 原地载入（P033 教训链，三试方得正解）：
            # ① PeftModel.from_pretrained(model, ck_dir)——model 已被 get_peft_model 包装，
            #    再包一层嵌套且全参数冻冻（backward: does not require grad）
            # ② 裸 safetensors + load_state_dict——save_pretrained 剥掉适配器名
            #    （lora_A.default.weight→lora_A.weight），336 张量全 unexpected 静默不载入
            # ③ 正解：set_peft_model_state_dict = get_peft_model_state_dict 的官方逆操作，
            #    内部负责回插适配器名；原地写入保留参数对象身份（optimizer 有效）与 grad 标志
            from peft.utils.save_and_load import load_peft_weights, set_peft_model_state_dict
            sd = load_peft_weights(ck_dir)
            res = set_peft_model_state_dict(model, sd, adapter_name="default")
            n_lora = sum(1 for k in sd if "lora_" in k)
            missing, unexpected = list(res.missing_keys), list(res.unexpected_keys)
            print(f"[resume] 载入 LoRA 张量 {n_lora} 个 | missing={len(missing)} unexpected={len(unexpected)}")
            if n_lora == 0 or len(unexpected) > 0:
                print(f"[resume] ✗ 载入异常，拒绝开跑: unexpected={unexpected[:3]}")
                sys.exit(3)
            # 双保险：显式恢复 LoRA 参数可训
            n_train = 0
            for n, p in model.named_parameters():
                if "lora_" in n:
                    p.requires_grad = True
                    n_train += p.numel()
            print(f"[resume] 可训 LoRA 参数 {n_train/1e6:.2f}M 已复启用")
            resume_step = abs_step
            if kind == "epoch":
                start_epoch = abs_step // spe_est
                resume_batch_idx = 0
            else:
                start_epoch = (abs_step - 1) // spe_est
                resume_batch_idx = abs_step - start_epoch * spe_est
            print(f"[resume] 续跑位置: epoch {start_epoch+1}/{args.epochs} 内第 {resume_batch_idx} 批之后（全局 step {resume_step}）")
            # RNG 顺序复现：恢复该 epoch 开训前 RNG 态，使 randperm 数据顺序与中断前一致（R89 §3 禁改数据顺序）
            rng_file = os.path.join(args.out, "rng_epoch_state.json")
            if os.path.exists(rng_file):
                try:
                    st = json.load(open(rng_file, encoding="utf-8"))
                    if st.get("epoch") == start_epoch:
                        torch.set_rng_state(torch.tensor(
                            list(base64.b64decode(st["state_b64"])), dtype=torch.uint8))
                        print(f"[resume] RNG 态已恢复至 epoch {start_epoch+1} 开训前（数据顺序复现）")
                except Exception as e:
                    print(f"[resume] RNG 态恢复失败（继续跑，数据顺序将重洗）: {e}")
        else:
            print("[resume] 无 checkpoint，从头训练")

    encoded = [e for e in (encode_example(tok, m, args.max_len) for m in samples) if e]
    print(f"编码完成: {len(encoded)} 条")

    # ---- 类别加权（治少数类塌缩）----
    # v0.1 冒烟教训：7 keep vs 1 reject 时 LoRA 对一切输入判 keep。
    # v2（2026-10-17，v0.6/v0.7 回归复盘）：auto 对称化——原版只在 keep 为多数时给 reject 上权，
    #   但 733 语料 keep 仅 41%（NP300 内部 33%），keep 梯度被稀释，val keep 崩塌至 38-52%
    #   （v0.7 epochs=2 46/63 → v0.6 epochs=4 52/63，单调上升=欠拟合而非记忆化）。
    #   对称规则：少数类上权，权重 = 多数/少数，上限 ×10，按 token 加权 CE。
    sample_w = [1.0] * len(encoded)
    if args.class_weight == "auto":
        n_rej = sum(1 for s in samples if "判定：reject" in s[-1]["content"])
        n_keep = len(samples) - n_rej
        if n_rej > 0 and n_keep > 0 and n_keep != n_rej:
            if n_keep > n_rej:
                w, minor_is_rej = min(10.0, n_keep / n_rej), True
            else:
                w, minor_is_rej = min(10.0, n_rej / n_keep), False
            for i, s in enumerate(samples):
                if ("判定：reject" in s[-1]["content"]) == minor_is_rej:
                    sample_w[i] = w
            minor = "reject" if minor_is_rej else "keep"
            print(f"类别加权: {minor}(少数类) 权重 ×{w:.2f}（keep {n_keep} / reject {n_rej}）")
        else:
            print("类别加权: 无需（类别均衡或无 reject）")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    steps_per_epoch = (len(encoded) + args.batch_size - 1) // args.batch_size
    total_steps = steps_per_epoch * args.epochs
    warmup = max(1, int(0.1 * total_steps))

    def lr_at(step):
        if step < warmup:
            return args.lr * (step + 1) / warmup
        import math
        p = (step - warmup) / max(1, total_steps - warmup)
        return args.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * p)))

    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    log = []
    t0 = time.time()
    step = resume_step  # resume 时 lr schedule 连续（warmup/cosine 对齐全局 step）
    os.makedirs(args.out, exist_ok=True)  # rng 态/ckpt 落盘前置建目录（原位置在循环后会崩）
    model.train()
    manifest_hyper = {"epochs": args.epochs, "lr": args.lr, "batch_size": args.batch_size,
                      "lora_r": args.lora_r, "lora_alpha": args.lora_alpha,
                      "threads": args.threads, "class_weight": args.class_weight,
                      "max_len": args.max_len, "ckpt_every": args.ckpt_every}
    step_ckpt_log, epoch_ckpt_log = [], []
    for epoch in range(start_epoch, args.epochs):
        # RNG 开训前态落盘（resume 时复现本 epoch 数据顺序，R89 §3 禁改数据顺序）
        rng_before = torch.get_rng_state()
        with open(os.path.join(args.out, "rng_epoch_state.json"), "w", encoding="utf-8") as f:
            json.dump({"epoch": epoch,
                       "state_b64": base64.b64encode(rng_before.numpy().tobytes()).decode()}, f)
        order = torch.randperm(len(encoded)).tolist()
        ep_loss, ep_n = 0.0, 0
        for bi, i in enumerate(range(0, len(encoded), args.batch_size)):
            if epoch == start_epoch and bi < resume_batch_idx:
                continue  # step 级续跑：跳过本 epoch 已训批次（精确续跑，不重训不双算）
            batch_idx = order[i:i + args.batch_size]
            maxl = max(len(encoded[j]["input_ids"]) for j in batch_idx)
            input_ids, labels, attn, wts = [], [], [], []
            for j in batch_idx:
                ids = encoded[j]["input_ids"]
                lab = encoded[j]["labels"]
                pad = maxl - len(ids)
                input_ids.append(ids + [pad_id] * pad)
                labels.append(lab + [-100] * pad)
                attn.append([1] * len(ids) + [0] * pad)
                wts.append([sample_w[j]] * len(ids) + [0.0] * pad)
            input_ids_t = torch.tensor(input_ids)
            labels_t = torch.tensor(labels)
            wts_t = torch.tensor(wts)
            for g in opt.param_groups:
                g["lr"] = lr_at(step)
            out = model(input_ids=input_ids_t, attention_mask=torch.tensor(attn))
            # 按 token 加权 CE（-100 处权重置 0，ignore_index 兜底）
            shift_logits = out.logits[:, :-1, :].contiguous()
            shift_labels = labels_t[:, 1:].contiguous()
            shift_w = wts_t[:, 1:].contiguous()
            tok_loss = torch.nn.functional.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                reduction="none",
                ignore_index=-100,
            ).view(shift_labels.size())
            denom = (shift_w * (shift_labels != -100)).sum()
            loss = (tok_loss * shift_w).sum() / denom.clamp(min=1.0)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            ep_loss += float(loss.detach())
            ep_n += 1
            step += 1
            if step % 5 == 0 or step == total_steps:
                print(f"  step {step}/{total_steps}  loss={float(loss.detach()):.4f}  lr={lr_at(step-1):.2e}  elapsed={time.time()-t0:.0f}s")
            # ---- step 级定期 ckpt（LH R89 §4：间隔 ≤50 step，落盘即落 manifest）----
            if args.ckpt_every and step % args.ckpt_every == 0:
                sc = os.path.join(args.out, f"ckpt_step{step}")
                os.makedirs(sc, exist_ok=True)
                model.save_pretrained(sc)
                adapter_sha = _sha256_file(os.path.join(sc, "adapter_model.safetensors")) \
                    if os.path.exists(os.path.join(sc, "adapter_model.safetensors")) else None
                step_ckpt_log.append({"step": step, "dir": sc, "adapter_sha256": adapter_sha,
                                      "created_at": datetime.datetime.now().isoformat(timespec="seconds")})
                mp = write_ckpt_manifest(args.out, run_tag=os.path.basename(args.out),
                                         corpus_path=train_path, corpus_sha=corpus_sha,
                                         corpus_lines=len(samples), seed=args.seed,
                                         hyper=manifest_hyper, step_ckpts=step_ckpt_log,
                                         epoch_ckpts=epoch_ckpt_log, latest_step=step)
                sync_ckpt_to_share(args.ckpt_share, mp, step_ckpt_dir=sc)
                print(f"  [ckpt] step{step} → {sc}")
        avg = ep_loss / max(1, ep_n)
        log.append({"epoch": epoch + 1, "avg_loss": round(avg, 4)})
        print(f"epoch {epoch+1}/{args.epochs}  avg_loss={avg:.4f}")
        # ---- epoch 末 checkpoint（LH P0：崩溃最多丢 1 个 epoch，不白跑全程）----
        # 粒度说明：只存 adapter（~35MB），不含 optimizer moments——epoch 末重训该 epoch
        # 数据 moments 几步即重建，可接受；step 级 checkpoint 留 v1.1 前评估
        ck_dir = os.path.join(args.out, f"ckpt_epoch{epoch+1}")
        os.makedirs(ck_dir, exist_ok=True)
        model.save_pretrained(ck_dir)
        with open(os.path.join(ck_dir, "ckpt_meta.json"), "w", encoding="utf-8") as f:
            json.dump({"epoch": epoch + 1, "epochs_total": args.epochs,
                       "avg_loss": round(avg, 4), "wall_time_s": round(time.time() - t0, 1),
                       "step": step, "seed": args.seed,
                       "note": "epoch 末 adapter 快照；--resume 从此 epoch 后继续"},
                      f, ensure_ascii=False, indent=2)
        epoch_ckpt_log.append({"epoch": epoch + 1, "dir": ck_dir, "step": step,
                               "avg_loss": round(avg, 4),
                               "created_at": datetime.datetime.now().isoformat(timespec="seconds")})
        mp = write_ckpt_manifest(args.out, run_tag=os.path.basename(args.out),
                                 corpus_path=train_path, corpus_sha=corpus_sha,
                                 corpus_lines=len(samples), seed=args.seed,
                                 hyper=manifest_hyper, step_ckpts=step_ckpt_log,
                                 epoch_ckpts=epoch_ckpt_log, latest_step=step)
        sync_ckpt_to_share(args.ckpt_share, mp)
        print(f"  [ckpt] epoch{epoch+1} → {ck_dir}")

    os.makedirs(args.out, exist_ok=True)
    model.save_pretrained(args.out)
    tok.save_pretrained(args.out)
    meta = {
        "version": "v0.1",
        "base_model": args.base_model,
        "samples": len(samples),
        "epochs": args.epochs,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "trainable_params": trainable,
        "total_params": total,
        "epoch_log": log,
        "wall_time_s": round(time.time() - t0, 1),
        "seed": args.seed,
        "class_weight": args.class_weight,
        "corpus_path": os.path.abspath(train_path),
        "corpus_sha256": corpus_sha,
        "sample_weights": {"minority_x": max(sample_w) if sample_w else 1.0},
    }
    with open(os.path.join(args.out, "train_log.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    mp = write_ckpt_manifest(args.out, run_tag=os.path.basename(args.out),
                             corpus_path=train_path, corpus_sha=corpus_sha,
                             corpus_lines=len(samples), seed=args.seed,
                             hyper=manifest_hyper, step_ckpts=step_ckpt_log,
                             epoch_ckpts=epoch_ckpt_log, latest_step=step,
                             extra={"final": True, "wall_time_s": round(time.time() - t0, 1)})
    sync_ckpt_to_share(args.ckpt_share, mp)
    print(f"\n===== TRAIN_DONE ({time.time()-t0:.0f}s) → {args.out} =====")


if __name__ == "__main__":
    main()
