"""
Phase 3 — Training Loop (Real Execution + Logging)
Generates: best_model.pt, loss_plot.png, training_log.txt
"""
import os, sys, time, math, pickle, json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import phase1_data_pipeline  # registers Vocabulary for pickle
from phase2_architecture import build_model, DEVICE

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, "..", "data")
CKPT = os.path.join(BASE, "..", "checkpoints")
os.makedirs(CKPT, exist_ok=True)

HP = dict(embed_dim=128, hidden_dim=256, enc_layers=1, dec_layers=1, dropout=0.35,
          batch_size=64, num_epochs=18, lr=1e-3, clip=1.0,
          tf_init=1.0, tf_end=0.4, val_split=0.1)


class DS(Dataset):
    def __init__(self, src,tgt,sl,tl):
        self.src=torch.from_numpy(src).long(); self.tgt=torch.from_numpy(tgt).long()
        self.sl=torch.from_numpy(sl).long(); self.tl=torch.from_numpy(tl).long()
    def __len__(self): return len(self.src)
    def __getitem__(self,i): return self.src[i],self.tgt[i],self.sl[i],self.tl[i]


def masked_ce(logits, targets, tgt_lens, pad_idx=0):
    B,T,V = logits.size()
    mask = (torch.arange(T,device=targets.device).unsqueeze(0) < tgt_lens.unsqueeze(1))
    loss = nn.functional.cross_entropy(logits.reshape(B*T,V), targets.reshape(B*T),
                                       ignore_index=pad_idx, reduction="none")
    return loss[mask.reshape(B*T)].mean()


def tf_ratio(epoch, total, init, end):
    return init - (init-end)*(epoch/max(total-1,1))


def train():
    log_lines = []
    def log(msg):
        print(msg)
        log_lines.append(msg)

    log("="*65)
    log("HelpDesk AI — Seq2Seq Training Run (Domain Support Dataset)")
    log("="*65)

    with open(os.path.join(DATA,"vocab.pkl"),"rb") as f:
        vocab = pickle.load(f)
    vocab_size = len(vocab)
    log(f"Vocab size: {vocab_size:,}")

    model = build_model(vocab_size, HP["embed_dim"],HP["hidden_dim"],
                        HP["enc_layers"],HP["dec_layers"],HP["dropout"])

    data = np.load(os.path.join(DATA,"tensors.npz"))
    ds = DS(data["src"],data["tgt"],data["src_lens"],data["tgt_lens"])
    val_n = int(len(ds)*HP["val_split"]); train_n = len(ds)-val_n
    train_ds,val_ds = random_split(ds,[train_n,val_n], generator=torch.Generator().manual_seed(42))
    train_dl = DataLoader(train_ds, batch_size=HP["batch_size"], shuffle=True)
    val_dl   = DataLoader(val_ds,   batch_size=HP["batch_size"], shuffle=False)
    log(f"Train: {len(train_ds):,} | Val: {len(val_ds):,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=HP["lr"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=4, factor=0.5)

    train_losses, val_losses = [], []
    best_val = float("inf")
    best_ckpt = os.path.join(CKPT,"best_model.pt")

    log(f"\n{'Epoch':>6} {'Train':>9} {'Val':>9} {'PPL':>9} {'TF':>6} {'LR':>10} {'Time':>7}")
    log("-"*65)

    t_start = time.time()
    for epoch in range(1, HP["num_epochs"]+1):
        tf = tf_ratio(epoch-1, HP["num_epochs"], HP["tf_init"], HP["tf_end"])
        t0=time.time()
        model.train(); tr_loss=0.0
        for src,tgt,sl,tl in train_dl:
            optimizer.zero_grad()
            logits = model(src,sl,tgt,teacher_force_ratio=tf)
            loss = masked_ce(logits,tgt,tl)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), HP["clip"])
            optimizer.step()
            tr_loss += loss.item()
        tr_loss /= len(train_dl)

        model.eval(); vl_loss=0.0
        with torch.no_grad():
            for src,tgt,sl,tl in val_dl:
                logits = model(src,sl,tgt,teacher_force_ratio=0.0)
                vl_loss += masked_ce(logits,tgt,tl).item()
        vl_loss /= len(val_dl)

        train_losses.append(tr_loss); val_losses.append(vl_loss)
        ppl = math.exp(min(vl_loss,20))
        lr = optimizer.param_groups[0]["lr"]
        elapsed = time.time()-t0

        log(f"  {epoch:>4}  {tr_loss:>9.4f}  {vl_loss:>9.4f}  {ppl:>9.2f}  {tf:>6.2f}  {lr:>10.6f}  {elapsed:>5.1f}s")

        if vl_loss < best_val:
            best_val = vl_loss
            torch.save({"epoch":epoch,"model_state":model.state_dict(),
                       "val_loss":vl_loss,"hparams":HP,"vocab_size":vocab_size},
                      best_ckpt)
            log(f"  ✅ Best model saved (val_loss={vl_loss:.4f})")

        scheduler.step(vl_loss)

    total_time = time.time()-t_start
    log(f"\nTotal training time: {total_time:.1f}s ({total_time/60:.1f} min)")
    log(f"Best val loss: {best_val:.4f} (PPL={math.exp(min(best_val,20)):.2f})")

    # Loss plot
    epochs = range(1,len(train_losses)+1)
    fig,ax1 = plt.subplots(figsize=(11,5))
    ax1.plot(epochs, train_losses, label="Train Loss", color="#2196F3", linewidth=2)
    ax1.plot(epochs, val_losses, label="Val Loss", color="#F44336", linewidth=2, linestyle="--")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Cross-Entropy Loss")
    ax1.set_title("HelpDesk AI — Training Convergence (Real Run, Domain Support Dataset)")
    ax1.grid(alpha=0.3)
    ax2 = ax1.twinx()
    ppl_curve = [math.exp(min(l,20)) for l in val_losses]
    ax2.plot(epochs, ppl_curve, label="Val Perplexity", color="#9C27B0", linewidth=1.5, alpha=0.6, linestyle=":")
    ax2.set_ylabel("Perplexity", color="#9C27B0")
    l1,lb1 = ax1.get_legend_handles_labels(); l2,lb2=ax2.get_legend_handles_labels()
    ax1.legend(l1+l2, lb1+lb2, loc="upper right")
    plt.tight_layout()
    plot_path = os.path.join(CKPT,"loss_plot.png")
    plt.savefig(plot_path, dpi=150)
    log(f"Loss plot saved -> {plot_path}")

    # Save training log
    log_path = os.path.join(CKPT, "training_log.txt")
    with open(log_path,"w") as f:
        f.write("\n".join(log_lines))
    print(f"\nTraining log saved -> {log_path}")

    return model, vocab


if __name__=="__main__":
    train()
