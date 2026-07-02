"""
Phase 4 — Inference (Greedy + Beam Search)
"""
import os, sys, pickle
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import phase1_data_pipeline
from phase1_data_pipeline import clean, MAX_LEN, SOS, EOS, PAD, UNK
from phase2_architecture import build_model, DEVICE

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, "..", "data")
CKPT = os.path.join(BASE, "..", "checkpoints")


def load_model(ckpt_path=None):
    ckpt_path = ckpt_path or os.path.join(CKPT, "best_model.pt")
    with open(os.path.join(DATA, "vocab.pkl"), "rb") as f:
        vocab = pickle.load(f)
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    hp = ckpt["hparams"]
    model = build_model(len(vocab), hp["embed_dim"], hp["hidden_dim"],
                        hp["enc_layers"], hp["dec_layers"], 0.0)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"Loaded checkpoint epoch {ckpt['epoch']} (val_loss={ckpt['val_loss']:.4f})")
    return model, vocab


def encode_input(text, vocab, max_len=MAX_LEN):
    cleaned = clean(text)
    ids = vocab.encode(cleaned)
    ids = ids[:max_len+1]
    true_len = len(ids)
    pad_idx = vocab.word2idx[PAD]
    ids = ids + [pad_idx]*max(0, max_len+1-true_len)
    src = torch.tensor([ids], dtype=torch.long)
    src_lens = torch.tensor([true_len], dtype=torch.long)
    return src, src_lens


@torch.no_grad()
def beam_search(model, src, src_lens, vocab, beam_width=5, max_out=MAX_LEN, length_penalty=0.6):
    sos_idx, eos_idx, pad_idx = vocab.word2idx[SOS], vocab.word2idx[EOS], vocab.word2idx[PAD]
    enc_outputs, enc_hidden = model.encoder(src, src_lens)
    init_hidden = enc_hidden.unsqueeze(0).repeat(model.decoder.n_layers,1,1)
    init_tok = torch.tensor([sos_idx], dtype=torch.long)
    beams = [(0.0, [], init_tok, init_hidden)]
    completed = []
    for _ in range(max_out):
        new_beams = []
        for score, seq, token, hidden in beams:
            logits, new_hidden, _ = model.decoder.forward_step(token, hidden, enc_outputs)
            log_probs = torch.log_softmax(logits, dim=-1).squeeze(0)
            top_probs, top_ids = log_probs.topk(beam_width)
            for log_p, tok_id in zip(top_probs.tolist(), top_ids.tolist()):
                new_score = score - log_p
                new_seq = seq + [tok_id]
                if tok_id == eos_idx:
                    lp = ((5+len(new_seq))/6)**length_penalty
                    completed.append((new_score/lp, new_seq[:-1]))
                else:
                    new_beams.append((new_score, new_seq, torch.tensor([tok_id]), new_hidden))
        new_beams.sort(key=lambda x: x[0])
        beams = new_beams[:beam_width]
        if not beams: break
    if not completed and beams:
        completed = [(beams[0][0], beams[0][1])]
    if not completed:
        return "i am not sure how to respond to that"
    completed.sort(key=lambda x: x[0])
    words = [vocab.idx2word.get(i,UNK) for i in completed[0][1] if i not in (pad_idx,sos_idx,eos_idx)]
    return " ".join(words) if words else "i am not sure how to respond to that"


def run_test_transcript():
    """Run real inference and save a genuine transcript."""
    model, vocab = load_model()
    test_inputs = [
        "i forgot my password",
        "why was i charged twice",
        "the app keeps crashing",
        "i want to cancel my subscription",
        "where is my order",
        "i need to speak to a human agent",
        "hello",
        "thank you for your help",
    ]
    transcript = []
    transcript.append("="*60)
    transcript.append("HelpDesk AI — Phase 4 Inference Transcript (Real Run)")
    transcript.append("Model: best_model.pt | Decode: Beam Search (k=5)")
    transcript.append("="*60)
    for inp in test_inputs:
        src, src_lens = encode_input(inp, vocab)
        reply = beam_search(model, src, src_lens, vocab, beam_width=5)
        line_q = f"You: {inp}"
        line_a = f"Bot: {reply}"
        print(line_q); print(line_a); print()
        transcript.append(line_q)
        transcript.append(line_a)
        transcript.append("")

    out_path = os.path.join(CKPT, "inference_transcript.txt")
    with open(out_path, "w") as f:
        f.write("\n".join(transcript))
    print(f"Transcript saved -> {out_path}")


if __name__ == "__main__":
    run_test_transcript()
