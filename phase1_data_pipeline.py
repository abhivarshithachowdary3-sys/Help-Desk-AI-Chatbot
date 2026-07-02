"""
Phase 2 — Seq2Seq Architecture: BiGRU Encoder + Bahdanau Attention + GRU Decoder
"""
import torch, torch.nn as nn, torch.nn.functional as F, random

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, n_layers=1, dropout=0.2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(embed_dim, hidden_dim, num_layers=n_layers,
                          batch_first=True, bidirectional=True,
                          dropout=dropout if n_layers>1 else 0.0)
        self.fc = nn.Linear(hidden_dim*2, hidden_dim)

    def forward(self, src, src_lens):
        embedded = self.dropout(self.embedding(src))
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded, src_lens.cpu().clamp(min=1), batch_first=True, enforce_sorted=False)
        packed_out, hidden = self.gru(packed)
        outputs,_ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)
        merged = torch.tanh(self.fc(torch.cat([hidden[-2],hidden[-1]],dim=-1)))
        return outputs, merged


class BahdanauAttention(nn.Module):
    def __init__(self, dec_h, enc_h):
        super().__init__()
        self.W_a = nn.Linear(dec_h, dec_h, bias=False)
        self.U_a = nn.Linear(enc_h*2, dec_h, bias=False)
        self.v = nn.Linear(dec_h, 1, bias=False)

    def forward(self, dec_hidden, enc_outputs):
        T = enc_outputs.size(1)
        s = self.W_a(dec_hidden).unsqueeze(1).expand(-1,T,-1)
        h = self.U_a(enc_outputs)
        energy = self.v(torch.tanh(s+h)).squeeze(-1)
        weights = F.softmax(energy, dim=-1)
        context = torch.bmm(weights.unsqueeze(1), enc_outputs).squeeze(1)
        return context, weights


class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, enc_hidden_dim, n_layers=1, dropout=0.2):
        super().__init__()
        self.hidden_dim=hidden_dim; self.vocab_size=vocab_size; self.n_layers=n_layers
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.dropout = nn.Dropout(dropout)
        self.attention = BahdanauAttention(hidden_dim, enc_hidden_dim)
        self.gru = nn.GRU(embed_dim+enc_hidden_dim*2, hidden_dim, num_layers=n_layers,
                          batch_first=True, dropout=dropout if n_layers>1 else 0.0)
        self.out = nn.Linear(hidden_dim+enc_hidden_dim*2, vocab_size)

    def forward_step(self, token, hidden, enc_outputs):
        embedded = self.dropout(self.embedding(token.unsqueeze(1)))
        context, weights = self.attention(hidden[-1], enc_outputs)
        gru_in = torch.cat([embedded, context.unsqueeze(1)], dim=-1)
        out, hidden = self.gru(gru_in, hidden)
        out = out.squeeze(1)
        logits = self.out(torch.cat([out, context], dim=-1))
        return logits, hidden, weights

    def forward(self, tgt, hidden, enc_outputs, teacher_force_ratio=0.5, sos_idx=1):
        B,T = tgt.size()
        all_logits = torch.zeros(B,T,self.vocab_size, device=tgt.device)
        if hidden.dim()==2:
            hidden = hidden.unsqueeze(0).repeat(self.n_layers,1,1)
        token = torch.full((B,), sos_idx, dtype=torch.long, device=tgt.device)
        for t in range(T):
            logits, hidden, _ = self.forward_step(token, hidden, enc_outputs)
            all_logits[:,t]=logits
            token = tgt[:,t] if random.random()<teacher_force_ratio else logits.argmax(-1)
        return all_logits


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder=encoder; self.decoder=decoder

    def forward(self, src, src_lens, tgt, teacher_force_ratio=0.5):
        enc_outputs, enc_hidden = self.encoder(src, src_lens)
        return self.decoder(tgt, enc_hidden, enc_outputs, teacher_force_ratio)


def build_model(vocab_size, embed_dim=128, hidden_dim=256, enc_layers=1, dec_layers=1, dropout=0.2):
    encoder = Encoder(vocab_size, embed_dim, hidden_dim, enc_layers, dropout)
    decoder = Decoder(vocab_size, embed_dim, hidden_dim, hidden_dim, dec_layers, dropout)
    model = Seq2Seq(encoder, decoder).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model built | Params: {n_params:,} | Device: {DEVICE}")
    return model


if __name__=="__main__":
    m = build_model(500)
    src = torch.randint(0,500,(4,19)); src_lens=torch.tensor([19,17,15,12])
    tgt = torch.randint(0,500,(4,19))
    out = m(src, src_lens, tgt)
    print("Output shape:", out.shape)
