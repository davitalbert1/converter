import os
import math
import json
import re
from typing import List, Dict, Tuple
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from dataset import get_augmented_dataset

# Hiperparâmetros
PAD_TOKEN = "<pad>"
SOS_TOKEN = "<sos>"
EOS_TOKEN = "<eos>"
UNK_TOKEN = "<unk>"
TASK_PY2CPP = "<2cpp>"
TASK_CPP2PY = "<2py>"

SPECIAL_TOKENS = [PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, UNK_TOKEN, TASK_PY2CPP, TASK_CPP2PY]

class CodeTokenizer:
    def __init__(self):
        self.token2id: Dict[str, int] = {}
        self.id2token: Dict[int, str] = {}
        for token in SPECIAL_TOKENS:
            self.add_token(token)

    def add_token(self, token: str) -> int:
        if token not in self.token2id:
            idx = len(self.token2id)
            self.token2id[token] = idx
            self.id2token[idx] = token
            return idx
        return self.token2id[token]

    def tokenize(self, text: str) -> List[str]:
        # Substitui quebras de linha e espacos por tokens explicitos para reproducao exata
        # Regex que captura operadores compostos, palavras, numeros e pontuacao
        pattern = r"(?:\r?\n| {4}|\t|::|<<|>>|==|!=|<=|>=|\+\+|--|\+=|-=|\*=|/=|->|&&|\|\||[a-zA-Z_]\w*|\d+|[^\s\w])"
        tokens = []
        for line in text.splitlines(keepends=True):
            if line == "\n" or line == "\r\n":
                tokens.append("<newline>")
                continue
            line_tokens = re.findall(pattern, line)
            for t in line_tokens:
                if t in ("\n", "\r\n"):
                    tokens.append("<newline>")
                elif t == "    " or t == "\t":
                    tokens.append("<indent>")
                elif t.strip():
                    tokens.append(t)
        return tokens

    def build_vocab(self, texts: List[str]):
        self.add_token("<newline>")
        self.add_token("<indent>")
        for text in texts:
            for tok in self.tokenize(text): self.add_token(tok)

    def encode(self, text: str, prefix_token: str = None) -> List[int]:
        tokens = self.tokenize(text)
        ids = []
        if prefix_token: ids.append(self.token2id.get(prefix_token, self.token2id[UNK_TOKEN]))
        ids.append(self.token2id[SOS_TOKEN])
        for t in tokens: ids.append(self.token2id.get(t, self.token2id[UNK_TOKEN]))
        ids.append(self.token2id[EOS_TOKEN])
        return ids

    def decode(self, ids: List[int]) -> str:
        tokens = []
        for i in ids:
            tok = self.id2token.get(i, "")
            if tok in (PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, TASK_PY2CPP, TASK_CPP2PY): continue
            tokens.append(tok)

        # Reconstrução do texto preservando quebras de linha e indentação
        code = ""
        for t in tokens:
            if t == "<newline>": code += "\n"
            elif t == "<indent>": code += "    "
            elif t in (",", ";", ")", "}", "]"): code = code.rstrip(" ") + t + " "
            elif t in ("(", "{", "["): code += t
            else: code += t + " "

        return code.strip()

    def save(self, filepath: str):
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump({"token2id": self.token2id}, f, indent=2)

    @classmethod
    def load(cls, filepath: str):
        with open(filepath, "r", encoding="utf-8") as f: data = json.load(f)
        tok = cls()
        tok.token2id = data["token2id"]
        tok.id2token = {v: k for k, v in tok.token2id.items()}
        return tok

    @property
    def vocab_size(self) -> int:
        return len(self.token2id)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 2048):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]

class CodeSeq2SeqTransformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 128,
        nhead: int = 4,
        num_encoder_layers: int = 3,
        num_decoder_layers: int = 3,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        pad_idx: int = 0,
    ):
        super().__init__()
        self.d_model = d_model
        self.pad_idx = pad_idx

        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.pos_encoder = PositionalEncoding(d_model)

        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )

        self.fc_out = nn.Linear(d_model, vocab_size)

    def generate_square_subsequent_mask(self, sz: int, device: torch.device) -> torch.Tensor:
        return torch.triu(torch.ones((sz, sz), device=device, dtype=torch.bool), diagonal=1)

    def create_pad_mask(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor == self.pad_idx

    def forward(self, src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        src_pad_mask = self.create_pad_mask(src)
        tgt_pad_mask = self.create_pad_mask(tgt)
        tgt_mask = self.generate_square_subsequent_mask(tgt.size(1), src.device)

        src_emb = self.pos_encoder(self.embedding(src) * math.sqrt(self.d_model))
        tgt_emb = self.pos_encoder(self.embedding(tgt) * math.sqrt(self.d_model))

        out = self.transformer(
            src=src_emb,
            tgt=tgt_emb,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_pad_mask,
            tgt_key_padding_mask=tgt_pad_mask,
        )
        return self.fc_out(out)

class ParallelCodeDataset(Dataset):
    def __init__(self, pairs: List[Dict[str, str]], tokenizer: CodeTokenizer):
        self.samples = []
        for pair in pairs:
            # Python -> C++
            src_py = tokenizer.encode(pair["py"], prefix_token=TASK_PY2CPP)
            tgt_cpp = tokenizer.encode(pair["cpp"])
            self.samples.append((src_py, tgt_cpp))

            # C++ -> Python
            src_cpp = tokenizer.encode(pair["cpp"], prefix_token=TASK_CPP2PY)
            tgt_py = tokenizer.encode(pair["py"])
            self.samples.append((src_cpp, tgt_py))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[List[int], List[int]]:
        return self.samples[idx]

def collate_fn(batch: List[Tuple[List[int], List[int]]], pad_idx: int):
    src_batch, tgt_batch = zip(*batch)
    max_src_len = max(len(s) for s in src_batch)
    max_tgt_len = max(len(t) for t in tgt_batch)

    padded_src = [s + [pad_idx] * (max_src_len - len(s)) for s in src_batch]
    padded_tgt = [t + [pad_idx] * (max_tgt_len - len(t)) for t in tgt_batch]

    return torch.tensor(padded_src, dtype=torch.long), torch.tensor(padded_tgt, dtype=torch.long)

def train_ai_model(epochs: int = 15, batch_size: int = 16, lr: float = 1e-3, checkpoint_dir: str = "checkpoints", dataset_path: str = None):
    os.makedirs(checkpoint_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[IA] Dispositivo selecionado para treinamento: {device}", flush=True)

    # 1. Carrega dados e constrói tokenizer (de arquivo JSON externo ou pares embutidos)
    pairs = get_augmented_dataset(dataset_path)
    tokenizer = CodeTokenizer()
    all_texts = [p["py"] for p in pairs] + [p["cpp"] for p in pairs]
    tokenizer.build_vocab(all_texts)
    vocab_path = os.path.join(checkpoint_dir, "vocab.json")
    tokenizer.save(vocab_path)
    print(f"[IA] Vocabulário construído com {tokenizer.vocab_size} tokens salvo em {vocab_path}", flush=True)

    # 2. Prepara dataset e dataloader
    dataset = ParallelCodeDataset(pairs, tokenizer)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, tokenizer.token2id[PAD_TOKEN]),
    )
    print(f"[IA] Dataset bidirecional carregado com {len(dataset)} amostras de treino.", flush=True)

    # 3. Instancia modelo
    pad_idx = tokenizer.token2id[PAD_TOKEN]
    model = CodeSeq2SeqTransformer(
        vocab_size=tokenizer.vocab_size,
        d_model=128,
        nhead=4,
        num_encoder_layers=3,
        num_decoder_layers=3,
        dim_feedforward=256,
        dropout=0.1,
        pad_idx=pad_idx,
    ).to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    best_loss = float("inf")
    model_save_path = os.path.join(checkpoint_dir, "best_model.pt")

    print("[IA] Iniciando treinamento da rede neural Transformer...", flush=True)
    model.train()
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        for src, tgt in dataloader:
            src, tgt = src.to(device), tgt.to(device)

            # Entrada do decoder exclui o ultimo token, target exclui o primeiro token (<sos>)
            tgt_input = tgt[:, :-1]
            tgt_expected = tgt[:, 1:]

            optimizer.zero_grad()
            output = model(src, tgt_input)
            loss = criterion(output.reshape(-1, output.shape[-1]), tgt_expected.reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        if avg_loss < best_loss:
            best_loss = avg_loss
            tmp_path = model_save_path + ".tmp"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "loss": best_loss,
                    "vocab_size": tokenizer.vocab_size,
                },
                tmp_path,
            )
            if os.path.exists(model_save_path):
                os.remove(model_save_path)
            os.rename(tmp_path, model_save_path)

        if epoch % 5 == 0 or epoch == 1 or epoch == epochs:
            print(f"  -> Época {epoch:02d}/{epochs:02d} | Perda média (Loss): {avg_loss:.4f} (Melhor: {best_loss:.4f})", flush=True)

    print(f"[IA] Treinamento concluído com sucesso! Modelo salvo em: {model_save_path}", flush=True)
    return model, tokenizer

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Treina modelo de IA Transformer para conversão de código.")
    parser.add_argument("--epochs", type=int, default=15, help="Número de épocas")
    parser.add_argument("--batch-size", type=int, default=8, help="Tamanho do batch")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--dataset", type=str, default=None, help="Caminho para JSON externo com pares de treino [{py, cpp}] (opcional).")
    args = parser.parse_args()

    train_ai_model(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, dataset_path=args.dataset)
