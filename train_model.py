import os
import math
import json
import re
import argparse
from typing import List, Dict, Tuple
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from dataset import get_augmented_dataset

PAD_TOKEN = "<pad>"
SOS_TOKEN = "<sos>"
EOS_TOKEN = "<eos>"
UNK_TOKEN = "<unk>"
TASK_PY2CPP = "<2cpp>"
TASK_CPP2PY = "<2py>"

SPECIAL_TOKENS = [
    PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, UNK_TOKEN,
    TASK_PY2CPP, TASK_CPP2PY
]

class CodeTokenizer:
    def __init__(self):
        self.token2id: Dict[str, int] = {}
        self.id2token: Dict[int, str] = {}

        for token in SPECIAL_TOKENS: self.add_token(token)

    def add_token(self, token: str) -> int:
        if token not in self.token2id:
            idx = len(self.token2id)
            self.token2id[token] = idx
            self.id2token[idx] = token
        return self.token2id[token]

    def tokenize(self, text: str) -> List[str]:
        pattern = (
            r"(?:\r?\n| {4}|\t|::|<<|>>|==|!=|<=|>=|"
            r"\+\+|--|\+=|-=|\*=|/=|->|&&|\|\||"
            r"[a-zA-Z_]\w*|\d+|[^\s\w])"
        )

        tokens = []

        for line in text.splitlines(keepends=True):
            if line in ("\n", "\r\n"):
                tokens.append("<newline>")
                continue

            line_tokens = re.findall(pattern, line)

            for token in line_tokens:
                if token in ("\n", "\r\n"):
                    tokens.append("<newline>")
                elif token in ("    ", "\t"):
                    tokens.append("<indent>")
                elif token.strip():
                    tokens.append(token)

        return tokens

    def build_vocab(self, texts: List[str]):
        self.add_token("<newline>")
        self.add_token("<indent>")

        for text in texts:
            for token in self.tokenize(text): self.add_token(token)

    def encode(self, text: str, prefix_token: str = None) -> List[int]:
        ids = []

        if prefix_token:
            ids.append(self.token2id.get(prefix_token, self.token2id[UNK_TOKEN]))

        ids.append(self.token2id[SOS_TOKEN])

        for token in self.tokenize(text):
            ids.append(self.token2id.get(token, self.token2id[UNK_TOKEN]))

        ids.append(self.token2id[EOS_TOKEN])
        return ids

    def decode(self, ids: List[int]) -> str:
        tokens = []

        for idx in ids:
            token = self.id2token.get(int(idx), "")

            if token in (
                PAD_TOKEN, SOS_TOKEN, EOS_TOKEN,
                TASK_PY2CPP, TASK_CPP2PY, UNK_TOKEN
            ):
                continue

            tokens.append(token)

        code = ""

        for token in tokens:
            if token == "<newline>":
                code = code.rstrip() + "\n"
            elif token == "<indent>":
                code += "    "
            elif token in (",", ";", ")", "}", "]"):
                code = code.rstrip() + token
                if token in (",", ";"): code += " "
            elif token in ("(", "["):
                code += token
            elif token == "{":
                code = code.rstrip() + " {"
            elif token == ".":
                code = code.rstrip() + "."
            elif token in ("::", "->", "++", "--"):
                code = code.rstrip() + token
            else:
                if code and not code.endswith((" ", "\n", "(", "[", ".")):
                    code += " "
                code += token

        return code.strip()

    def save(self, filepath: str):
        with open(filepath, "w", encoding="utf-8") as file:
            json.dump(
                {"token2id": self.token2id},
                file,
                indent=2,
                ensure_ascii=False
            )

    @classmethod
    def load(cls, filepath: str):
        with open(filepath, "r", encoding="utf-8") as file:
            data = json.load(file)

        tokenizer = cls()
        tokenizer.token2id = data["token2id"]
        tokenizer.id2token = {
            int(idx): token
            for token, idx in tokenizer.token2id.items()
        }
        return tokenizer

    @property
    def vocab_size(self) -> int:
        return len(self.token2id)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 2048):
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(0, d_model, 2).float()
            * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[:pe[:, 1::2].shape[1]])

        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.size(1) > self.pe.size(1):
            raise ValueError(
                f"Sequência com {x.size(1)} tokens excede "
                f"o limite de {self.pe.size(1)}."
            )

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

    def generate_square_subsequent_mask(self, size: int, device: torch.device) -> torch.Tensor:
        return torch.triu(
            torch.ones(
                (size, size), device=device, dtype=torch.bool
            ),
            diagonal=1
        )

    def forward(self, src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        src_pad_mask = src == self.pad_idx
        tgt_pad_mask = tgt == self.pad_idx
        tgt_mask = self.generate_square_subsequent_mask(tgt.size(1), src.device)

        src_emb = self.pos_encoder(self.embedding(src) * math.sqrt(self.d_model))
        tgt_emb = self.pos_encoder(self.embedding(tgt) * math.sqrt(self.d_model))

        output = self.transformer(
            src=src_emb,
            tgt=tgt_emb,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_pad_mask,
            tgt_key_padding_mask=tgt_pad_mask,
            memory_key_padding_mask=src_pad_mask,
        )

        return self.fc_out(output)

class ParallelCodeDataset(Dataset):
    def __init__(self, pairs: List[Dict[str, str]], tokenizer: CodeTokenizer):
        self.samples = []

        for pair in pairs:
            src_py = tokenizer.encode(pair["py"], prefix_token=TASK_PY2CPP)
            tgt_cpp = tokenizer.encode(pair["cpp"])
            self.samples.append((src_py, tgt_cpp))

            src_cpp = tokenizer.encode(pair["cpp"], prefix_token=TASK_CPP2PY)
            tgt_py = tokenizer.encode(pair["py"])
            self.samples.append((src_cpp, tgt_py))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[List[int], List[int]]:
        return self.samples[idx]

def collate_fn(batch, pad_idx: int):
    src_batch, tgt_batch = zip(*batch)

    max_src = max(len(seq) for seq in src_batch)
    max_tgt = max(len(seq) for seq in tgt_batch)

    padded_src = [seq + [pad_idx] * (max_src - len(seq)) for seq in src_batch]
    padded_tgt = [seq + [pad_idx] * (max_tgt - len(seq)) for seq in tgt_batch]

    return (
        torch.tensor(padded_src, dtype=torch.long),
        torch.tensor(padded_tgt, dtype=torch.long)
    )

def train_ai_model(epochs: int = 100, cycles: int = 10, batch_size: int = 8, lr: float = 1e-3, checkpoint_dir: str = "checkpoints", dataset_path: str = None):
    os.makedirs(checkpoint_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n[IA] Dispositivo: {device}", flush=True)
    print(f"[IA] Ciclos: {cycles}", flush=True)
    print(f"[IA] Épocas por ciclo: {epochs}", flush=True)
    print(f"[IA] Total planejado: {cycles * epochs} épocas", flush=True)

    # Carrega dataset
    pairs = get_augmented_dataset(dataset_path)

    if not pairs: raise ValueError("O dataset está vazio.")

    tokenizer = CodeTokenizer()

    all_texts = [pair["py"] for pair in pairs] + [pair["cpp"] for pair in pairs]
    tokenizer.build_vocab(all_texts)

    vocab_path = os.path.join(checkpoint_dir, "vocab.json")
    tokenizer.save(vocab_path)

    print(f"[IA] Vocabulário: {tokenizer.vocab_size} tokens")
    print(f"[IA] Dataset: {len(pairs)} pares", flush=True)

    pad_idx = tokenizer.token2id[PAD_TOKEN]

    dataset = ParallelCodeDataset(pairs, tokenizer)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda batch: collate_fn(batch, pad_idx),
        pin_memory=(device.type == "cuda"),
    )

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
    model_save_path = os.path.join(checkpoint_dir, "best_model.pt")

    best_loss = float("inf")
    global_epoch = 0

    # LOOP EXTERNO: CICLOS
    for cycle in range(1, cycles + 1):
        print("\n" + "=" * 65, flush=True)
        print(f" CICLO {cycle}/{cycles}", flush=True)
        print("=" * 65, flush=True)

        # LOOP INTERNO: ÉPOCAS
        for epoch in range(1, epochs + 1):
            global_epoch += 1
            model.train()

            total_loss = 0.0
            total_correct = 0
            total_tokens = 0

            for src, tgt in dataloader:
                src = src.to(device, non_blocking=True)
                tgt = tgt.to(device, non_blocking=True)

                tgt_input = tgt[:, :-1]
                tgt_expected = tgt[:, 1:]

                optimizer.zero_grad(set_to_none=True)

                output = model(src, tgt_input)

                loss = criterion(output.reshape(-1, output.size(-1)), tgt_expected.reshape(-1))

                loss.backward()

                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                optimizer.step()

                total_loss += loss.item()

                # Accuracy por token, ignorando padding
                predictions = output.argmax(dim=-1)
                mask = tgt_expected != pad_idx

                total_correct += ((predictions == tgt_expected) & mask).sum().item()
                total_tokens += mask.sum().item()

            avg_loss = total_loss / max(len(dataloader), 1)
            accuracy = (
                100.0 * total_correct / total_tokens
                if total_tokens else 0.0
            )

            # Salva o melhor modelo pela menor loss
            if avg_loss < best_loss:
                best_loss = avg_loss

                checkpoint = {
                    "cycle": cycle,
                    "epoch": epoch,
                    "global_epoch": global_epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": best_loss,
                    "accuracy": accuracy,
                    "vocab_size": tokenizer.vocab_size,
                }

                temp_path = model_save_path + ".tmp"
                torch.save(checkpoint, temp_path)
                os.replace(temp_path, model_save_path)

                best_marker = " [MELHOR]"
            else:
                best_marker = ""

            # Exibe progresso no terminal
            if (epoch == 1 or epoch % 5 == 0 or epoch == epochs):
                print(
                    f"[Ciclo {cycle}/{cycles}] "
                    f"[Época {epoch}/{epochs}] "
                    f"[Total {global_epoch}/{cycles * epochs}] "
                    f"Loss: {avg_loss:.4f} | "
                    f"ACC: {accuracy:.2f}% | "
                    f"Melhor Loss: {best_loss:.4f}"
                    f"{best_marker}",
                    flush=True
                )

        print(f"[IA] Ciclo {cycle}/{cycles} concluído.", flush=True)

    print("\n" + "=" * 65, flush=True)
    print("[IA] TODOS OS CICLOS FORAM CONCLUÍDOS!", flush=True)
    print(f"[IA] Melhor Loss: {best_loss:.4f}", flush=True)
    print(f"[IA] Modelo: {model_save_path}", flush=True)
    print(f"[IA] Vocabulário: {vocab_path}", flush=True)
    print("=" * 65, flush=True)

    return model, tokenizer

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Treinamento Transformer Python <-> C++")
    parser.add_argument("--epochs", type=int, default=100, help="Épocas por ciclo")
    parser.add_argument("--cycles", type=int, default=10, help="Quantidade de ciclos automáticos")
    parser.add_argument("--batch-size", type=int, default=8, help="Tamanho do batch")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--dataset", type=str, default=None, help="Caminho para JSON do dataset")

    args = parser.parse_args()

    if args.epochs < 1 or args.cycles < 1 or args.batch_size < 1:
        parser.error("epochs, cycles e batch-size devem ser >= 1")

    train_ai_model(
        epochs=args.epochs,
        cycles=args.cycles,
        batch_size=args.batch_size,
        lr=args.lr,
        dataset_path=args.dataset,
    )
