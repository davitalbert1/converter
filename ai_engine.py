import os
import re
from typing import Optional
import torch
from train_model import CodeTokenizer, CodeSeq2SeqTransformer, PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, TASK_PY2CPP, TASK_CPP2PY
from library_mapping import LibraryConverter

class AIEngine:
    def __init__(self, checkpoint_dir: str = "checkpoints"):
        self.checkpoint_dir = checkpoint_dir
        self.model_path = os.path.join(checkpoint_dir, "best_model.pt")
        self.vocab_path = os.path.join(checkpoint_dir, "vocab.json")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model: Optional[CodeSeq2SeqTransformer] = None
        self.tokenizer: Optional[CodeTokenizer] = None
        if self.is_model_available(): self.load_model()

    def is_model_available(self) -> bool:
        if self.model is not None and self.tokenizer is not None: return True
        return os.path.exists(self.model_path) and os.path.exists(self.vocab_path)

    def load_model(self):
        try:
            self.tokenizer = CodeTokenizer.load(self.vocab_path)
            pad_idx = self.tokenizer.token2id[PAD_TOKEN]

            self.model = CodeSeq2SeqTransformer(
                vocab_size=self.tokenizer.vocab_size,
                d_model=128,
                nhead=4,
                num_encoder_layers=3,
                num_decoder_layers=3,
                dim_feedforward=256,
                dropout=0.1,
                pad_idx=pad_idx,
            ).to(self.device)

            checkpoint = torch.load(self.model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.model.eval()
        except Exception as e:
            print(f"[!] Aviso: Falha ao carregar checkpoint do modelo ({e}). O conversor usará o motor estrutural.")
            self.model = None
            self.tokenizer = None

    def translate_neural(self, code: str, direction: str = "py2cpp", max_length: int = 150) -> str:
        if not self.is_model_available(): raise RuntimeError("Modelo de IA não encontrado! Execute 'py train_model.py' primeiro para treinar o modelo.")

        task_token = TASK_PY2CPP if direction == "py2cpp" else TASK_CPP2PY
        src_ids = self.tokenizer.encode(code, prefix_token=task_token)
        src_tensor = torch.tensor([src_ids], dtype=torch.long, device=self.device)

        # Inicia geração com SOS_TOKEN
        sos_id = self.tokenizer.token2id[SOS_TOKEN]
        eos_id = self.tokenizer.token2id[EOS_TOKEN]
        tgt_ids = [sos_id]

        with torch.no_grad():
            for _ in range(max_length):
                tgt_tensor = torch.tensor([tgt_ids], dtype=torch.long, device=self.device)
                output = self.model(src_tensor, tgt_tensor)
                # Seleciona token com maior probabilidade no último passo
                next_token_id = output[:, -1, :].argmax(dim=-1).item()
                tgt_ids.append(next_token_id)

                if next_token_id == eos_id: break

        return self.tokenizer.decode(tgt_ids)

    def detect_imports_with_ai(self, code: str) -> list:
        """Detecta bibliotecas usadas no codigo e retorna os nomes dos
        headers C++ correspondentes, com base nos mapeamentos externos
        (sem hardcode de pares de linguagem)."""
        loader = LibraryConverter.get_loader()
        detected = []
        import_pattern = re.compile(
            r"^\s*(?:import\s+([a-zA-Z0-9_\.,\s]+)|from\s+([a-zA-Z0-9_\.]+)\s+import\s+.*)",
            re.MULTILINE,
        )
        for m in import_pattern.finditer(code):
            if m.group(1):
                modules = [mod.strip().split()[0] for mod in m.group(1).split(",")]
            else:
                modules = [m.group(2).strip()]
            for mod in modules:
                base_mod = mod.split(".")[0]
                key = (
                    mod if mod in loader.py_to_cpp_imports
                    else (base_mod if base_mod in loader.py_to_cpp_imports else None)
                )
                if key:
                    for inc in loader.py_to_cpp_imports[key]:
                        header = inc.replace("#include <", "").replace(">", "").replace('"', "")
                        if header not in detected:
                            detected.append(header)
        return detected
