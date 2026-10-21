# Conversor Bidirecional de Código Python <-> C++ com IA e Mapeamento de Bibliotecas

Este projeto implementa uma solução completa e robusta em Python capaz de converter códigos **Python para C++** e **C++ para Python**, convertendo automaticamente bibliotecas e imports exclusivos de cada ecossistema e integrando uma **Rede Neural de Inteligência Artificial Transformer Seq2Seq** treinada para detecção e tradução de código.

---

## 🚀 Arquitetura e Destaques

1. **Conversão Inteligente de Bibliotecas e Módulos Exclusivos (`library_mapping.py`)**:
   - Mapeia declarações de importação e `#include` bidirecionalmente:
     - `math` $\longleftrightarrow$ `<cmath>` (`math.sqrt`, `math.pow`, `math.sin`, `math.pi` $\leftrightarrow$ `std::sqrt`, `std::pow`, `std::sin`, `M_PI`)
     - `time` / `datetime` $\longleftrightarrow$ `<chrono>`, `<thread>` (`time.sleep(n)` $\leftrightarrow$ `std::this_thread::sleep_for(...)`)
     - `json` $\longleftrightarrow$ `<nlohmann/json.hpp>` (`json.dumps` / `json.loads` $\leftrightarrow$ `dump()` / `parse()`)
     - `os` / `os.path` / `pathlib` $\longleftrightarrow$ `<filesystem>` (`os.path.exists` $\leftrightarrow$ `std::filesystem::exists`)
     - `sys` $\longleftrightarrow$ `<iostream>`, `<cstdlib>` (`print(...)` $\leftrightarrow$ `std::cout`, `sys.exit` $\leftrightarrow$ `std::exit`)
     - `typing` / coleções $\longleftrightarrow$ `<vector>`, `<string>`, `<unordered_map>`, etc.

2. **Transpiler Estrutural e AST (`ast_transpiler.py`)**:
   - Analisa sintaxe abstrata do Python (`ast.parse`) e estruturas de C++ para inferir tipos (`auto`, `int`, `double`, `std::string`, `void`), controlar fluxo (`if/elif/else`, `while`, `for i in range(...)` $\leftrightarrow$ `for (int i = 0; i < N; i++)`), definir funções e blocos.

3. **Inteligência Artificial Transformer Seq2Seq (`train_model.py` e `ai_engine.py`)**:
   - Modelo PyTorch completo baseado na arquitetura Transformer com *Positional Encoding*, *Multi-Head Self-Attention* e camadas *Encoder-Decoder*.
   - Treinamento bidirecional unificado com tokens de controle de tarefa (`<2cpp>` e `<2py>`).
   - Salva checkpoints (`checkpoints/best_model.pt`) e vocabulário tokenizado (`checkpoints/vocab.json`).
   - O conversor (`CodeConverter`) integra e utiliza o modelo treinado para apoiar a conversão.

---

## 📁 Estrutura de Arquivos

```
F:\converter\
├── requirements.txt      # Dependências do projeto (torch, transformers, rich, etc.)
├── library_mapping.py    # Dicionários de mapeamento e regras de bibliotecas e funções
├── ast_transpiler.py     # Transpiler de árvore sintática (AST) e parser estrutural
├── dataset.py            # Gerador de dataset paralelo Python <-> C++
├── train_model.py        # Script completo de treinamento do modelo de IA Transformer
├── ai_engine.py          # Motor de inferência e predição neural do modelo treinado
├── converter.py          # Classe principal CodeConverter (modo hybrid, ai e ast)
├── main.py               # Interface de linha de comando (CLI)
├── test_converter.py     # Suíte de testes automatizados de ponta a ponta
└── checkpoints/          # Diretório onde os modelos treinados (.pt e vocab.json) são salvos
```

---

## 🛠️ Instalação e Requisitos

Certifique-se de ter Python 3.10+ instalado com PyTorch:

```bash
pip install -r requirements.txt
```

---

## 🧠 Treinando o Modelo de IA

Para treinar o modelo de IA Transformer localmente com o dataset paralelo sintético:

```bash
# Treinamento via script dedicado:
py train_model.py --epochs 15 --batch-size 16

# Ou via main.py:
py main.py --train --epochs 15
```

O treinamento executa automaticamente na GPU via CUDA (se disponível) ou na CPU, salvando os pesos em `checkpoints/best_model.pt`.

---

## 💻 Uso via Linha de Comando (CLI)

### 1. Converter Python para C++
```bash
py main.py --code "import math\nprint(math.sqrt(16))"
```
**Saída C++:**
```cpp
#include <cmath>
#include <iostream>

std::cout << std::sqrt(16) << std::endl;
```

### 2. Converter C++ para Python
```bash
py main.py --code "#include <iostream>\n#include <cmath>\nint main() { std::cout << std::sqrt(25) << std::endl; return 0; }"
```
**Saída Python:**
```python
import math
import sys

def main():
    print(math.sqrt(25))
    return 0

if __name__ == '__main__':
    main()
```

### 3. Converter Arquivos
```bash
# Converte arquivo Python gerando arquivo C++
py main.py --file script.py --output output.cpp

# Converte arquivo C++ gerando arquivo Python
py main.py --file source.cpp --output output.py
```

---

## 🐍 Uso Direto no Código Python

```python
from converter import CodeConverter

converter = CodeConverter()

# Conversão Python -> C++
codigo_py = """
import math
import time

def process(r):
    time.sleep(0.5)
    return math.pi * math.pow(r, 2)
"""

codigo_cpp = converter.convert(codigo_py, direction="py2cpp", mode="hybrid")
print(codigo_cpp)

# Conversão C++ -> Python
codigo_cpp = """
#include <iostream>
#include <cmath>

double hypotenuse(double a, double b) {
    return std::sqrt(std::pow(a, 2) + std::pow(b, 2));
}
"""

codigo_py = converter.convert(codigo_cpp, direction="cpp2py", mode="hybrid")
print(codigo_py)
```

---

## 🧪 Executando os Testes Automatizados

O arquivo `test_converter.py` testa todo o pipeline: conversão de bibliotecas exclusivas, conversão bidirecional e o treinamento e uso do modelo de IA:

```bash
py test_converter.py
```
