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
├── library_mapping.py    # Carrega mapeamentos de bibliotecas e funções a partir de JSONs externos
├── ast_transpiler.py     # Transpiler de árvore sintática (AST) e parser estrutural
├── dataset.py            # Gerador/carregador de dataset paralelo Python <-> C++
├── train_model.py        # Script completo de treinamento do modelo de IA Transformer
├── ai_engine.py          # Motor de inferência e predição neural do modelo treinado
├── converter.py          # Classe principal CodeConverter (modo hybrid, ai e ast)
├── main.py               # Interface de linha de comando (CLI)
├── test_converter.py     # Suíte de testes automatizados de ponta a ponta
├── mappings/             # Mapeamentos JSON de linguagem (imports, funções, tipos, operadores, literais, sintaxe)
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

## 🧩 Extensibilidade sem hardcode

Todo o conhecimento de linguagem (imports, funções, tipos, operadores,
literais e padrões sintáticos) está em arquivos JSON dentro de `mappings/`.
O código-fonte **não contém hardcode de pares Python <-> C++**: o motor de
conversão é genérico e guiado por configuração.

Para adicionar suporte a uma **nova biblioteca ou função**, basta editar ou
acrescentar entradas nos JSONs de `mappings/`, sem alterar nenhuma linha de
Python.

Exemplo: para suportar uma biblioteca fictícia `mylib`, crie um diretório
`meus_mapeamentos/` com:

- `py_to_cpp_imports.json`: `{ "mylib": ["#include <mylib.h>"] }`
- `py_to_cpp_functions.json`: `{ "mylib\.foo\((.*?)\)": "mylib_foo(\1)" }`

E use o conversor apontando para esse diretório (ele é **mesclado** aos
mapeamentos padrão):

```bash
py main.py --code "import mylib
x = mylib.foo(42)" --mappings-dir meus_mapeamentos
```

Arquivos JSON disponíveis em `mappings/`:
- `py_to_cpp_imports.json` / `cpp_to_py_imports.json` — imports ↔ includes
- `py_to_cpp_functions.json` / `cpp_to_py_functions.json` — chamadas de função
- `type_annotations.json` — anotações de tipo Python → C++
- `operators.json` — operadores da AST Python → símbolos C++
- `literals.json` — literais (verdadeiro/falso/nulo)
- `syntax.json` — padrões sintáticos do C++ (main, função, for, while, if,
  atribuição, return)

O treinamento da IA também aceita um dataset externo, evitando pares de treino
hardcoded:
```bash
py train_model.py --dataset pares_treino.json --epochs 15
```
ou
```bash
py main.py --train --dataset pares_treino.json
```
