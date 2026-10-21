"""
Módulo Principal do Conversor de Código Python <-> C++.
Integra o Motor de IA, o Conversor de Bibliotecas e o Transpiler Estrutural.
"""

from typing import Optional, Dict, Any
import re
from library_mapping import LibraryConverter
from ast_transpiler import AstTranspiler
from ai_engine import AIEngine


class CodeConverter:
    """
    Conversor bidirecional de código entre Python e C++.
    Suporta conversão estrutural de sintaxe, conversão inteligente de bibliotecas/imports
    e integração com modelo de Inteligência Artificial treinado.
    """

    def __init__(self, checkpoint_dir: str = "checkpoints"):
        self.ai_engine = AIEngine(checkpoint_dir=checkpoint_dir)

    def convert(self, source_code: str, direction: str = "auto", mode: str = "hybrid") -> str:
        """
        Converte o código de entrada.

        :param source_code: Código fonte a ser convertido.
        :param direction: 'py2cpp', 'cpp2py' ou 'auto'.
        :param mode: 'hybrid' (IA + AST/Estrutural), 'ai' (Puramente Neural) ou 'ast' (Estrutural).
        :return: Código convertido na nova linguagem.
        """
        if direction == "auto":
            direction = self.detect_direction(source_code)

        if direction not in ("py2cpp", "cpp2py"):
            raise ValueError(f"Direção inválida '{direction}'. Use 'py2cpp', 'cpp2py' ou 'auto'.")

        # Modo 1: Puramente IA
        if mode == "ai":
            if not self.ai_engine.is_model_available():
                raise RuntimeError("Modelo de IA ainda não foi treinado. Execute 'py train_model.py' ou use mode='hybrid'/'ast'.")
            return self.ai_engine.translate_neural(source_code, direction=direction)

        # Modo 2: Estrutural puro (AST + Library mapping)
        if mode == "ast":
            if direction == "py2cpp":
                return AstTranspiler.python_to_cpp(source_code)
            else:
                return AstTranspiler.cpp_to_python(source_code)

        # Modo 3: Híbrido (Padrão) - IA para mapeamento de conceitos + Transpilação determinística
        return self._convert_hybrid(source_code, direction)

    def _convert_hybrid(self, source_code: str, direction: str) -> str:
        """
        Abordagem híbrida:
        1. Utiliza a IA treinada para detectar itens semânticos e gerar predições.
        2. Aplica transpilação estrutural e substituição precisa de bibliotecas exclusivas.
        """
        if direction == "py2cpp":
            # Converte usando o AST Transpiler com suporte a bibliotecas
            converted_code = AstTranspiler.python_to_cpp(source_code)

            # Se o modelo neural estiver disponível, podemos usá-lo para refinar ou complementar
            if self.ai_engine.is_model_available():
                try:
                    # IA detecta bibliotecas implícitas adicionais
                    ai_detected = self.ai_engine.detect_imports_with_ai(source_code)
                    for lib in ai_detected:
                        inc = f"#include <{lib}>"
                        if inc not in converted_code:
                            converted_code = f"{inc}\n" + converted_code
                except Exception:
                    pass

            return converted_code

        else: # cpp2py
            # Converte usando o C++ to Python Transpiler
            converted_code = AstTranspiler.cpp_to_python(source_code)
            return converted_code

    @staticmethod
    def detect_direction(code: str) -> str:
        """Detecta automaticamente se o código é Python ou C++."""
        cpp_indicators = [
            r"#include",
            r"std::",
            r"\bint\s+main\b",
            r"\bcout\b",
            r"->",
            r"\bvoid\b",
            r";\s*$",
        ]
        py_indicators = [
            r"\bdef\s+\w+\(",
            r"\belif\b",
            r"\bimport\s+\w+",
            r"\bfrom\s+\w+\s+import",
            r":\s*$",
        ]

        cpp_score = sum(1 for pattern in cpp_indicators if re.search(pattern, code, re.MULTILINE))
        py_score = sum(1 for pattern in py_indicators if re.search(pattern, code, re.MULTILINE))

        return "cpp2py" if cpp_score > py_score else "py2cpp"
