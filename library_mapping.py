import json
import os
import re
from typing import List, Set, Tuple, Dict, Any

DEFAULT_MAPPINGS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mappings")


class MappingLoader:
    """Carrega conhecimento de linguagem (imports, funcoes, tipos, operadores,
    literais e padroes sintaticos) de arquivos JSON externos. O motor de
    conversao nao contem hardcode de pares Python <-> C++; e generico e
    guiado por configuracao."""

    FILES = {
        "py_to_cpp_imports": "py_to_cpp_imports.json",
        "py_to_cpp_functions": "py_to_cpp_functions.json",
        "cpp_to_py_imports": "cpp_to_py_imports.json",
        "cpp_to_py_functions": "cpp_to_py_functions.json",
        "type_annotations": "type_annotations.json",
        "operators": "operators.json",
        "literals": "literals.json",
        "syntax": "syntax.json",
    }

    def __init__(self, mappings_dirs=None):
        if mappings_dirs is None:
            mappings_dirs = [DEFAULT_MAPPINGS_DIR]
        elif isinstance(mappings_dirs, str):
            mappings_dirs = [mappings_dirs]
        self.mappings_dirs = list(mappings_dirs)
        self.py_to_cpp_imports: Dict[str, List[str]] = {}
        self.py_to_cpp_functions: Dict[str, str] = {}
        self.cpp_to_py_imports: Dict[str, List[str]] = {}
        self.cpp_to_py_functions: Dict[str, str] = {}
        self.type_annotations: Dict[str, str] = {}
        self.operators: Dict[str, str] = {}
        self.literals: Dict[str, str] = {}
        self.syntax: Dict[str, Any] = {}
        self._load_all()

    def _load_json(self, filename: str) -> Any:
        """Mescla o mesmo arquivo de todos os diretorios configurados.
        Diretorios posteriores sobrescrevem chaves dos anteriores."""
        result: Any = {}
        for mappings_dir in self.mappings_dirs:
            path = os.path.join(mappings_dir, filename)
            if os.path.exists(path):
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    if isinstance(result, dict) and isinstance(data, dict):
                        result.update(data)
                    elif not result:
                        result = data
                except (json.JSONDecodeError, OSError):
                    pass
        return result

    def _load_all(self):
        self.py_to_cpp_imports = self._load_json(self.FILES["py_to_cpp_imports"]) or {}
        self.py_to_cpp_functions = self._load_json(self.FILES["py_to_cpp_functions"]) or {}
        self.cpp_to_py_imports = self._load_json(self.FILES["cpp_to_py_imports"]) or {}
        self.cpp_to_py_functions = self._load_json(self.FILES["cpp_to_py_functions"]) or {}
        self.type_annotations = self._load_json(self.FILES["type_annotations"]) or {}
        self.operators = self._load_json(self.FILES["operators"]) or {}
        self.literals = self._load_json(self.FILES["literals"]) or {}
        self.syntax = self._load_json(self.FILES["syntax"]) or {}

    def reload(self):
        self._load_all()


class LibraryConverter:
    """Converte imports e chamadas de funcao entre Python e C++ usando unicamente
    mapeamentos carregados de arquivos JSON externos."""

    _loader: MappingLoader = MappingLoader()

    @classmethod
    def set_mappings_dir(cls, mappings_dir: str) -> None:
        """Acrescenta um diretorio customizado aos mapeamentos padrao, permitindo
        adicionar ou sobrescrever pares de linguagem sem alterar o codigo-fonte."""
        cls._loader = MappingLoader([DEFAULT_MAPPINGS_DIR, mappings_dir])

    @classmethod
    def get_loader(cls) -> MappingLoader:
        return cls._loader

    @classmethod
    def reload_mappings(cls) -> None:
        cls._loader.reload()

    @staticmethod
    def convert_python_imports(code: str) -> Tuple[List[str], str]:
        loader = LibraryConverter._loader
        detected: Set[str] = set()
        clean: List[str] = []
        pattern = re.compile(
            r"^\s*(?:import\s+([a-zA-Z0-9_\.,\s]+)|from\s+([a-zA-Z0-9_\.]+)\s+import\s+.*)"
        )
        for line in code.splitlines():
            m = pattern.match(line)
            if m:
                modules = ([mod.strip().split()[0] for mod in m.group(1).split(",")]
                           if m.group(1) else [m.group(2).strip()])
                for mod in modules:
                    base = mod.split(".")[0]
                    if mod in loader.py_to_cpp_imports:
                        detected.update(loader.py_to_cpp_imports[mod])
                    elif base in loader.py_to_cpp_imports:
                        detected.update(loader.py_to_cpp_imports[base])
            else:
                clean.append(line)
        if "print(" in code:
            for inc in loader.py_to_cpp_imports.get("iostream", []):
                detected.add(inc)
        return sorted(list(detected)), "\n".join(clean)

    @staticmethod
    def convert_functions_py_to_cpp(code: str) -> str:
        loader = LibraryConverter._loader
        converted = code
        for pat, repl in loader.py_to_cpp_functions.items():
            converted = re.sub(pat, repl, converted)
        return converted

    @staticmethod
    def convert_cpp_includes(code: str) -> Tuple[List[str], str]:
        loader = LibraryConverter._loader
        detected: Set[str] = set()
        clean: List[str] = []
        pattern = re.compile(r"^\s*#include\s+([<\"].*?[>\"])")
        for line in code.splitlines():
            m = pattern.match(line)
            if m:
                header = m.group(1).replace(chr(34), "")
                key = f"<{header}>" if not header.startswith("<") else header
                if key in loader.cpp_to_py_imports:
                    for imp in loader.cpp_to_py_imports[key]:
                        detected.add(imp)
                elif header in loader.cpp_to_py_imports:
                    for imp in loader.cpp_to_py_imports[header]:
                        detected.add(imp)
            else:
                clean.append(line)
        return sorted(list(detected)), "\n".join(clean)

    @staticmethod
    def convert_functions_cpp_to_py(code: str) -> str:
        loader = LibraryConverter._loader
        converted = code
        for pat, repl in loader.cpp_to_py_functions.items():
            converted = re.sub(pat, repl, converted)
        return converted
