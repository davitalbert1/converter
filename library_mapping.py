"""
Mapeamento inteligente de bibliotecas e módulos entre Python e C++.
Converte tanto as declarações de import / #include quanto as chamadas de funções equivalentes.
"""

from typing import Dict, List, Set, Tuple
import re

# Mapeamento de imports Python -> C++
PY_TO_CPP_IMPORTS = {
    "math": ["#include <cmath>"],
    "json": ["#include <nlohmann/json.hpp>"],
    "sys": ["#include <iostream>", "#include <cstdlib>"],
    "os": ["#include <filesystem>", "#include <cstdlib>"],
    "os.path": ["#include <filesystem>"],
    "pathlib": ["#include <filesystem>"],
    "time": ["#include <chrono>", "#include <thread>"],
    "datetime": ["#include <chrono>", "#include <ctime>"],
    "random": ["#include <random>"],
    "re": ["#include <regex>"],
    "threading": ["#include <thread>", "#include <mutex>"],
    "collections": ["#include <unordered_map>", "#include <deque>"],
    "typing": ["#include <vector>", "#include <string>", "#include <memory>"],
    "requests": ["#include <cpr/cpr.h>"],
    "sqlite3": ["#include <sqlite3.h>"],
    "shutil": ["#include <filesystem>"],
    "subprocess": ["#include <cstdlib>"],
}

# Mapeamento de funções/métodos Python -> C++
PY_TO_CPP_FUNCTIONS = {
    # math
    r"math\.sqrt\((.*?)\)": r"std::sqrt(\1)",
    r"math\.pow\((.*?),\s*(.*?)\)": r"std::pow(\1, \2)",
    r"math\.sin\((.*?)\)": r"std::sin(\1)",
    r"math\.cos\((.*?)\)": r"std::cos(\1)",
    r"math\.tan\((.*?)\)": r"std::tan(\1)",
    r"math\.floor\((.*?)\)": r"std::floor(\1)",
    r"math\.ceil\((.*?)\)": r"std::ceil(\1)",
    r"math\.pi": r"M_PI",
    r"math\.e": r"M_E",
    r"math\.abs\((.*?)\)": r"std::abs(\1)",
    r"abs\((.*?)\)": r"std::abs(\1)",

    # time
    r"time\.sleep\((.*?)\)": r"std::this_thread::sleep_for(std::chrono::milliseconds(static_cast<int>((\1) * 1000)))",
    r"time\.time\(\)": r"(std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::system_clock::now().time_since_epoch()).count())",

    # json
    r"json\.loads\((.*?)\)": r"nlohmann::json::parse(\1)",
    r"json\.dumps\((.*?)\)": r"nlohmann::json(\1).dump()",

    # os / filesystem
    r"os\.path\.exists\((.*?)\)": r"std::filesystem::exists(\1)",
    r"os\.remove\((.*?)\)": r"std::filesystem::remove(\1)",
    r"os\.mkdir\((.*?)\)": r"std::filesystem::create_directory(\1)",
    r"os\.system\((.*?)\)": r"std::system(\1.c_str())",

    # sys
    r"sys\.exit\((.*?)\)": r"std::exit(\1)",
    
    # print e input
    r"print\((.*?)\)": r"std::cout << \1 << std::endl",
    r"len\((.*?)\)": r"\1.size()",
}

# Mapeamento C++ -> Python (#include -> import)
CPP_TO_PY_IMPORTS = {
    "<cmath>": ["import math"],
    "<cmath.h>": ["import math"],
    "<math.h>": ["import math"],
    "<nlohmann/json.hpp>": ["import json"],
    "<json.hpp>": ["import json"],
    "<iostream>": ["import sys"],
    "<filesystem>": ["import os", "from pathlib import Path"],
    "<cstdlib>": ["import sys", "import os"],
    "<chrono>": ["import time"],
    "<thread>": ["import threading", "import time"],
    "<ctime>": ["import time"],
    "<random>": ["import random"],
    "<regex>": ["import re"],
    "<vector>": [],  # lista nativa no Python
    "<string>": [],  # str nativa no Python
    "<map>": ["from collections import defaultdict"],
    "<unordered_map>": ["from collections import defaultdict"],
    "<algorithm>": [],  # builtins: min, max, sorted
    "<cpr/cpr.h>": ["import requests"],
    "<sqlite3.h>": ["import sqlite3"],
    "<fstream>": [], # open() nativo no Python
}

# Mapeamento de funções/métodos C++ -> Python
CPP_TO_PY_FUNCTIONS = {
    # cmath
    r"std::sqrt\((.*?)\)": r"math.sqrt(\1)",
    r"std::pow\((.*?),\s*(.*?)\)": r"math.pow(\1, \2)",
    r"std::sin\((.*?)\)": r"math.sin(\1)",
    r"std::cos\((.*?)\)": r"math.cos(\1)",
    r"std::tan\((.*?)\)": r"math.tan(\1)",
    r"std::floor\((.*?)\)": r"math.floor(\1)",
    r"std::ceil\((.*?)\)": r"math.ceil(\1)",
    r"std::abs\((.*?)\)": r"abs(\1)",
    r"M_PI": r"math.pi",
    r"M_E": r"math.e",

    # chrono / sleep
    r"std::this_thread::sleep_for\(std::chrono::milliseconds\((.*?)\)\);?": r"time.sleep((\1) / 1000.0)",
    r"std::this_thread::sleep_for\(std::chrono::seconds\((.*?)\)\);?": r"time.sleep(\1)",

    # json
    r"nlohmann::json::parse\((.*?)\)": r"json.loads(\1)",
    r"nlohmann::json\((.*?)\)\.dump\(\)": r"json.dumps(\1)",

    # filesystem
    r"std::filesystem::exists\((.*?)\)": r"os.path.exists(\1)",
    r"std::filesystem::remove\((.*?)\)": r"os.remove(\1)",
    r"std::filesystem::create_directory\((.*?)\)": r"os.mkdir(\1)",
    r"std::system\((.*?)\)": r"os.system(\1)",

    # iostream
    r"std::cout\s*<<\s*(.*?)\s*<<\s*std::endl;?": r"print(\1)",
    r"std::cout\s*<<\s*(.*?);": r"print(\1, end='')",
    r"std::exit\((.*?)\);?": r"sys.exit(\1)",
}


class LibraryConverter:
    """Gerencia a detecção e conversão de bibliotecas e chamadas de métodos."""

    @staticmethod
    def convert_python_imports(code: str) -> Tuple[List[str], str]:
        """
        Detecta imports no código Python e retorna os #includes C++ correspondentes
        e o código sem as linhas de import.
        """
        detected_cpp_includes: Set[str] = set()
        clean_lines: List[str] = []

        import_pattern = re.compile(r"^\s*(?:import\s+([a-zA-Z0-9_\.,\s]+)|from\s+([a-zA-Z0-9_\.]+)\s+import\s+.*)")

        for line in code.splitlines():
            m = import_pattern.match(line)
            if m:
                # Modulos importados
                if m.group(1):
                    modules = [mod.strip().split()[0] for mod in m.group(1).split(",")]
                else:
                    modules = [m.group(2).strip()]

                for mod in modules:
                    # Mapeia modulo completo ou prefixo
                    base_mod = mod.split(".")[0]
                    if mod in PY_TO_CPP_IMPORTS:
                        detected_cpp_includes.update(PY_TO_CPP_IMPORTS[mod])
                    elif base_mod in PY_TO_CPP_IMPORTS:
                        detected_cpp_includes.update(PY_TO_CPP_IMPORTS[base_mod])
            else:
                clean_lines.append(line)

        # Se houver uso implicito de cout / iostream
        if "print(" in code and "#include <iostream>" not in detected_cpp_includes:
            detected_cpp_includes.add("#include <iostream>")

        return sorted(list(detected_cpp_includes)), "\n".join(clean_lines)

    @staticmethod
    def convert_cpp_includes(code: str) -> Tuple[List[str], str]:
        """
        Detecta #include no código C++ e retorna os imports Python correspondentes
        e o código sem as linhas de #include.
        """
        detected_py_imports: Set[str] = set()
        clean_lines: List[str] = []

        include_pattern = re.compile(r"^\s*#include\s+([<\"].*?[>\"])")

        for line in code.splitlines():
            m = include_pattern.match(line)
            if m:
                header = m.group(1).replace('"', "")
                header_key = f"<{header}>" if not header.startswith("<") else header
                if header_key in CPP_TO_PY_IMPORTS:
                    for imp in CPP_TO_PY_IMPORTS[header_key]:
                        detected_py_imports.add(imp)
                elif header in CPP_TO_PY_IMPORTS:
                    for imp in CPP_TO_PY_IMPORTS[header]:
                        detected_py_imports.add(imp)
            else:
                clean_lines.append(line)

        return sorted(list(detected_py_imports)), "\n".join(clean_lines)

    @staticmethod
    def convert_functions_py_to_cpp(code: str) -> str:
        """Substitui chamadas de métodos e funções Python para C++."""
        converted = code
        for pattern, replacement in PY_TO_CPP_FUNCTIONS.items():
            converted = re.sub(pattern, replacement, converted)
        return converted

    @staticmethod
    def convert_functions_cpp_to_py(code: str) -> str:
        """Substitui chamadas de métodos e funções C++ para Python."""
        converted = code
        for pattern, replacement in CPP_TO_PY_FUNCTIONS.items():
            converted = re.sub(pattern, replacement, converted)
        return converted
