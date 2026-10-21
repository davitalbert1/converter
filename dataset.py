from typing import List, Dict
import json

DATASET_PAIRS = [
    # 1. Biblioteca math / cmath
    {
        "py": "import math\n\ndef calc_circle_area(radius):\n    return math.pi * math.pow(radius, 2)",
        "cpp": "#include <cmath>\n\ndouble calc_circle_area(double radius) {\n    return M_PI * std::pow(radius, 2);\n}",
    },
    {
        "py": "import math\n\ndef hypotenuse(a, b):\n    return math.sqrt(math.pow(a, 2) + math.pow(b, 2))",
        "cpp": "#include <cmath>\n\ndouble hypotenuse(double a, double b) {\n    return std::sqrt(std::pow(a, 2) + std::pow(b, 2));\n}",
    },
    {
        "py": "import math\n\ndef get_sine(angle):\n    return math.sin(angle)",
        "cpp": "#include <cmath>\n\ndouble get_sine(double angle) {\n    return std::sin(angle);\n}",
    },

    # 2. Biblioteca time / chrono
    {
        "py": "import time\n\ndef wait_seconds(sec):\n    time.sleep(sec)",
        "cpp": "#include <chrono>\n#include <thread>\n\nvoid wait_seconds(int sec) {\n    std::this_thread::sleep_for(std::chrono::seconds(sec));\n}",
    },
    {
        "py": "import time\n\ndef wait_half_sec():\n    time.sleep(0.5)",
        "cpp": "#include <chrono>\n#include <thread>\n\nvoid wait_half_sec() {\n    std::this_thread::sleep_for(std::chrono::milliseconds(500));\n}",
    },

    # 3. Biblioteca json / nlohmann/json
    {
        "py": "import json\n\ndef parse_data(raw_text):\n    return json.loads(raw_text)",
        "cpp": "#include <nlohmann/json.hpp>\n\nnlohmann::json parse_data(const std::string& raw_text) {\n    return nlohmann::json::parse(raw_text);\n}",
    },
    {
        "py": "import json\n\ndef serialize_data(data):\n    return json.dumps(data)",
        "cpp": "#include <nlohmann/json.hpp>\n\nstd::string serialize_data(const nlohmann::json& data) {\n    return data.dump();\n}",
    },

    # 4. Biblioteca os / sys / filesystem
    {
        "py": "import os\n\ndef check_file(path):\n    return os.path.exists(path)",
        "cpp": "#include <filesystem>\n\nbool check_file(const std::string& path) {\n    return std::filesystem::exists(path);\n}",
    },
    {
        "py": "import os\n\ndef delete_file(path):\n    os.remove(path)",
        "cpp": "#include <filesystem>\n\nvoid delete_file(const std::string& path) {\n    std::filesystem::remove(path);\n}",
    },
    {
        "py": "import sys\n\ndef terminate_app(code):\n    sys.exit(code)",
        "cpp": "#include <cstdlib>\n\nvoid terminate_app(int code) {\n    std::exit(code);\n}",
    },

    # 5. Funções básicas, condicionais e laços
    {
        "py": "def add(a, b):\n    return a + b",
        "cpp": "int add(int a, int b) {\n    return a + b;\n}",
    },
    {
        "py": "def is_even(num):\n    if num % 2 == 0:\n        return True\n    else:\n        return False",
        "cpp": "bool is_even(int num) {\n    if (num % 2 == 0) {\n        return true;\n    } else {\n        return false;\n    }\n}",
    },
    {
        "py": "def count_up(n):\n    for i in range(n):\n        print(i)",
        "cpp": "#include <iostream>\n\nvoid count_up(int n) {\n    for (int i = 0; i < n; i++) {\n        std::cout << i << std::endl;\n    }\n}",
    },
    {
        "py": "def factorial(n):\n    res = 1\n    for i in range(1, n + 1):\n        res = res * i\n    return res",
        "cpp": "int factorial(int n) {\n    int res = 1;\n    for (int i = 1; i <= n; i++) {\n        res = res * i;\n    }\n    return res;\n}",
    },
    {
        "py": "def countdown(start):\n    while start > 0:\n        print(start)\n        start -= 1",
        "cpp": "#include <iostream>\n\nvoid countdown(int start) {\n    while (start > 0) {\n        std::cout << start << std::endl;\n        start -= 1;\n    }\n}",
    },
    {
        "py": "def greet(name):\n    print(\"Hello, \", name)",
        "cpp": "#include <iostream>\n#include <string>\n\nvoid greet(const std::string& name) {\n    std::cout << \"Hello, \" << name << std::endl;\n}",
    },
]

def get_augmented_dataset() -> List[Dict[str, str]]:
    dataset = list(DATASET_PAIRS)

    # Adiciona variações sintéticas
    variations = [
        ("radius", "r"),
        ("a, b", "x, y"),
        ("num", "val"),
        ("sec", "seconds"),
        ("path", "filepath"),
        ("start", "counter"),
    ]

    for item in DATASET_PAIRS:
        for old, new in variations:
            if old in item["py"]:
                py_var = item["py"].replace(old, new)
                cpp_var = item["cpp"].replace(old, new)
                dataset.append({"py": py_var, "cpp": cpp_var})

    return dataset

def save_dataset_json(filepath: str = "dataset.json"):
    data = get_augmented_dataset()
    with open(filepath, "w", encoding="utf-8") as f: json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"Dataset salvo com {len(data)} exemplos em {filepath}")

if __name__ == "__main__":
    save_dataset_json()
