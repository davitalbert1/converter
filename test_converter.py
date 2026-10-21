from converter import CodeConverter
from train_model import train_ai_model

def test_python_to_cpp():
    print("\n" + "=" * 60)
    print("TESTE 1: Python -> C++ com conversão de bibliotecas (math, time, os, sys, json)")
    print("=" * 60)

    py_code = """import math
import time
import json
import os
import sys

def compute_metrics(radius):
    area = math.pi * math.pow(radius, 2)
    root = math.sqrt(area)
    time.sleep(1.0)
    print("Calculated root:", root)
    return root
"""
    converter = CodeConverter()
    cpp_code = converter.convert(py_code, direction="py2cpp", mode="hybrid")
    print("CÓDIGO ORIGINAL (PYTHON):")
    print(py_code)
    print("\nCÓDIGO RESULTANTE (C++):")
    print(cpp_code)

    assert "#include <cmath>" in cpp_code, "Deveria conter #include <cmath>"
    assert "#include <chrono>" in cpp_code or "#include <thread>" in cpp_code, "Deveria conter includes de chrono/thread"
    assert "M_PI" in cpp_code or "std::pow" in cpp_code, "Deveria converter math.pi/pow para C++"
    print("\n[OK] Teste 1 passou com sucesso!")

def test_cpp_to_python():
    print("\n" + "=" * 60)
    print("TESTE 2: C++ -> Python com conversão de includes (#include <cmath>, <iostream>, <chrono>)")
    print("=" * 60)

    cpp_code = """#include <iostream>
#include <cmath>
#include <chrono>
#include <thread>

double calculate_hypotenuse(double a, double b) {
    double result = std::sqrt(std::pow(a, 2) + std::pow(b, 2));
    std::cout << result << std::endl;
    return result;
}

int main() {
    calculate_hypotenuse(3.0, 4.0);
    return 0;
}
"""
    converter = CodeConverter()
    py_code = converter.convert(cpp_code, direction="cpp2py", mode="hybrid")
    print("CÓDIGO ORIGINAL (C++):")
    print(cpp_code)
    print("\nCÓDIGO RESULTANTE (PYTHON):")
    print(py_code)

    assert "import math" in py_code, "Deveria conter import math"
    assert "math.sqrt" in py_code, "Deveria converter std::sqrt para math.sqrt"
    assert "def calculate_hypotenuse" in py_code, "Deveria converter declaracao de funcao para Python"
    print("\n[OK] Teste 2 passou com sucesso!")

def test_ai_training_and_inference():
    print("\n" + "=" * 60)
    print("TESTE 3: Treinamento e Inferência do Modelo de IA Transformer")
    print("=" * 60)

    # Executa o treino por 10 épocas
    print("Treinando modelo de IA por 10 épocas...")
    train_ai_model(epochs=10, batch_size=16)

    converter = CodeConverter()
    assert converter.ai_engine.is_model_available(), "Checkpoint do modelo de IA deveria existir!"

    sample_py = "import math\ndef calc_circle_area(radius):\n    return math.pi * math.pow(radius, 2)"
    pred_cpp = converter.convert(sample_py, direction="py2cpp", mode="hybrid")
    print("\nCódigo Python de entrada:")
    print(sample_py)
    print("\nCódigo C++ gerado com auxílio da IA:")
    print(pred_cpp)

    print("\n[OK] Teste 3 passou com sucesso! Modelo treinado e integrado ao conversor.")

if __name__ == "__main__":
    test_python_to_cpp()
    test_cpp_to_python()
    test_ai_training_and_inference()
    print("\n" + "=" * 60)
    print("TODOS OS TESTES FORAM CONCLUÍDOS COM SUCESSO!")
    print("=" * 60)
