"""
Interface de Linha de Comando (CLI) para o Conversor de Código Python <-> C++.
"""

import sys
import argparse
from pathlib import Path
from converter import CodeConverter
from train_model import train_ai_model


def main():
    parser = argparse.ArgumentParser(
        description="Conversor Bidirecional de Código Python <-> C++ com IA e Mapeamento de Bibliotecas"
    )
    parser.add_argument(
        "--file", "-f", type=str, help="Caminho do arquivo de código a ser convertido"
    )
    parser.add_argument(
        "--code", "-c", type=str, help="Snippet de código passado diretamente como texto"
    )
    parser.add_argument(
        "--output", "-o", type=str, help="Caminho do arquivo de saída (opcional)"
    )
    parser.add_argument(
        "--direction",
        "-d",
        type=str,
        default="auto",
        choices=["auto", "py2cpp", "cpp2py"],
        help="Direção da conversão (padrão: auto)",
    )
    parser.add_argument(
        "--mode",
        "-m",
        type=str,
        default="hybrid",
        choices=["hybrid", "ai", "ast"],
        help="Modo de conversão: hybrid (IA + Estrutural), ai (Puramente Neural), ast (Estrutural)",
    )
    parser.add_argument(
        "--train", action="store_true", help="Dispara o treinamento do modelo de IA"
    )
    parser.add_argument(
        "--epochs", type=int, default=15, help="Número de épocas para o treinamento da IA"
    )

    args = parser.parse_args()

    if args.train:
        print("[*] Iniciando treinamento do modelo de IA...")
        train_ai_model(epochs=args.epochs)
        print("[OK] Treinamento finalizado.")
        return

    # Obter o código de entrada
    source_code = None
    if args.file:
        file_path = Path(args.file)
        if not file_path.exists():
            print(f"[!] Erro: Arquivo {args.file} não foi encontrado.", file=sys.stderr)
            sys.exit(1)
        source_code = file_path.read_text(encoding="utf-8")
    elif args.code:
        source_code = args.code
    else:
        print("Digite o código a ser convertido (pressione Ctrl+Z no Windows ou Ctrl+D no Linux para finalizar):")
        source_code = sys.stdin.read()

    if not source_code.strip():
        print("[!] Nenhum código fornecido.", file=sys.stderr)
        sys.exit(1)

    converter = CodeConverter()
    try:
        converted = converter.convert(source_code, direction=args.direction, mode=args.mode)
        
        if args.output:
            Path(args.output).write_text(converted, encoding="utf-8")
            print(f"[OK] Código convertido salvo em: {args.output}")
        else:
            print("\n" + "=" * 50)
            print("--- CÓDIGO CONVERTIDO ---")
            print("=" * 50)
            print(converted)
            print("=" * 50)

    except Exception as e:
        print(f"[!] Erro durante a conversão: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
