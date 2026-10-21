
import ast
import re
from typing import List, Dict
from library_mapping import LibraryConverter


class PythonToCppVisitor(ast.NodeVisitor):
    """Transpila a AST do Python para C++ usando unicamente mapeamentos
    carregados de arquivos JSON externos (tipos, operadores, literais)."""

    def __init__(self, loader=None):
        self.loader = loader or LibraryConverter.get_loader()
        self.indent_level = 0
        self.code_lines: List[str] = []
        self.type_inference: Dict[str, str] = {}
        self.includes: set = set()

    def indent(self) -> str:
        return "    " * self.indent_level

    def visit_Module(self, node: ast.Module):
        for stmt in node.body:
            self.visit(stmt)

    def visit_Import(self, node: ast.Import):
        for alias in node.names:
            incs, _ = LibraryConverter.convert_python_imports(f"import {alias.name}")
            self.includes.update(incs)

    def visit_ImportFrom(self, node: ast.ImportFrom):
        mod = node.module or ""
        incs, _ = LibraryConverter.convert_python_imports(f"from {mod} import ...")
        self.includes.update(incs)

    def visit_FunctionDef(self, node: ast.FunctionDef):
        ret_type = "void"
        if node.returns:
            ret_type = self._map_type_annotation(node.returns)
        else:
            for sub_node in ast.walk(node):
                if isinstance(sub_node, ast.Return) and sub_node.value is not None:
                    ret_type = self._infer_expr_type(sub_node.value)
                    break

        params = []
        for arg in node.args.args:
            arg_name = arg.arg
            arg_type = "auto"
            if arg.annotation:
                arg_type = self._map_type_annotation(arg.annotation)
            params.append(f"{arg_type} {arg_name}")

        param_str = ", ".join(params)
        self.code_lines.append(f"{self.indent()}{ret_type} {node.name}({param_str}) {{")
        self.indent_level += 1
        for stmt in node.body:
            self.visit(stmt)
        self.indent_level -= 1
        self.code_lines.append(f"{self.indent()}}}\n")

    def visit_Return(self, node: ast.Return):
        if node.value:
            val_str = self.visit_expr(node.value)
            self.code_lines.append(f"{self.indent()}return {val_str};")
        else:
            self.code_lines.append(f"{self.indent()}return;")

    def visit_Assign(self, node: ast.Assign):
        val_str = self.visit_expr(node.value)
        val_type = self._infer_expr_type(node.value)
        for target in node.targets:
            if isinstance(target, ast.Name):
                var_name = target.id
                if var_name not in self.type_inference:
                    self.type_inference[var_name] = val_type
                    self.code_lines.append(f"{self.indent()}{val_type} {var_name} = {val_str};")
                else:
                    self.code_lines.append(f"{self.indent()}{var_name} = {val_str};")
            else:
                target_str = self.visit_expr(target)
                self.code_lines.append(f"{self.indent()}{target_str} = {val_str};")

    def visit_AugAssign(self, node: ast.AugAssign):
        target = self.visit_expr(node.target)
        op = self._get_op_symbol(node.op)
        val = self.visit_expr(node.value)
        self.code_lines.append(f"{self.indent()}{target} {op}= {val};")

    def visit_If(self, node: ast.If):
        test_str = self.visit_expr(node.test)
        self.code_lines.append(f"{self.indent()}if ({test_str}) {{")
        self.indent_level += 1
        for stmt in node.body:
            self.visit(stmt)
        self.indent_level -= 1
        if node.orelse:
            if len(node.orelse) == 1 and isinstance(node.orelse[0], ast.If):
                self.code_lines.append(f"{self.indent()}}} else ")
                elif_node = node.orelse[0]
                elif_test = self.visit_expr(elif_node.test)
                last_line = self.code_lines.pop()
                self.code_lines.append(f"{last_line}if ({elif_test}) {{")
                self.indent_level += 1
                for stmt in elif_node.body:
                    self.visit(stmt)
                self.indent_level -= 1
                if elif_node.orelse:
                    self.code_lines.append(f"{self.indent()}}} else {{")
                    self.indent_level += 1
                    for stmt in elif_node.orelse:
                        self.visit(stmt)
                    self.indent_level -= 1
                    self.code_lines.append(f"{self.indent()}}}")
                else:
                    self.code_lines.append(f"{self.indent()}}}")
            else:
                self.code_lines.append(f"{self.indent()}}} else {{")
                self.indent_level += 1
                for stmt in node.orelse:
                    self.visit(stmt)
                self.indent_level -= 1
                self.code_lines.append(f"{self.indent()}}}")
        else:
            self.code_lines.append(f"{self.indent()}}}")

    def visit_While(self, node: ast.While):
        test_str = self.visit_expr(node.test)
        self.code_lines.append(f"{self.indent()}while ({test_str}) {{")
        self.indent_level += 1
        for stmt in node.body:
            self.visit(stmt)
        self.indent_level -= 1
        self.code_lines.append(f"{self.indent()}}}")

    def visit_For(self, node: ast.For):
        if isinstance(node.iter, ast.Call) and isinstance(node.iter.func, ast.Name) and node.iter.func.id == "range":
            args = node.iter.args
            target = self.visit_expr(node.target)
            if len(args) == 1:
                start = "0"
                stop = self.visit_expr(args[0])
                step = "1"
            elif len(args) == 2:
                start = self.visit_expr(args[0])
                stop = self.visit_expr(args[1])
                step = "1"
            else:
                start = self.visit_expr(args[0])
                stop = self.visit_expr(args[1])
                step = self.visit_expr(args[2])

            if step == "1":
                incr = f"{target}++"
            else:
                incr = f"{target} += {step}"

            self.code_lines.append(f"{self.indent()}for (int {target} = {start}; {target} < {stop}; {incr}) {{")
        else:
            target = self.visit_expr(node.target)
            iterable = self.visit_expr(node.iter)
            self.code_lines.append(f"{self.indent()}for (const auto& {target} : {iterable}) {{")

        self.indent_level += 1
        for stmt in node.body:
            self.visit(stmt)
        self.indent_level -= 1
        self.code_lines.append(f"{self.indent()}}}")

    def visit_Expr(self, node: ast.Expr):
        expr_str = self.visit_expr(node.value)
        if expr_str:
            self.code_lines.append(f"{self.indent()}{expr_str};")

    def visit_expr(self, node: ast.AST) -> str:
        if isinstance(node, ast.Constant):
            if isinstance(node.value, str):
                return f'"{node.value}"'
            elif isinstance(node.value, bool):
                return self.loader.literals.get("cpp_true", "true") if node.value else self.loader.literals.get("cpp_false", "false")
            elif node.value is None:
                return self.loader.literals.get("cpp_null", "nullptr")
            return str(node.value)
        elif isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.BinOp):
            left = self.visit_expr(node.left)
            right = self.visit_expr(node.right)
            op = self._get_op_symbol(node.op)
            return f"({left} {op} {right})"
        elif isinstance(node, ast.UnaryOp):
            op = self._get_unary_op_symbol(node.op)
            operand = self.visit_expr(node.operand)
            return f"{op}{operand}"
        elif isinstance(node, ast.Compare):
            left = self.visit_expr(node.left)
            comparators = []
            for op, comp in zip(node.ops, node.comparators):
                op_str = self._get_comp_symbol(op)
                comp_str = self.visit_expr(comp)
                comparators.append(f"{op_str} {comp_str}")
            return f"({left} {' '.join(comparators)})"
        elif isinstance(node, ast.BoolOp):
            op_str = self._get_op_symbol(node.op)
            values = [self.visit_expr(v) for v in node.values]
            return f"({op_str.join(values)})"
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id == "print":
                self.includes.add("#include <iostream>")
                args = [self.visit_expr(arg) for arg in node.args]
                cout_parts = " << ".join(args)
                return f"std::cout << {cout_parts} << std::endl"
            func_name = self.visit_expr(node.func)
            args = [self.visit_expr(arg) for arg in node.args]
            return f"{func_name}({', '.join(args)})"
        elif isinstance(node, ast.List):
            elts = [self.visit_expr(e) for e in node.elts]
            self.includes.add("#include <vector>")
            return f"{{{', '.join(elts)}}}"
        elif isinstance(node, ast.Attribute):
            val = self.visit_expr(node.value)
            return f"{val}.{node.attr}"
        elif isinstance(node, ast.Subscript):
            val = self.visit_expr(node.value)
            idx = self.visit_expr(node.slice)
            return f"{val}[{idx}]"
        return "<expr>"

    def _infer_expr_type(self, node: ast.AST) -> str:
        if isinstance(node, ast.Constant):
            if isinstance(node.value, bool):
                return "bool"
            if isinstance(node.value, int):
                return "int"
            if isinstance(node.value, float):
                return "double"
            if isinstance(node.value, str):
                self.includes.add("#include <string>")
                return "std::string"
        elif isinstance(node, ast.List):
            self.includes.add("#include <vector>")
            if node.elts:
                elem_type = self._infer_expr_type(node.elts[0])
                return f"std::vector<{elem_type}>"
            return "std::vector<int>"
        return "auto"

    def _map_type_annotation(self, node: ast.AST) -> str:
        if isinstance(node, ast.Name):
            return self.loader.type_annotations.get(node.id, "auto")
        return "auto"

    def _get_op_symbol(self, op: ast.AST) -> str:
        return self.loader.operators.get(type(op).__name__, "+")

    def _get_unary_op_symbol(self, op: ast.AST) -> str:
        return self.loader.operators.get(type(op).__name__, "+")

    def _get_comp_symbol(self, op: ast.AST) -> str:
        return self.loader.operators.get(type(op).__name__, "==")

class CppToPythonTranspiler:
    """Transpila C++ para Python usando unicamente mapeamentos e padroes
    sintaticos carregados de arquivos JSON externos."""

    @staticmethod
    def transpile(cpp_code: str, loader=None) -> str:
        loader = loader or LibraryConverter.get_loader()
        syntax = loader.syntax

        imports, clean_code = LibraryConverter.convert_cpp_includes(cpp_code)

        cpp_true = syntax.get("cpp_bool_true", "true")
        cpp_false = syntax.get("cpp_bool_false", "false")
        cpp_null = syntax.get("cpp_null", "nullptr")
        py_true = syntax.get("py_bool_true", "True")
        py_false = syntax.get("py_bool_false", "False")
        py_none = syntax.get("py_none", "None")

        # Normaliza blocos e instrucoes em linhas separadas
        clean_code = re.sub(r"\{\s*", "{\n", clean_code)
        clean_code = re.sub(r";\s*", ";\n", clean_code)
        clean_code = re.sub(r"\}\s*", "\n}\n", clean_code)

        clean_code = LibraryConverter.convert_functions_cpp_to_py(clean_code)

        lines = clean_code.splitlines()
        py_lines: List[str] = []
        indent_level = 0

        for imp in imports:
            py_lines.append(imp)
        if imports:
            py_lines.append("")

        i = 0
        while i < len(lines):
            line = lines[i].strip()
            i += 1

            if not line:
                continue

            # Fecha bloco
            if line == "}":
                indent_level = max(0, indent_level - 1)
                continue

            # Funcao main
            if re.match(syntax.get("main_pattern", r"(?:int|void)\s+main\s*\([^)]*\)\s*\{?"), line):
                py_lines.append(" " * (indent_level * 4) + "def main():")
                indent_level += 1
                continue

            # Declaracao de funcao generica
            func_match = re.match(syntax.get("func_decl_pattern", r"(?:(?:int|double|float|void|std::string|bool|auto|long|char|size_t)\s+)+([a-zA-Z_]\w*)\s*\(([^)]*)\)\s*\{?"), line)
            if func_match and func_match.group(1) not in ("if", "while", "for", "switch"):
                name = func_match.group(1)
                raw_params = func_match.group(2)
                params = []
                for p in raw_params.split(","):
                    p = p.strip()
                    if p:
                        parts = p.split()
                        params.append(parts[-1].lstrip("&*"))
                py_lines.append(" " * (indent_level * 4) + f"def {name}({', '.join(params)}):")
                indent_level += 1
                continue

            # For padrao C++: for (int i = 0; i < N; i++)
            for_match = re.match(syntax.get("for_standard_pattern", r"for\s*\(\s*(?:int|size_t|auto|long|char)?\s*([a-zA-Z_]\w*)\s*=\s*([^;]+);\s*\1\s*<\s*([^;]+);\s*(?:\+\+\1|\1\+\+|\1\s*\+\=\s*1)\s*\)\s*\{?"), line)
            if for_match:
                var = for_match.group(1)
                start = for_match.group(2).strip()
                stop = for_match.group(3).strip()
                range_call = f"range({stop})" if start == "0" else f"range({start}, {stop})"
                py_lines.append(" " * (indent_level * 4) + f"for {var} in {range_call}:")
                indent_level += 1
                continue

            # Range-based for: for (const auto& item : items)
            range_for_match = re.match(syntax.get("for_range_pattern", r"for\s*\([^:]*?\s+([a-zA-Z_]\w*)\s*:\s*([^)]+)\)\s*\{?"), line)
            if range_for_match:
                var = range_for_match.group(1)
                container = range_for_match.group(2).strip()
                py_lines.append(" " * (indent_level * 4) + f"for {var} in {container}:")
                indent_level += 1
                continue

            # While loop
            while_match = re.match(syntax.get("while_pattern", r"while\s*\((.*?)\)\s*\{?"), line)
            if while_match:
                cond = while_match.group(1).strip()
                cond = cond.replace("&&", "and").replace("||", "or").replace("!", "not ")
                py_lines.append(" " * (indent_level * 4) + f"while {cond}:")
                indent_level += 1
                continue

            # If / Else if / Else
            if_match = re.match(syntax.get("if_pattern", r"if\s*\((.*?)\)\s*\{?"), line)
            elif_match = re.match(syntax.get("elif_pattern", r"\}\s*else\s+if\s*\((.*?)\)\s*\{?"), line) or re.match(syntax.get("elif_pattern", r"else\s+if\s*\((.*?)\)\s*\{?"), line)
            else_match = re.match(syntax.get("else_pattern", r"\}\s*else\s*\{?"), line) or re.match(syntax.get("else_pattern", r"else\s*\{?"), line)

            if elif_match:
                cond = elif_match.group(1).strip().replace("&&", "and").replace("||", "or").replace("!", "not ")
                indent_level = max(0, indent_level - 1)
                py_lines.append(" " * (indent_level * 4) + f"elif {cond}:")
                indent_level += 1
                continue
            elif if_match:
                cond = if_match.group(1).strip().replace("&&", "and").replace("||", "or").replace("!", "not ")
                py_lines.append(" " * (indent_level * 4) + f"if {cond}:")
                indent_level += 1
                continue
            elif else_match:
                indent_level = max(0, indent_level - 1)
                py_lines.append(" " * (indent_level * 4) + "else:")
                indent_level += 1
                continue

            # Return
            ret_match = re.match(syntax.get("ret_pattern", r"return(?:\s+(.*?))?;"), line)
            if ret_match:
                ret_val = ret_match.group(1)
                if ret_val:
                    ret_val = ret_val.replace("true", py_true).replace("false", py_false).replace("nullptr", py_none)
                    py_lines.append(" " * (indent_level * 4) + f"return {ret_val}")
                else:
                    py_lines.append(" " * (indent_level * 4) + "return")
                continue

            # Atribuicao / declaracao de variavel
            assign_match = re.match(syntax.get("assign_pattern", r"(?:(?:int|double|float|std::string|auto|bool|long|char|size_t)\s+)?([a-zA-Z_]\w*)\s*=\s*(.*?);"), line)
            if assign_match:
                var = assign_match.group(1)
                val = assign_match.group(2).strip()
                val = val.replace("true", py_true).replace("false", py_false).replace("nullptr", py_none)
                py_lines.append(" " * (indent_level * 4) + f"{var} = {val}")
                continue

            # Expressao simples / chamada de funcao
            stmt = line.rstrip(";").rstrip("{").strip()
            stmt = stmt.replace("true", py_true).replace("false", py_false).replace("nullptr", py_none)
            if stmt:
                py_lines.append(" " * (indent_level * 4) + stmt)

        if "def main():" in py_lines:
            py_lines.append("\nif __name__ == '__main__':")
            py_lines.append("    main()")

        return "\n".join(py_lines)


class AstTranspiler:
    @staticmethod
    def python_to_cpp(py_code: str, loader=None) -> str:
        loader = loader or LibraryConverter.get_loader()
        cpp_includes, py_code_clean = LibraryConverter.convert_python_imports(py_code)

        try:
            tree = ast.parse(py_code_clean)
            visitor = PythonToCppVisitor(loader=loader)
            visitor.visit(tree)
            cpp_body = "\n".join(visitor.code_lines)
            all_includes = sorted(list(set(cpp_includes).union(visitor.includes)))
        except Exception:
            cpp_body = py_code_clean
            all_includes = cpp_includes

        cpp_body = LibraryConverter.convert_functions_py_to_cpp(cpp_body)

        include_header = "\n".join(all_includes)
        if include_header:
            include_header += "\n\n"

        return include_header + cpp_body

    @staticmethod
    def cpp_to_python(cpp_code: str, loader=None) -> str:
        loader = loader or LibraryConverter.get_loader()
        return CppToPythonTranspiler.transpile(cpp_code, loader=loader)

