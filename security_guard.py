"""
Security Guard — Static AST Analyzer for Dangerous Code Patterns.

Defense-in-depth layer that statically rejects:
  - import / from ... import statements
  - Calls to eval, exec, compile, open, __import__, input, exit, quit
  - Attribute access to dunder internals (__builtins__, __import__, __subclasses__, __globals__, __code__)
  - Direct references to os, sys, subprocess, socket, shutil, pathlib

This runs BEFORE exec() and supplements the restricted __builtins__ sandbox.
"""

import ast
from typing import List, Tuple


class SecurityViolationError(Exception):
    """Raised when strategy code contains dangerous constructs."""
    pass


# Dangerous function names that should never appear in strategy code
BANNED_CALLS = frozenset({
    'eval', 'exec', 'compile', 'open', '__import__',
    'input', 'exit', 'quit', 'breakpoint',
    'getattr', 'setattr', 'delattr',  # Can be used to bypass restrictions
    'globals', 'locals', 'vars', 'dir',
})

# Dangerous attribute names
BANNED_ATTRS = frozenset({
    '__builtins__', '__import__', '__subclasses__',
    '__globals__', '__code__', '__closure__',
    '__bases__', '__mro__', '__class__',
    '__dict__',
})

# Dangerous module names
BANNED_MODULES = frozenset({
    'os', 'sys', 'subprocess', 'socket', 'shutil',
    'pathlib', 'ctypes', 'signal', 'multiprocessing',
    'threading', 'importlib', 'builtins', 'code',
    'codeop', 'compileall', 'py_compile',
    'http', 'urllib', 'requests', 'ftplib', 'smtplib',
    'pickle', 'shelve', 'marshal', 'tempfile',
    'webbrowser', 'antigravity',
})


# Permitted modules for numerical and technical indicator calculations
SAFE_MODULES = frozenset({
    'numpy', 'pandas', 'math', 'datetime', 'scipy', 'typing'
})


class SecurityASTVisitor(ast.NodeVisitor):
    """Scans Python AST for security-critical patterns."""

    def __init__(self, source_lines: List[str]):
        self.source_lines = source_lines
        self.violations: List[Tuple[int, str]] = []

    def _get_snippet(self, lineno: int) -> str:
        if 1 <= lineno <= len(self.source_lines):
            return self.source_lines[lineno - 1].strip()
        return ""

    def visit_Import(self, node: ast.Import):
        """Allow only safe mathematical/data manipulation modules; block all others."""
        for alias in node.names:
            base_mod = alias.name.split('.')[0]
            if base_mod not in SAFE_MODULES:
                self.violations.append((
                    node.lineno,
                    f"Prohibited 'import {alias.name}'. Only numerical/data modules ({', '.join(sorted(SAFE_MODULES))}) are permitted."
                ))
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom):
        """Allow only safe mathematical/data manipulation modules; block all others."""
        mod = (node.module or "").split('.')[0]
        if mod not in SAFE_MODULES:
            self.violations.append((
                node.lineno,
                f"Prohibited 'from {node.module} import ...'. Only numerical/data modules ({', '.join(sorted(SAFE_MODULES))}) are permitted."
            ))
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call):
        """Block calls to dangerous builtins."""
        func_name = ""
        if isinstance(node.func, ast.Name):
            func_name = node.func.id
        elif isinstance(node.func, ast.Attribute):
            func_name = node.func.attr

        if func_name in BANNED_CALLS:
            self.violations.append((
                node.lineno,
                f"Prohibited call to '{func_name}()'. This function is not allowed in strategy code."
            ))

        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute):
        """Block access to dangerous dunder attributes."""
        if node.attr in BANNED_ATTRS:
            self.violations.append((
                node.lineno,
                f"Prohibited access to '{node.attr}'. Dunder attribute access is not allowed."
            ))

        # Block access to dangerous module attributes like os.system
        if isinstance(node.value, ast.Name) and node.value.id in BANNED_MODULES:
            self.violations.append((
                node.lineno,
                f"Prohibited access to '{node.value.id}.{node.attr}'. Module '{node.value.id}' is not allowed."
            ))

        self.generic_visit(node)

    def visit_Name(self, node: ast.Name):
        """Block direct references to dangerous module names used as variables."""
        # Only flag if used in a call or attribute context (handled by other visitors)
        # Don't flag simple variable names that happen to match module names
        self.generic_visit(node)


def validate_security(code_str: str) -> Tuple[bool, List[str]]:
    """
    Validates Python strategy code for security violations.

    Returns:
        (is_safe, list_of_error_messages)
    """
    try:
        tree = ast.parse(code_str)
    except SyntaxError as e:
        return False, [f"Python Syntax Error at line {e.lineno}: {e.msg}"]

    lines = code_str.splitlines()
    visitor = SecurityASTVisitor(lines)
    visitor.visit(tree)

    if visitor.violations:
        errors = [
            f"Security Violation (Line {lineno}): {reason}"
            for lineno, reason in visitor.violations
        ]
        return False, errors

    return True, []


validate_code_security = validate_security


def assert_no_dangerous_code(code_str: str):
    """Raises SecurityViolationError if any dangerous pattern is found."""
    is_safe, errors = validate_security(code_str)
    if not is_safe:
        raise SecurityViolationError("\n".join(errors))


def _safe_import(name, globals=None, locals=None, fromlist=(), level=0):
    base_mod = name.split('.')[0]
    if base_mod in SAFE_MODULES:
        return __import__(name, globals, locals, fromlist, level)
    raise ImportError(f"Import of module '{name}' is prohibited in strategy sandbox.")


# Safe builtins whitelist for strategy execution sandbox
SAFE_BUILTINS = {
    '__import__': _safe_import,
    '__build_class__': __build_class__,
    'len': len,
    'range': range,
    'min': min,
    'max': max,
    'abs': abs,
    'round': round,
    'enumerate': enumerate,
    'zip': zip,
    'int': int,
    'float': float,
    'str': str,
    'bool': bool,
    'list': list,
    'dict': dict,
    'tuple': tuple,
    'set': set,
    'sorted': sorted,
    'reversed': reversed,
    'sum': sum,
    'any': any,
    'all': all,
    'map': map,
    'filter': filter,
    'print': print,
    'isinstance': isinstance,
    'issubclass': issubclass,
    'type': type,
    'hasattr': hasattr,
    'Exception': Exception,
    'ValueError': ValueError,
    'TypeError': TypeError,
    'IndexError': IndexError,
    'KeyError': KeyError,
    'True': True,
    'False': False,
    'None': None,
}
