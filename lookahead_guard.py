"""
Lookahead Bias Static Analyzer & Guard.

Hardcoded protection against future-peeking in algorithmic trading strategies.
Parses the Abstract Syntax Tree (AST) of user/AI-submitted Python strategies
and strictly rejects any constructs that leak future information into past bars.
"""

import ast
from typing import List, Tuple


class LookaheadBiasError(Exception):
    """Raised when strategy code contains lookahead bias patterns."""
    def __init__(self, message: str, line: int = None, code_snippet: str = None):
        self.message = message
        self.line = line
        self.code_snippet = code_snippet
        super().__init__(f"Line {line}: {message}" if line else message)


class LookaheadASTVisitor(ast.NodeVisitor):
    """
    Scans Python AST for operations known to introduce lookahead bias:
    1. Negative shifts: .shift(-n), .pct_change(-n)
    2. Centered rolling windows: .rolling(..., center=True)
    3. Backward fills: .bfill(), .fillna(method='bfill')
    4. Forward indexing in loops: iloc[i + n], [idx + n]
    """

    def __init__(self, source_lines: List[str]):
        self.source_lines = source_lines
        self.violations: List[Tuple[int, str, str]] = []

    def _get_snippet(self, lineno: int) -> str:
        if 1 <= lineno <= len(self.source_lines):
            return self.source_lines[lineno - 1].strip()
        return ""

    def visit_Call(self, node: ast.Call):
        func_name = ""
        # Check method calls like df['close'].shift(-1)
        if isinstance(node.func, ast.Attribute):
            func_name = node.func.attr
        elif isinstance(node.func, ast.Name):
            func_name = node.func.id

        # 1. Check shift / pct_change / diff with negative argument
        if func_name in ('shift', 'pct_change', 'diff'):
            # Positional arguments
            if node.args:
                first_arg = node.args[0]
                if isinstance(first_arg, ast.UnaryOp) and isinstance(first_arg.op, ast.USub):
                    self.violations.append((
                        node.lineno,
                        f"Prohibited negative shift/offset in '{func_name}()'. This looks into future bars!",
                        self._get_snippet(node.lineno)
                    ))
                elif isinstance(first_arg, ast.Constant) and isinstance(first_arg.value, (int, float)) and first_arg.value < 0:
                    self.violations.append((
                        node.lineno,
                        f"Prohibited negative offset ({first_arg.value}) in '{func_name}()'.",
                        self._get_snippet(node.lineno)
                    ))
            # Keyword arguments like shift(periods=-1)
            for kw in node.keywords:
                if kw.arg in ('periods', 'n', 'step'):
                    if isinstance(kw.value, ast.UnaryOp) and isinstance(kw.value.op, ast.USub):
                        self.violations.append((
                            node.lineno,
                            f"Prohibited negative {kw.arg} in '{func_name}()'. Future data leak detected!",
                            self._get_snippet(node.lineno)
                        ))
                    elif isinstance(kw.value, ast.Constant) and isinstance(kw.value.value, (int, float)) and kw.value.value < 0:
                        self.violations.append((
                            node.lineno,
                            f"Prohibited negative {kw.arg} ({kw.value.value}) in '{func_name}()'.",
                            self._get_snippet(node.lineno)
                        ))

        # Check np.roll with negative shift
        if func_name == 'roll' or (isinstance(node.func, ast.Attribute) and node.func.attr == 'roll'):
            if len(node.args) >= 2:
                shift_arg = node.args[1]
                if (isinstance(shift_arg, ast.UnaryOp) and isinstance(shift_arg.op, ast.USub)) or \
                   (isinstance(shift_arg, ast.Constant) and isinstance(shift_arg.value, (int, float)) and shift_arg.value < 0):
                    self.violations.append((
                        node.lineno,
                        "Prohibited negative shift in 'roll()'. Future data leak detected!",
                        self._get_snippet(node.lineno)
                    ))

        # 2. Check rolling(..., center=True)
        if func_name == 'rolling':
            for kw in node.keywords:
                if kw.arg == 'center':
                    if isinstance(kw.value, ast.Constant) and kw.value.value is True:
                        self.violations.append((
                            node.lineno,
                            "Prohibited 'center=True' in rolling window. Centered windows use future bars!",
                            self._get_snippet(node.lineno)
                        ))

        # 3. Check bfill() / backward fills
        if func_name in ('bfill', 'backfill'):
            self.violations.append((
                node.lineno,
                f"Prohibited '{func_name}()'. Backwards filling propagates future prices into the past.",
                self._get_snippet(node.lineno)
            ))

        if func_name == 'fillna':
            for kw in node.keywords:
                if kw.arg == 'method':
                    if isinstance(kw.value, ast.Constant) and str(kw.value.value).lower() in ('bfill', 'backfill'):
                        self.violations.append((
                            node.lineno,
                            "Prohibited 'fillna(method=\"bfill\")'. Future values cannot backfill past candles.",
                            self._get_snippet(node.lineno)
                        ))

        self.generic_visit(node)

    def visit_Subscript(self, node: ast.Subscript):
        # 4. Check reverse slicing e.g. [::-1] (reversing chronological time)
        slice_node = node.slice
        if isinstance(slice_node, ast.Slice):
            if slice_node.step is not None:
                step_val = slice_node.step
                if (isinstance(step_val, ast.UnaryOp) and isinstance(step_val.op, ast.USub)) or \
                   (isinstance(step_val, ast.Constant) and isinstance(step_val.value, (int, float)) and step_val.value < 0):
                    self.violations.append((
                        node.lineno,
                        "Prohibited negative step slice (e.g. [::-1]). Reversing time series leaks future data!",
                        self._get_snippet(node.lineno)
                    ))

        # 5. Check forward indexing e.g. iloc[i + 1] or df.iloc[current + n]
        if isinstance(node.value, ast.Attribute) and node.value.attr in ('iloc', 'loc'):
            if isinstance(slice_node, ast.BinOp):
                if isinstance(slice_node.op, ast.Add):
                    if isinstance(slice_node.right, ast.Constant) and isinstance(slice_node.right.value, (int, float)) and slice_node.right.value > 0:
                        self.violations.append((
                            node.lineno,
                            f"Prohibited forward indexing in {node.value.attr}. Accessing future bars is forbidden.",
                            self._get_snippet(node.lineno)
                        ))

        self.generic_visit(node)


def validate_strategy_code(code_str: str) -> Tuple[bool, List[str]]:
    """
    Validates Python strategy source code for lookahead bias.

    Returns:
        (is_valid, list_of_error_messages)
    """
    try:
        tree = ast.parse(code_str)
    except SyntaxError as e:
        return False, [f"Python Syntax Error at line {e.lineno}: {e.msg}"]

    lines = code_str.splitlines()
    visitor = LookaheadASTVisitor(lines)
    visitor.visit(tree)

    if visitor.violations:
        errors = [
            f"Lookahead Bias Violation (Line {lineno}): {reason}\n   >> {snippet}"
            for lineno, reason, snippet in visitor.violations
        ]
        return False, errors

    return True, []


def assert_no_lookahead(code_str: str):
    """
    Raises LookaheadBiasError if any violation is found.
    """
    is_valid, errors = validate_strategy_code(code_str)
    if not is_valid:
        raise LookaheadBiasError("\n".join(errors))
