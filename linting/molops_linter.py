#!/usr/bin/env python3
"""
MolOps Pipeline Linter -- domain-specific static analysis for cheminformatics code.

"""
from __future__ import annotations

import argparse
import ast
import sys
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path


class Severity(StrEnum):
    ERROR = "ERROR"
    WARNING = "WARNING"


@dataclass
class Violation:
    rule: str
    severity: Severity
    message: str
    line: int = 0

    def __str__(self) -> str:
        loc = f":{self.line}" if self.line else ""
        return f"  [{self.severity}] {self.rule}{loc}  {self.message}"


class PipelineLinter(ast.NodeVisitor):
    """AST-based linter for MolOps pipeline modules."""

    def __init__(self, source: str, filepath: Path) -> None:
        self.source = source
        self.filepath = filepath
        self.violations: list[Violation] = []
        self._has_module_docstring = False
        self._function_names: list[str] = []

    def lint(self) -> list[Violation]:
        tree = ast.parse(self.source, filename=str(self.filepath))
        self._check_module_docstring(tree)
        self.visit(tree)
        return self.violations

    def _add(self, rule: str, sev: Severity, msg: str, line: int = 0) -> None:
        self.violations.append(Violation(rule=rule, severity=sev, message=msg, line=line))

    def _check_module_docstring(self, tree: ast.Module) -> None:
        if (
            tree.body
            and isinstance(tree.body[0], ast.Expr)
            and isinstance(tree.body[0].value, ast.Constant)
        ):
            self._has_module_docstring = True
        else:
            self._add(
                "ML001", Severity.WARNING,
                "Module is missing a docstring explaining its purpose.",
            )

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._check_function_docstring(node)
        self._check_bare_except(node)
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:  # type: ignore[override]
        self._check_function_docstring(node)  # type: ignore[arg-type]
        self._check_bare_except(node)  # type: ignore[arg-type]
        self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign) -> None:
        """Check that physicochemical threshold constants are in valid ranges."""
        for target in node.targets:
            if isinstance(target, ast.Name) and isinstance(node.value, ast.Constant):
                name = target.id.upper()
                val = node.value.value
                if isinstance(val, (int, float)):
                    if "THRESHOLD" in name and "AD" in name:
                        if not (0.0 <= val <= 1.0):
                            self._add(
                                "ML002", Severity.ERROR,
                                f"AD threshold '{target.id}' = {val} must be between 0.0 and 1.0",
                                node.lineno,
                            )
                    if "PH" in name and not (0.0 <= val <= 14.0):
                        self._add(
                            "ML003", Severity.ERROR,
                            f"pH constant '{target.id}' = {val} must be between 0 and 14",
                            node.lineno,
                        )
        self.generic_visit(node)

    def _check_function_docstring(self, node: ast.FunctionDef) -> None:
        if not (
            node.body
            and isinstance(node.body[0], ast.Expr)
            and isinstance(node.body[0].value, ast.Constant)
        ):
            self._add(
                "ML004", Severity.WARNING,
                f"Function '{node.name}' is missing a docstring.",
                node.lineno,
            )

    def _check_bare_except(self, node: ast.FunctionDef) -> None:
        for child in ast.walk(node):
            if isinstance(child, ast.ExceptHandler) and child.type is None:
                self._add(
                    "ML005", Severity.ERROR,
                    f"Bare 'except:' in '{node.name}'. Catch specific exceptions.",
                    getattr(child, "lineno", 0),
                )


def lint_file(path: Path, strict: bool = False) -> list[Violation]:
    source = path.read_text(encoding="utf-8")
    linter = PipelineLinter(source, path)
    violations = linter.lint()
    if strict:
        for v in violations:
            if v.severity == Severity.WARNING:
                v.severity = Severity.ERROR
    return violations


def lint_path(target: Path, strict: bool = False) -> dict[Path, list[Violation]]:
    results: dict[Path, list[Violation]] = {}
    if target.is_file():
        results[target] = lint_file(target, strict)
    elif target.is_dir():
        for f in sorted(target.rglob("*.py")):
            if f.name.startswith("_"):
                continue
            results[f] = lint_file(f, strict)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="MolOps pipeline linter")
    parser.add_argument("targets", nargs="+", metavar="PATH")
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    total_errors = 0
    total_warnings = 0

    for t in args.targets:
        results = lint_path(Path(t), strict=args.strict)
        for filepath, violations in results.items():
            errors = [v for v in violations if v.severity == Severity.ERROR]
            warnings = [v for v in violations if v.severity == Severity.WARNING]
            total_errors += len(errors)
            total_warnings += len(warnings)
            if violations or not args.quiet:
                status = "FAIL" if errors else ("WARN" if warnings else "OK  ")
                print(f"\n{status}  {filepath}")
                for v in violations:
                    print(str(v))

    print(f"\n{'--'*30}")
    print(f"MolOps Lint: {total_errors} error(s), {total_warnings} warning(s)")

    if total_errors:
        sys.exit(1)


if __name__ == "__main__":
    main()