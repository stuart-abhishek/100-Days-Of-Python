#!/usr/bin/env python3
# Day 16 — MiniML: Mini functional language with Hindley-Milner type inference
# Author: Stuart Abhishek
# Concepts: parsing, AST, interpreter, Hindley-Milner type inference, unification, let-polymorphism
#
# Usage:
#   python day16_miniml.py         # interactive REPL
#   python day16_miniml.py file.ml # run expressions in a file (one per line or a single expression)
#
# Language syntax (very small):
#   Integers: 42
#   Booleans: true, false
#   Variables: x, foo
#   Lambda (anonymous): \x -> expr        (backslash for lambda)
#   Application: f x y                     (juxtaposition, left-assoc)
#   Let: let name = expr in expr
#   If: if cond then expr else expr
#   Parentheses: ( ... )
#   Comments: # rest-of-line
#
# Examples:
#   let id = \x -> x in id 5
#   let map = \f -> \xs -> ...  (compose small examples)
#
# This is intentionally educational & concise; not optimized for performance.

from __future__ import annotations
import re
import sys
from dataclasses import dataclass
from typing import Dict, List, Tuple, Union, Optional, Set
import itertools
import copy

# -------------------------
# Lexer / Parser (simple)
# -------------------------

TOKEN_RE = re.compile(r"""
    \s*(?:
      (?P<int>\d+)
      |(?P<bool>true|false)
      |(?P<lambda>\\)              # backslash for lambda
      |(?P<arrow>->)
      |(?P<let>let\b)
      |(?P<in>in\b)
      |(?P<if>if\b)
      |(?P<then>then\b)
      |(?P<else>else\b)
      |(?P<ident>[a-zA-Z_][a-zA-Z0-9_]*)
      |(?P<lparen>\()
      |(?P<rparen>\))
      |(?P<equals>=)
      |(?P<comment>\#.*)
      |(?P<unknown>.)
    )
""", re.VERBOSE)

@dataclass
class Token:
    kind: str
    value: str

def lex(s: str) -> List[Token]:
    pos = 0
    tokens = []
    while pos < len(s):
        m = TOKEN_RE.match(s, pos)
        if not m:
            raise SyntaxError(f"Unexpected input at {pos}: {s[pos:]}")
        pos = m.end()
        kind = m.lastgroup
        if kind == "comment":
            continue
        val = m.group(kind)
        tokens.append(Token(kind, val))
    return tokens

# AST nodes
@dataclass
class Expr:
    pass

@dataclass
class EInt(Expr):
    value: int

@dataclass
class EBool(Expr):
    value: bool

@dataclass
class EVar(Expr):
    name: str

@dataclass
class ELam(Expr):
    arg: str
    body: Expr

@dataclass
class EApp(Expr):
    func: Expr
    arg: Expr

@dataclass
class ELet(Expr):
    name: str
    value: Expr
    body: Expr

@dataclass
class EIf(Expr):
    cond: Expr
    then: Expr
    els: Expr

# Parser (recursive descent)
class Parser:
    def __init__(self, tokens: List[Token]):
        self.tokens = tokens
        self.i = 0

    def peek(self) -> Optional[Token]:
        return self.tokens[self.i] if self.i < len(self.tokens) else None

    def pop(self) -> Token:
        t = self.peek()
        if t is None:
            raise SyntaxError("Unexpected EOF")
        self.i += 1
        return t

    def optional(self, kind: str) -> Optional[Token]:
        t = self.peek()
        if t and t.kind == kind:
            return self.pop()
        return None

    def expect(self, kind: str) -> Token:
        t = self.pop()
        if t.kind != kind:
            raise SyntaxError(f"Expected {kind} got {t.kind}")
        return t

    def parse(self) -> Expr:
        return self.parse_expr()

    def parse_expr(self) -> Expr:
        # parse let / if / lambda or application/atom
        t = self.peek()
        if t and t.kind == "let":
            self.pop()
            name = self.expect("ident").value
            self.expect("equals")
            val = self.parse_expr()
            self.expect("in")
            body = self.parse_expr()
            return ELet(name, val, body)
        if t and t.kind == "if":
            self.pop()
            cond = self.parse_expr()
            self.expect("then")
            th = self.parse_expr()
            self.expect("else")
            el = self.parse_expr()
            return EIf(cond, th, el)
        # lambda
        if t and t.kind == "lambda":
            self.pop()
            arg = self.expect("ident").value
            self.expect("arrow")
            body = self.parse_expr()
            return ELam(arg, body)
        # otherwise parse application (left-assoc)
        return self.parse_app()

    def parse_app(self) -> Expr:
        node = self.parse_atom()
        while True:
            t = self.peek()
            # application when next token starts an atom
            if t and t.kind in ("int", "bool", "ident", "lparen", "lambda"):
                rhs = self.parse_atom()
                node = EApp(node, rhs)
            else:
                break
        return node

    def parse_atom(self) -> Expr:
        t = self.peek()
        if t is None:
            raise SyntaxError("Unexpected EOF at atom")
        if t.kind == "int":
            self.pop()
            return EInt(int(t.value))
        if t.kind == "bool":
            self.pop()
            return EBool(t.value == "true")
        if t.kind == "ident":
            self.pop()
            return EVar(t.value)
        if t.kind == "lparen":
            self.pop()
            e = self.parse_expr()
            self.expect("rparen")
            return e
        if t.kind == "lambda":
            return self.parse_expr()  # lambda handled in parse_expr
        raise SyntaxError(f"Unexpected token in atom: {t.kind} ({t.value})")

def parse_program(s: str) -> Expr:
    tokens = lex(s)
    p = Parser(tokens)
    ast = p.parse()
    if p.peek() is not None:
        raise SyntaxError("Extra tokens after expression")
    return ast

# -------------------------
# Interpreter (environment)
# -------------------------

Value = Union[int, bool, "Closure"]

@dataclass
class Closure:
    arg: str
    body: Expr
    env: Dict[str, Value]

def eval_expr(expr: Expr, env: Dict[str, Value]) -> Value:
    if isinstance(expr, EInt):
        return expr.value
    if isinstance(expr, EBool):
        return expr.value
    if isinstance(expr, EVar):
        if expr.name in env:
            return env[expr.name]
        raise NameError(f"Unbound variable {expr.name}")
    if isinstance(expr, ELam):
        return Closure(expr.arg, expr.body, env.copy())
    if isinstance(expr, EApp):
        f = eval_expr(expr.func, env)
        a = eval_expr(expr.arg, env)
        if not isinstance(f, Closure):
            raise TypeError("Attempt to call non-function")
        new_env = f.env.copy()
        new_env[f.arg] = a
        return eval_expr(f.body, new_env)
    if isinstance(expr, ELet):
        val = eval_expr(expr.value, env)
        new_env = env.copy()
        new_env[expr.name] = val
        return eval_expr(expr.body, new_env)
    if isinstance(expr, EIf):
        c = eval_expr(expr.cond, env)
        if not isinstance(c, bool):
            raise TypeError("Condition must be boolean")
        return eval_expr(expr.then if c else expr.els, env)
    raise RuntimeError("Unknown AST node in eval")

# -------------------------
# Hindley-Milner Type System
# -------------------------

# Types
@dataclass(frozen=True)
class Type:
    pass

@dataclass(frozen=True)
class TInt(Type):
    pass

@dataclass(frozen=True)
class TBool(Type):
    pass

@dataclass
class TVar(Type):
    id: int

@dataclass
class TFun(Type):
    a: Type
    b: Type

# Type schemes for polymorphism
@dataclass
class Scheme:
    vars: List[int]
    t: Type

# Type environment maps var -> Scheme
TypeEnv = Dict[str, Scheme]

# Type variable generator
class TVarGen:
    def __init__(self):
        self._counter = 0
    def fresh(self) -> TVar:
        v = TVar(self._counter)
        self._counter += 1
        return v

# Substitutions: map type var id -> Type
Subst = Dict[int, Type]

def apply_subst(t: Type, s: Subst) -> Type:
    if isinstance(t, TInt) or isinstance(t, TBool):
        return t
    if isinstance(t, TVar):
        if t.id in s:
            return apply_subst(s[t.id], s)
        return t
    if isinstance(t, TFun):
        return TFun(apply_subst(t.a, s), apply_subst(t.b, s))
    raise RuntimeError("Unknown type in apply_subst")

def apply_subst_scheme(sch: Scheme, s: Subst) -> Scheme:
    # avoid substituting bound vars
    s2 = {k: v for k, v in s.items() if k not in sch.vars}
    return Scheme(sch.vars, apply_subst(sch.t, s2))

def compose_subst(s1: Subst, s2: Subst) -> Subst:
    # s1 after s2: apply s1 to values in s2, then merge
    s2_applied = {k: apply_subst(v, s1) for k, v in s2.items()}
    res = s1.copy()
    res.update(s2_applied)
    return res

def free_type_vars(t: Type) -> Set[int]:
    if isinstance(t, TInt) or isinstance(t, TBool):
        return set()
    if isinstance(t, TVar):
        return {t.id}
    if isinstance(t, TFun):
        return free_type_vars(t.a) | free_type_vars(t.b)
    return set()

def free_type_vars_scheme(sch: Scheme) -> Set[int]:
    return free_type_vars(sch.t) - set(sch.vars)

def free_type_vars_env(env: TypeEnv) -> Set[int]:
    s = set()
    for sch in env.values():
        s |= free_type_vars_scheme(sch)
    return s

# Unification
class UnifyError(Exception):
    pass

def occurs_check(var: int, t: Type) -> bool:
    return var in free_type_vars(t)

def unify(t1: Type, t2: Type) -> Subst:
    if isinstance(t1, TVar):
        return unify_var(t1.id, t2)
    if isinstance(t2, TVar):
        return unify_var(t2.id, t1)
    if isinstance(t1, TInt) and isinstance(t2, TInt):
        return {}
    if isinstance(t1, TBool) and isinstance(t2, TBool):
        return {}
    if isinstance(t1, TFun) and isinstance(t2, TFun):
        s1 = unify(t1.a, t2.a)
        s2 = unify(apply_subst(t1.b, s1), apply_subst(t2.b, s1))
        return compose_subst(s2, s1)
    raise UnifyError(f"Cannot unify {show_type(t1)} with {show_type(t2)}")

def unify_var(var: int, t: Type) -> Subst:
    if isinstance(t, TVar) and t.id == var:
        return {}
    if occurs_check(var, t):
        raise UnifyError("Occurs check failed")
    return {var: t}

# Generalize & instantiate
def generalize(env: TypeEnv, t: Type) -> Scheme:
    env_ftv = free_type_vars_env(env)
    tvs = list(free_type_vars(t) - env_ftv)
    return Scheme(tvs, t)

def instantiate(sch: Scheme, gen: TVarGen) -> Tuple[Type, Subst]:
    # replace bound vars with fresh type vars
    subst: Subst = {}
    for vid in sch.vars:
        subst[vid] = gen.fresh()
    t = apply_subst(sch.t, subst)
    # convert TVar objects to their ids mapping in subst? we used TVar instances directly
    # but our TVar uses ids; fresh returns TVar -> fine
    return t, {}

# Type inference (Algorithm W)
def infer(expr: Expr, env: TypeEnv, gen: TVarGen) -> Tuple[Subst, Type]:
    if isinstance(expr, EInt):
        return {}, TInt()
    if isinstance(expr, EBool):
        return {}, TBool()
    if isinstance(expr, EVar):
        if expr.name not in env:
            raise NameError(f"Unbound variable (in typing): {expr.name}")
        sch = env[expr.name]
        t_inst, _ = instantiate(sch, gen)
        return {}, t_inst
    if isinstance(expr, ELam):
        tv = gen.fresh()
        new_env = env.copy()
        new_env[expr.arg] = Scheme([], tv)
        s1, t1 = infer(expr.body, new_env, gen)
        return s1, TFun(apply_subst(tv, s1), t1)
    if isinstance(expr, EApp):
        s1, t1 = infer(expr.func, env, gen)
        env2 = {k: apply_subst_scheme(v, s1) for k, v in env.items()}
        s2, t2 = infer(expr.arg, env2, gen)
        tv = gen.fresh()
        # unify t1 (after s2) with t_arg -> tv
        s3 = unify(apply_subst(t1, s2), TFun(t2, tv))
        s = compose_subst(s3, compose_subst(s2, s1))
        return s, apply_subst(tv, s3)
    if isinstance(expr, ELet):
        # infer value
        s1, t1 = infer(expr.value, env, gen)
        env2 = {k: apply_subst_scheme(v, s1) for k, v in env.items()}
        sch = generalize(env2, t1)
        new_env = env2.copy()
        new_env[expr.name] = sch
        s2, t2 = infer(expr.body, new_env, gen)
        s = compose_subst(s2, s1)
        return s, t2
    if isinstance(expr, EIf):
        s1, tc = infer(expr.cond, env, gen)
        env2 = {k: apply_subst_scheme(v, s1) for k, v in env.items()}
        s2, tt = infer(expr.then, env2, gen)
        env3 = {k: apply_subst_scheme(v, compose_subst(s2, s1)) for k, v in env.items()}
        s3, te = infer(expr.els, env3, gen)
        s4 = unify(apply_subst(tc, compose_subst(s3, compose_subst(s2, s1))), TBool())
        s5 = unify(apply_subst(tt, s4), apply_subst(te, s4))
        s = compose_subst(s5, compose_subst(s4, compose_subst(s3, compose_subst(s2, s1))))
        return s, apply_subst(tt, s)
    raise RuntimeError("Unknown AST node in infer")

# Pretty printing types
def show_type(t: Type) -> str:
    if isinstance(t, TInt): return "Int"
    if isinstance(t, TBool): return "Bool"
    if isinstance(t, TVar): return f"t{t.id}"
    if isinstance(t, TFun):
        a = show_type(t.a)
        b = show_type(t.b)
        return f"({a} -> {b})"
    return "?"

# -------------------------
# Top-level helpers & REPL
# -------------------------

def infer_and_print(s: str, env: TypeEnv):
    ast = parse_program(s)
    gen = TVarGen()
    try:
        subst, t = infer(ast, env, gen)
        t = apply_subst(t, subst)
        print("Type:", show_type(t))
    except Exception as e:
        print("Type error:", e)

def eval_and_print(s: str, envv: Dict[str, Value]):
    ast = parse_program(s)
    try:
        v = eval_expr(ast, envv)
        if isinstance(v, Closure):
            print("<function>")
        else:
            print(v)
    except Exception as e:
        print("Runtime error:", e)

def repl():
    print("MiniML REPL — type :q to quit, :t <expr> to infer type")
    tenv: TypeEnv = {}
    venv: Dict[str, Value] = {}

    # add some base functions in environment (e.g., add)
    # We'll model add as a curried function: \x -> \y -> x + y
    # But to keep interpreter small, we can add builtins in REPL via let-binding parse
    while True:
        try:
            line = input("MiniML> ").strip()
        except EOFError:
            break
        if not line:
            continue
        if line == ":q":
            break
        if line.startswith(":t "):
            expr = line[3:].strip()
            try:
                infer_and_print(expr, tenv)
            except Exception as e:
                print("Error:", e)
            continue
        if line.startswith(":e "):
            expr = line[3:].strip()
            try:
                eval_and_print(expr, venv)
            except Exception as e:
                print("Error:", e)
            continue
        # default: try to run and infer
        try:
            # infer type and print
            ast = parse_program(line)
            gen = TVarGen()
            try:
                subst, t = infer(ast, tenv, gen)
                t = apply_subst(t, subst)
                print("Type:", show_type(t))
            except Exception as e:
                print("Type error:", e)
            # evaluate
            try:
                v = eval_expr(ast, venv)
                if isinstance(v, Closure):
                    print("<function>")
                else:
                    print(v)
            except Exception as e:
                print("Runtime error:", e)
        except Exception as e:
            print("Parse error:", e)

def run_file(path: str):
    with open(path, "r", encoding="utf-8") as f:
        txt = f.read()
    # allow multiple lines (will parse each non-empty)
    envv = {}
    tenv: TypeEnv = {}
    for line in txt.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        print(">>", line)
        try:
            gen = TVarGen()
            ast = parse_program(line)
            s, t = infer(ast, tenv, gen)
            t = apply_subst(t, s)
            print("Type:", show_type(t))
        except Exception as e:
            print("Type error:", e)
        try:
            v = eval_expr(ast, envv)
            print("Value:", "<function>" if isinstance(v, Closure) else v)
            # If it's let-binding, we must update envs, but we used ELet that evaluates in-place
        except Exception as e:
            print("Runtime error:", e)

# -------------------------
# Minimal tests / examples
# -------------------------

SAMPLE_PROGRAMS = [
    ("let id = \\x -> x in id 5", "Int"),
    ("let id = \\x -> x in id true", "Bool"),
    ("let const = \\x -> \\y -> x in const 5 true", "Int"),
    ("let compose = \\f -> \\g -> \\x -> f (g x) in compose", "(t0 -> t1) -> (t2 -> t0) -> t2 -> t1"),
    ("if true then 1 else 2", "Int"),
    ("let twice = \\f -> \\x -> f (f x) in twice (\\y -> y + 1) 5", "Int"),
]

def run_tests():
    print("Running sample tests...")
    for src, expected in SAMPLE_PROGRAMS:
        print("SRC:", src)
        try:
            ast = parse_program(src)
            gen = TVarGen()
            s, t = infer(ast, {}, gen)
            t = apply_subst(t, s)
            print("INFERRED:", show_type(t))
        except Exception as e:
            print("TYPE ERROR:", e)
        try:
            val = eval_expr(ast, {})
            print("EVAL:", val if not isinstance(val, Closure) else "<function>")
        except Exception as e:
            print("EVAL ERROR:", e)
        print("-" * 40)

# -------------------------
# Entry point
# -------------------------

if __name__ == "__main__":
    if len(sys.argv) == 1:
        repl()
    elif sys.argv[1] == "--test":
        run_tests()
    else:
        run_file(sys.argv[1])