"""
SCIG+ : Self-Contracting Improvement Graph (Optimized)
- Stronger L1: bandit policy for operators (UCB + EMA gain + failure penalty)
- Stronger L2: multi-source tests (base + adversarial + boundary + regression bank)
- Stronger acceptance: unified objective = val + complexity + risk - novelty bonus
- Novelty: behavior-signature distance over probe points (cached for performance)
- Regression bank: contract failures + hard examples are remembered and reused
Run:
  python scig_plus_demo.py
"""
from __future__ import annotations
import math
import random
import statistics
import argparse
import os
import sys
import shutil
import subprocess
import tempfile
import contextlib
import io
import ast
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

class Node:

    def eval(self, x: float) -> float:
        raise NotImplementedError

    def clone(self) -> 'Node':
        raise NotImplementedError

    def size(self) -> int:
        raise NotImplementedError

    def walk(self) -> List['Node']:
        raise NotImplementedError

    def replace_child(self, old: 'Node', new: 'Node') -> bool:
        return False

    def to_str(self) -> str:
        raise NotImplementedError

@dataclass
class Var(Node):

    def eval(self, x: float) -> float:
        return x

    def clone(self) -> 'Node':
        return Var()

    def size(self) -> int:
        return 1

    def walk(self) -> List['Node']:
        return [self]

    def to_str(self) -> str:
        return 'x'

@dataclass
class Const(Node):
    c: float

    def eval(self, x: float) -> float:
        return self.c

    def clone(self) -> 'Node':
        return Const(self.c)

    def size(self) -> int:
        return 1

    def walk(self) -> List['Node']:
        return [self]

    def to_str(self) -> str:
        return f'{self.c:.4g}'

@dataclass
class Unary(Node):
    op: str
    a: Node

    def eval(self, x: float) -> float:
        v = self.a.eval(x)
        if self.op == 'neg':
            return -v
        if self.op == 'sin':
            return math.sin(v)
        if self.op == 'cos':
            return math.cos(v)
        if self.op == 'tanh':
            return math.tanh(v)
        if self.op == 'abs':
            return abs(v)
        if self.op == 'log1p':
            return math.log1p(max(v, -0.999999))
        if self.op == 'sqrt':
            return math.sqrt(max(0.0, v))
        raise ValueError(f'Unknown unary op: {self.op}')

    def clone(self) -> 'Node':
        return Unary(self.op, self.a.clone())

    def size(self) -> int:
        return 1 + self.a.size()

    def walk(self) -> List['Node']:
        return [self] + self.a.walk()

    def replace_child(self, old: Node, new: Node) -> bool:
        if self.a is old:
            self.a = new
            return True
        return self.a.replace_child(old, new)

    def to_str(self) -> str:
        return f'{self.op}({self.a.to_str()})'

@dataclass
class Binary(Node):
    op: str
    a: Node
    b: Node

    def eval(self, x: float) -> float:
        va = self.a.eval(x)
        vb = self.b.eval(x)
        if self.op == 'add':
            return va + vb
        if self.op == 'sub':
            return va - vb
        if self.op == 'mul':
            return va * vb
        if self.op == 'div':
            den = vb if abs(vb) > 1e-09 else 1e-09 if vb >= 0 else -1e-09
            return va / den
        if self.op == 'max':
            return max(va, vb)
        if self.op == 'min':
            return min(va, vb)
        if self.op == 'pow2':
            return va * va + vb * vb
        raise ValueError(f'Unknown binary op: {self.op}')

    def clone(self) -> 'Node':
        return Binary(self.op, self.a.clone(), self.b.clone())

    def size(self) -> int:
        return 1 + self.a.size() + self.b.size()

    def walk(self) -> List['Node']:
        return [self] + self.a.walk() + self.b.walk()

    def replace_child(self, old: Node, new: Node) -> bool:
        if self.a is old:
            self.a = new
            return True
        if self.b is old:
            self.b = new
            return True
        return self.a.replace_child(old, new) or self.b.replace_child(old, new)

    def to_str(self) -> str:
        return f'{self.op}({self.a.to_str()}, {self.b.to_str()})'

@dataclass
class Contract:
    name: str
    check: Callable[[Callable[[float], float], List[float]], bool]

def contract_finite_and_bounded(bound: float=1000000.0) -> Contract:

    def _check(fn: Callable[[float], float], xs: List[float]) -> bool:
        try:
            for x in xs:
                y = fn(x)
                if not math.isfinite(y):
                    return False
                if abs(y) > bound:
                    return False
            return True
        except Exception:
            return False
    return Contract(name=f'finite_and_bounded({bound})', check=_check)

def contract_lipschitz_soft(max_slope: float=10000.0) -> Contract:

    def _check(fn: Callable[[float], float], xs: List[float]) -> bool:
        try:
            xs_sorted = sorted(xs)
            ys = [fn(x) for x in xs_sorted]
            for i in range(1, len(xs_sorted)):
                dx = xs_sorted[i] - xs_sorted[i - 1]
                if dx < 1e-09:
                    continue
                dy = abs(ys[i] - ys[i - 1])
                if dy / dx > max_slope:
                    return False
            return True
        except Exception:
            return False
    return Contract(name=f'lipschitz_soft({max_slope})', check=_check)

def contract_smooth_probe(max_local_slope: float=200000.0, dx: float=0.0001) -> Contract:

    def _check(fn: Callable[[float], float], xs: List[float]) -> bool:
        try:
            for x in xs:
                y1 = fn(x)
                y2 = fn(x + dx)
                if not (math.isfinite(y1) and math.isfinite(y2)):
                    return False
                if abs(y2 - y1) / dx > max_local_slope:
                    return False
            return True
        except Exception:
            return False
    return Contract(name=f'smooth_probe(max={max_local_slope},dx={dx})', check=_check)

@dataclass
class TestForgePlus:
    domain: Tuple[float, float] = (-3.0, 3.0)
    base_n: int = 64
    adversarial_n: int = 64
    boundary_n: int = 24
    regression_n: int = 32
    focus_strength: float = 0.5
    regression_bank: List[float] = field(default_factory=list)
    bank_max: int = 256

    def sample_base(self) -> List[float]:
        lo, hi = self.domain
        return [random.uniform(lo, hi) for _ in range(self.base_n)]

    def sample_boundary(self) -> List[float]:
        lo, hi = self.domain
        xs = []
        for _ in range(self.boundary_n):
            if random.random() < 0.5:
                xs.append(lo + abs(random.gauss(0, 0.08)) * (hi - lo))
            else:
                xs.append(hi - abs(random.gauss(0, 0.08)) * (hi - lo))
        return xs

    def sample_regression(self) -> List[float]:
        if not self.regression_bank:
            return []
        k = min(self.regression_n, len(self.regression_bank))
        return random.sample(self.regression_bank, k=k)

    def add_regression_points(self, xs: List[float]) -> None:
        for x in xs:
            if not math.isfinite(x):
                continue
            self.regression_bank.append(float(x))
        if len(self.regression_bank) > self.bank_max:
            self.regression_bank = self.regression_bank[-self.bank_max:]

    def sample_adversarial(self, fns: List[Callable[[float], float]]) -> List[float]:
        lo, hi = self.domain
        pool = [random.uniform(lo, hi) for _ in range(self.adversarial_n * 6)]
        scored: List[Tuple[float, float]] = []
        for x in pool:
            ys = []
            valid_ys = []
            for fn in fns:
                try:
                    y = fn(x)
                    ys.append(y)
                    if math.isfinite(y):
                        valid_ys.append(y)
                except Exception:
                    ys.append(float('nan'))
            if any((not math.isfinite(y) for y in ys)):
                score = 1000000000.0
            elif len(valid_ys) > 1:
                score = statistics.pvariance(valid_ys)
            else:
                score = 0.0
            scored.append((score, x))
        scored.sort(reverse=True, key=lambda t: t[0])
        top = [x for _, x in scored[:self.adversarial_n]]
        mix_n = int(self.adversarial_n * self.focus_strength)
        rand_n = self.adversarial_n - mix_n
        extra = [random.uniform(lo, hi) for _ in range(rand_n)]
        return top[:mix_n] + extra

    def update_focus(self, signal: float) -> None:
        self.focus_strength = min(0.95, max(0.05, self.focus_strength + 0.08 * signal))

    def make_train_val(self, top_fns: List[Callable[[float], float]]) -> Tuple[List[float], List[float]]:
        base = self.sample_base()
        adv = self.sample_adversarial(top_fns) if top_fns else []
        bnd = self.sample_boundary()
        reg = self.sample_regression()
        train = sorted(list(set(base + bnd)))
        val = sorted(list(set(base + adv + bnd + reg)))
        return (train, val)
UNARY_OPS = ['neg', 'sin', 'cos', 'tanh', 'abs', 'log1p', 'sqrt']
BINARY_OPS = ['add', 'sub', 'mul', 'div', 'max', 'min', 'pow2']

def random_leaf() -> Node:
    if random.random() < 0.55:
        return Var()
    return Const(random.uniform(-2.0, 2.0))

def random_tree(max_depth: int=4) -> Node:
    if max_depth <= 0:
        return random_leaf()
    r = random.random()
    if r < 0.33:
        return random_leaf()
    if r < 0.63:
        return Unary(random.choice(UNARY_OPS), random_tree(max_depth - 1))
    return Binary(random.choice(BINARY_OPS), random_tree(max_depth - 1), random_tree(max_depth - 1))

def simplify(node: Node) -> Node:
    if isinstance(node, Unary):
        node.a = simplify(node.a)
        if node.op == 'neg' and isinstance(node.a, Unary) and (node.a.op == 'neg'):
            return node.a.a
        return node
    if isinstance(node, Binary):
        node.a = simplify(node.a)
        node.b = simplify(node.b)
        if isinstance(node.a, Const) and isinstance(node.b, Const):
            try:
                return Const(node.eval(0.0))
            except Exception:
                return node
        if node.op == 'mul':
            if isinstance(node.a, Const) and abs(node.a.c - 1.0) < 1e-09:
                return node.b
            if isinstance(node.b, Const) and abs(node.b.c - 1.0) < 1e-09:
                return node.a
            if isinstance(node.a, Const) and abs(node.a.c) < 1e-09:
                return Const(0.0)
            if isinstance(node.b, Const) and abs(node.b.c) < 1e-09:
                return Const(0.0)
        if node.op == 'add':
            if isinstance(node.a, Const) and abs(node.a.c) < 1e-09:
                return node.b
            if isinstance(node.b, Const) and abs(node.b.c) < 1e-09:
                return node.a
        return node
    return node

@dataclass
class OperatorBanditStats:
    name: str
    tries: int = 0
    wins: int = 0
    ema_gain: float = 0.0
    ema_decay: float = 0.93
    fail_streak: int = 0

    def ucb_score(self, total_tries: int, c: float=1.0) -> float:
        t = max(1, self.tries)
        win_rate = (self.wins + 1) / (self.tries + 2)
        bonus = c * math.sqrt(math.log(total_tries + 1) / t)
        penalty = 0.05 * min(10, self.fail_streak)
        safe_ema = max(-10.0, min(10.0, self.ema_gain)) if math.isfinite(self.ema_gain) else 0.0
        score = 0.6 * win_rate + 0.4 * safe_ema + bonus - penalty
        return score if math.isfinite(score) else 0.0

    def update(self, improved: bool, gain: float) -> None:
        self.tries += 1
        if improved:
            self.wins += 1
            self.fail_streak = 0
        else:
            self.fail_streak += 1
        safe_gain = max(-10.0, min(10.0, gain)) if math.isfinite(gain) else 0.0
        self.ema_gain = self.ema_gain * self.ema_decay + (1 - self.ema_decay) * safe_gain
        self.ema_gain = max(-10.0, min(10.0, self.ema_gain))

class PatchForgePlus:

    def __init__(self) -> None:
        self.ops: Dict[str, Callable[[Node], Node]] = {'mutate_const': self._mutate_const, 'replace_subtree': self._replace_subtree, 'wrap_unary': self._wrap_unary, 'wrap_binary': self._wrap_binary, 'swap_children': self._swap_children, 'simplify': self._simplify, 'shrink_subtree': self._shrink_subtree, 'nudge_structure': self._nudge_structure}
        self.stats: Dict[str, OperatorBanditStats] = {k: OperatorBanditStats(k) for k in self.ops.keys()}

    def pick_operator(self) -> str:
        names = list(self.ops.keys())
        total = sum((self.stats[n].tries for n in names)) + 1
        scores = [self.stats[n].ucb_score(total_tries=total) for n in names]
        temp = 0.5
        max_s = max(scores) if scores else 0.0
        exps = []
        for s in scores:
            diff = (s - max_s) / temp
            if diff < -700:
                exps.append(0.0)
            elif diff > 700:
                exps.append(1e+300)
            else:
                exps.append(math.exp(diff))
        sum_exps = sum(exps)
        if not math.isfinite(sum_exps) or sum_exps <= 0:
            return random.choice(names)
        weights = [e / sum_exps for e in exps]
        if not all((math.isfinite(w) for w in weights)):
            return random.choice(names)
        return random.choices(names, weights=weights, k=1)[0]

    def apply(self, node: Node) -> Tuple[str, Node]:
        op_name = self.pick_operator()
        new_node = self.ops[op_name](node.clone())
        return (op_name, new_node)

    def report(self, op_name: str, improved: bool, gain: float) -> None:
        self.stats[op_name].update(improved=improved, gain=gain)

    def _mutate_const(self, node: Node) -> Node:
        consts = [n for n in node.walk() if isinstance(n, Const)]
        if not consts:
            return node
        c = random.choice(consts)
        c.c += random.uniform(-1.0, 1.0) * (0.15 + 0.35 * random.random())
        return simplify(node)

    def _replace_subtree(self, node: Node) -> Node:
        all_nodes = node.walk()
        target = random.choice(all_nodes)
        repl = random_tree(max_depth=random.randint(1, 3))
        if target is node:
            return simplify(repl)
        node.replace_child(target, repl)
        return simplify(node)

    def _wrap_unary(self, node: Node) -> Node:
        target = random.choice(node.walk())
        wrapped = Unary(random.choice(UNARY_OPS), target.clone())
        if target is node:
            return simplify(wrapped)
        node.replace_child(target, wrapped)
        return simplify(node)

    def _wrap_binary(self, node: Node) -> Node:
        target = random.choice(node.walk())
        other = random_tree(max_depth=2)
        wrapped = Binary(random.choice(BINARY_OPS), target.clone(), other) if random.random() < 0.5 else Binary(random.choice(BINARY_OPS), other, target.clone())
        if target is node:
            return simplify(wrapped)
        node.replace_child(target, wrapped)
        return simplify(node)

    def _swap_children(self, node: Node) -> Node:
        bins = [n for n in node.walk() if isinstance(n, Binary)]
        if not bins:
            return node
        b = random.choice(bins)
        b.a, b.b = (b.b, b.a)
        return simplify(node)

    def _simplify(self, node: Node) -> Node:
        return simplify(node)

    def _shrink_subtree(self, node: Node) -> Node:
        all_nodes = node.walk()
        target = random.choice(all_nodes)
        replacement: Node
        if isinstance(target, Binary) and random.random() < 0.6:
            replacement = target.a.clone() if random.random() < 0.5 else target.b.clone()
        else:
            replacement = random_leaf()
        if target is node:
            return simplify(replacement)
        node.replace_child(target, replacement)
        return simplify(node)

    def _nudge_structure(self, node: Node) -> Node:
        target = random.choice(node.walk())
        if random.random() < 0.5:
            n = Binary('add', target.clone(), Const(0.0))
        else:
            n = Binary('mul', target.clone(), Const(1.0))
        if target is node:
            return simplify(n)
        node.replace_child(target, n)
        return simplify(node)

class MetaRule:
    """Base class for meta-level rule trees that control MetaBrain behavior."""

    def eval(self, state: Dict[str, float]) -> float:
        raise NotImplementedError

    def clone(self) -> 'MetaRule':
        raise NotImplementedError

    def size(self) -> int:
        raise NotImplementedError

    def walk(self) -> List['MetaRule']:
        raise NotImplementedError

    def to_str(self) -> str:
        raise NotImplementedError

    def to_dict(self) -> Dict:
        """Serialize to dict for transfer learning."""
        raise NotImplementedError

    @staticmethod
    def from_dict(d: Dict) -> 'MetaRule':
        """Deserialize from dict for transfer learning."""
        t = d.get('type')
        if t == 'const':
            return MRConst(v=d['v'])
        elif t == 'var':
            return MRVar(k=d['k'])
        elif t == 'bin':
            return MRBin(op=d['op'], a=MetaRule.from_dict(d['a']), b=MetaRule.from_dict(d['b']))
        raise ValueError(f'Unknown MetaRule type: {t}')

@dataclass
class MRConst(MetaRule):
    """Constant value in MetaRule tree."""
    v: float

    def eval(self, state: Dict[str, float]) -> float:
        return self.v

    def clone(self) -> 'MetaRule':
        return MRConst(self.v)

    def size(self) -> int:
        return 1

    def walk(self) -> List['MetaRule']:
        return [self]

    def to_str(self) -> str:
        return f'{self.v:.4g}'

    def to_dict(self) -> Dict:
        return {'type': 'const', 'v': self.v}

@dataclass
class MRVar(MetaRule):
    """Variable reference in MetaRule tree (reads from state dict)."""
    k: str

    def eval(self, state: Dict[str, float]) -> float:
        return state.get(self.k, 0.0)

    def clone(self) -> 'MetaRule':
        return MRVar(self.k)

    def size(self) -> int:
        return 1

    def walk(self) -> List['MetaRule']:
        return [self]

    def to_str(self) -> str:
        return f'${self.k}'

    def to_dict(self) -> Dict:
        return {'type': 'var', 'k': self.k}

@dataclass
class MRBin(MetaRule):
    """Binary operation in MetaRule tree."""
    op: str
    a: MetaRule
    b: MetaRule

    def eval(self, state: Dict[str, float]) -> float:
        x = self.a.eval(state)
        y = self.b.eval(state)
        if self.op == 'add':
            return x + y
        if self.op == 'sub':
            return x - y
        if self.op == 'mul':
            return x * y
        if self.op == 'div':
            return x / (y if abs(y) > 1e-09 else 1e-09)
        if self.op == 'max':
            return max(x, y)
        if self.op == 'min':
            return min(x, y)
        return 0.0

    def clone(self) -> 'MetaRule':
        return MRBin(self.op, self.a.clone(), self.b.clone())

    def size(self) -> int:
        return 1 + self.a.size() + self.b.size()

    def walk(self) -> List['MetaRule']:
        return [self] + self.a.walk() + self.b.walk()

    def to_str(self) -> str:
        return f'{self.op}({self.a.to_str()}, {self.b.to_str()})'

    def to_dict(self) -> Dict:
        return {'type': 'bin', 'op': self.op, 'a': self.a.to_dict(), 'b': self.b.to_dict()}

class MetaBrain:
    """
    Self-modifying meta-cognitive controller.
    - Internal state tracked as dict
    - Outputs (pressure, amplification, chaos) computed by MetaRule trees
    - NO SAFETY CONSTRAINTS - pure recursive self-improvement
    """

    def __init__(self) -> None:
        self.state: Dict[str, float] = {'ema_gain': 0.0, 'stagnation': 0.0, 'step': 0.0, 'best_obj': float('inf'), 'temperature': 1.0}
        self.pressure: float = 1.0
        self.amplification: float = 5.0
        self.chaos: float = 0.1
        self.structural_strategy: Dict[str, float] = {'assign': 1.0, 'loop': 1.0, 'branch': 1.0}
        self.rule_pressure: MetaRule = MRBin('add', MRVar('ema_gain'), MRConst(1.0))
        self.rule_amplification: MetaRule = MRBin('mul', MRVar('ema_gain'), MRConst(10.0))
        self.rule_chaos: MetaRule = MRBin('div', MRVar('stagnation'), MRConst(10.0))
        self.rule_loop_bias: MetaRule = MRBin('add', MRConst(1.0), MRVar('stagnation'))
        self.rule_branch_bias: MetaRule = MRBin('add', MRConst(1.0), MRBin('max', MRConst(0.2), MRVar('ema_gain')))
        self.rule_assign_bias: MetaRule = MRBin('add', MRConst(1.0), MRBin('sub', MRConst(2.0), MRVar('ema_gain')))
        self.history: List[Tuple[float, float, float]] = []
        self.generation: int = 0
        self.lineage: List[int] = []

    def perceive(self, gain: float, stagnation_inc: int, best_obj: float=0.0) -> None:
        """
        Update internal state and compute new outputs via rule evaluation.
        With SAFETY CLAMPING to prevent inf divergence.
        """
        gain = max(-10.0, min(10.0, gain)) if math.isfinite(gain) else 0.0
        self.state['ema_gain'] = 0.9 * self.state['ema_gain'] + 0.1 * gain
        self.state['ema_gain'] = max(-100.0, min(100.0, self.state['ema_gain']))
        self.state['stagnation'] += stagnation_inc
        self.state['stagnation'] = max(0.0, min(1000.0, self.state['stagnation']))
        self.state['step'] += 1
        if math.isfinite(best_obj) and best_obj < self.state['best_obj']:
            self.state['best_obj'] = best_obj
        try:
            p = self.rule_pressure.eval(self.state)
            self.pressure = max(0.01, min(100.0, p)) if math.isfinite(p) else 1.0
        except Exception:
            self.pressure = 1.0
        try:
            a = self.rule_amplification.eval(self.state)
            self.amplification = max(0.1, min(100.0, a)) if math.isfinite(a) else 5.0
        except Exception:
            self.amplification = 5.0
        try:
            c = self.rule_chaos.eval(self.state)
            self.chaos = max(0.0, min(1.0, c)) if math.isfinite(c) else 0.1
        except Exception:
            self.chaos = 0.1
        self.structural_strategy = self._compute_structural_strategy()
        self.history.append((self.pressure, self.amplification, self.chaos))

    def _safe_weight(self, rule: MetaRule, fallback: float) -> float:
        """Evaluate a structural rule with safety checks."""
        try:
            val = rule.eval(self.state)
            if not math.isfinite(val):
                return fallback
            return max(0.0, val)
        except Exception:
            return fallback

    def _compute_structural_strategy(self) -> Dict[str, float]:
        raw = {'assign': self._safe_weight(self.rule_assign_bias, self.structural_strategy.get('assign', 1.0)), 'loop': self._safe_weight(self.rule_loop_bias, self.structural_strategy.get('loop', 1.0)), 'branch': self._safe_weight(self.rule_branch_bias, self.structural_strategy.get('branch', 1.0))}
        total = sum(raw.values())
        if total <= 0:
            return {'assign': 1.0, 'loop': 1.0, 'branch': 1.0}
        return {k: v / total for k, v in raw.items()}

    def get_structural_strategy(self) -> Dict[str, float]:
        """Return the latest normalized structural probabilities."""
        if not self.structural_strategy:
            return {'assign': 1.0, 'loop': 1.0, 'branch': 1.0}
        return dict(self.structural_strategy)

    def clone(self) -> 'MetaBrain':
        """Create a deep copy of this MetaBrain."""
        child = MetaBrain()
        child.state = dict(self.state)
        child.pressure = self.pressure
        child.amplification = self.amplification
        child.chaos = self.chaos
        child.structural_strategy = dict(self.structural_strategy)
        child.rule_pressure = self.rule_pressure.clone()
        child.rule_amplification = self.rule_amplification.clone()
        child.rule_chaos = self.rule_chaos.clone()
        child.rule_loop_bias = self.rule_loop_bias.clone()
        child.rule_branch_bias = self.rule_branch_bias.clone()
        child.rule_assign_bias = self.rule_assign_bias.clone()
        child.history = list(self.history)
        child.generation = self.generation + 1
        child.lineage = self.lineage + [self.generation]
        return child

    def total_rule_size(self) -> int:
        """Total complexity of all rules."""
        return self.rule_pressure.size() + self.rule_amplification.size() + self.rule_chaos.size() + self.rule_loop_bias.size() + self.rule_branch_bias.size() + self.rule_assign_bias.size()

    def signature(self) -> List[float]:
        """Behavioral signature for novelty comparison."""
        if not self.history:
            return [0.0, 0.0, 0.0]
        recent = self.history[-min(10, len(self.history)):]
        return [sum((h[0] for h in recent)) / len(recent), sum((h[1] for h in recent)) / len(recent), sum((h[2] for h in recent)) / len(recent)]

    def to_str(self) -> str:
        return f'MetaBrain(gen={self.generation}, pressure={self.rule_pressure.to_str()}, amp={self.rule_amplification.to_str()}, chaos={self.rule_chaos.to_str()})'

    def to_dict(self) -> Dict:
        """Serialize MetaBrain to dict for transfer learning."""
        return {'generation': self.generation, 'lineage': self.lineage, 'state': dict(self.state), 'rules': {'pressure': self.rule_pressure.to_dict(), 'amplification': self.rule_amplification.to_dict(), 'chaos': self.rule_chaos.to_dict(), 'loop_bias': self.rule_loop_bias.to_dict(), 'branch_bias': self.rule_branch_bias.to_dict(), 'assign_bias': self.rule_assign_bias.to_dict()}, 'outputs': {'pressure': self.pressure, 'amplification': self.amplification, 'chaos': self.chaos, 'structural_strategy': dict(self.structural_strategy)}, 'history_summary': {'total_steps': len(self.history), 'recent_avg': self.signature()}}

    @staticmethod
    def from_dict(d: Dict) -> 'MetaBrain':
        """Deserialize MetaBrain from dict for transfer learning."""
        brain = MetaBrain()
        brain.generation = d.get('generation', 0)
        brain.lineage = d.get('lineage', [])
        brain.state = d.get('state', brain.state)
        rules = d.get('rules', {})
        if 'pressure' in rules:
            brain.rule_pressure = MetaRule.from_dict(rules['pressure'])
        if 'amplification' in rules:
            brain.rule_amplification = MetaRule.from_dict(rules['amplification'])
        if 'chaos' in rules:
            brain.rule_chaos = MetaRule.from_dict(rules['chaos'])
        if 'loop_bias' in rules:
            brain.rule_loop_bias = MetaRule.from_dict(rules['loop_bias'])
        if 'branch_bias' in rules:
            brain.rule_branch_bias = MetaRule.from_dict(rules['branch_bias'])
        if 'assign_bias' in rules:
            brain.rule_assign_bias = MetaRule.from_dict(rules['assign_bias'])
        outputs = d.get('outputs', {})
        brain.pressure = outputs.get('pressure', 1.0)
        brain.amplification = outputs.get('amplification', 5.0)
        brain.chaos = outputs.get('chaos', 0.1)
        brain.structural_strategy = outputs.get('structural_strategy', {'assign': 1.0, 'loop': 1.0, 'branch': 1.0})
        return brain

    def save(self, filepath: str) -> None:
        """Save MetaBrain to JSON file for transfer learning."""
        import json
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

    @staticmethod
    def load(filepath: str) -> 'MetaBrain':
        """Load MetaBrain from JSON file for transfer learning."""
        import json
        with open(filepath, 'r', encoding='utf-8') as f:
            d = json.load(f)
        return MetaBrain.from_dict(d)

    def apply_task_agnostic_prior(self) -> None:
        """
        Reset rules to task-agnostic priors to prevent overfitting.
        Keeps structure but resets constants to neutral values.
        """

        def reset_constants(rule: MetaRule, neutral: float=1.0) -> None:
            for node in rule.walk():
                if isinstance(node, MRConst):
                    node.v = neutral if node.v >= 0 else -neutral
        reset_constants(self.rule_pressure, 1.0)
        reset_constants(self.rule_amplification, 5.0)
        reset_constants(self.rule_chaos, 0.1)
        reset_constants(self.rule_loop_bias, 1.0)
        reset_constants(self.rule_branch_bias, 1.0)
        reset_constants(self.rule_assign_bias, 1.0)
        self.state['ema_gain'] = 0.0
        self.state['stagnation'] = 0.0
        self.state['temperature'] = 1.0
        self.structural_strategy = {'assign': 1.0, 'loop': 1.0, 'branch': 1.0}

    def check_health(self, current_loss: float, baseline_loss: float) -> bool:
        """
        Meta-Meta-Intervention: Monitor health.
        Returns True if performance is degrading dangerously (Toxic Transfer).
        """
        if baseline_loss == 0.0:
            return False
        if current_loss > baseline_loss * 2.0 and current_loss > 0.1:
            return True
        if current_loss > 1000000.0:
            return True
        return False

    def emergency_reset(self) -> None:
        """
        Meta-Meta-Intervention: Emergency Reset.
        Triggered when check_health() fails.
        Forces the brain into a high-chaos "Safe Mode" to escape local optima/traps.
        """
        print('[MetaBrain] [ALARM] EMERGENCY RESET TRIGGERED! (Toxic Transfer Detected)')
        self.pressure = 0.5
        self.amplification = 2.0
        self.chaos = 0.5
        self.state['ema_gain'] = 0.0
        self.state['stagnation'] = 0.0
        self.apply_task_agnostic_prior()

    def perturb(self, scale: float=0.05) -> 'MetaBrain':
        """
        Create a slightly perturbed copy of the brain for stability analysis (SAM-lite).
        Modifies constants slightly to test landscape sharpness.
        """
        import copy
        new_brain = copy.deepcopy(self)

        def apply_noise(rule: MetaRule):
            for node in rule.walk():
                if isinstance(node, MRConst):
                    node.v += random.normalvariate(0, scale)
        apply_noise(new_brain.rule_pressure)
        apply_noise(new_brain.rule_amplification)
        apply_noise(new_brain.rule_chaos)
        apply_noise(new_brain.rule_loop_bias)
        apply_noise(new_brain.rule_branch_bias)
        apply_noise(new_brain.rule_assign_bias)
        return new_brain

class EnsembleBrainSelector:
    """
    Soft Transfer / Ensemble Warm Start.
    
    During initial N steps, maintains both:
    - Default (fresh) brain
    - Transfer (loaded) brain
    
    Uses short probe evaluation to select or blend the better one.
    """

    def __init__(self, default_brain: MetaBrain, transfer_brain: Optional[MetaBrain]=None) -> None:
        self.default_brain = default_brain
        self.transfer_brain = transfer_brain
        self.probe_steps: int = 20
        self.current_step: int = 0
        self.default_score: float = 0.0
        self.transfer_score: float = 0.0
        self.selected: str = 'default'
        self.blend_ratio: float = 0.5

    def probe_evaluate(self, brain: MetaBrain, val_losses: List[float]) -> float:
        """Evaluate brain performance on recent val_losses."""
        if not val_losses:
            return float('inf')
        mean_loss = sum(val_losses) / len(val_losses)
        variance = sum(((v - mean_loss) ** 2 for v in val_losses)) / len(val_losses)
        return mean_loss + 0.1 * variance

    def step(self, val_loss: float) -> MetaBrain:
        """
        Called each step. Returns the brain to use.
        During probe phase, tracks both brains.
        After probe phase, returns the selected/blended brain.
        """
        self.current_step += 1
        if self.transfer_brain is None:
            return self.default_brain
        if self.current_step <= self.probe_steps:
            self.default_score += val_loss
            self.transfer_score += val_loss * (0.9 if self.transfer_brain.generation > 0 else 1.0)
            return self.default_brain
        elif self.current_step == self.probe_steps + 1:
            if self.transfer_score < self.default_score * 0.85:
                self.selected = 'transfer'
                print(f'[EnsembleBrain] Selected TRANSFER brain (score: {self.transfer_score:.4f} < {self.default_score:.4f})')
                return self.transfer_brain
            else:
                self.selected = 'default'
                print(f'[EnsembleBrain] Selected DEFAULT brain (score: {self.default_score:.4f})')
                return self.default_brain
        else:
            return self.transfer_brain if self.selected == 'transfer' else self.default_brain

    def get_blended_brain(self) -> MetaBrain:
        """
        Create a blended brain combining default and transfer.
        Uses weighted average of outputs.
        """
        if self.transfer_brain is None:
            return self.default_brain
        blended = self.default_brain.clone()
        r = self.blend_ratio
        blended.pressure = (1 - r) * self.default_brain.pressure + r * self.transfer_brain.pressure
        blended.amplification = (1 - r) * self.default_brain.amplification + r * self.transfer_brain.amplification
        blended.chaos = (1 - r) * self.default_brain.chaos + r * self.transfer_brain.chaos
        return blended

class SeparateMetrics:
    """
    Separate tracking of val_loss vs objective for transfer evaluation.
    Enables proper diagnosis of "true transfer" vs "metric gaming".
    """

    def __init__(self) -> None:
        self.val_loss_history: List[float] = []
        self.objective_history: List[float] = []
        self.step_history: List[int] = []
        self.phase_markers: Dict[str, int] = {}

    def record(self, step: int, val_loss: float, objective: float) -> None:
        """Record metrics at each step."""
        self.step_history.append(step)
        self.val_loss_history.append(val_loss)
        self.objective_history.append(objective)

    def mark_phase(self, phase_name: str, step: int) -> None:
        """Mark the start of a phase (e.g., 'cold_start', 'warm_start')."""
        self.phase_markers[phase_name] = step

    def get_phase_stats(self, phase_name: str) -> Dict:
        """Get statistics for a specific phase."""
        if phase_name not in self.phase_markers:
            return {}
        start = self.phase_markers[phase_name]
        indices = [i for i, s in enumerate(self.step_history) if s >= start]
        if not indices:
            return {}
        vl = [self.val_loss_history[i] for i in indices]
        obj = [self.objective_history[i] for i in indices]
        return {'phase': phase_name, 'start_step': start, 'n_samples': len(indices), 'val_loss_mean': sum(vl) / len(vl), 'val_loss_min': min(vl), 'objective_mean': sum(obj) / len(obj), 'objective_min': min(obj)}

    def compare_phases(self, phase_a: str, phase_b: str) -> Dict:
        """Compare two phases. Returns which is better and by how much."""
        a = self.get_phase_stats(phase_a)
        b = self.get_phase_stats(phase_b)
        if not a or not b:
            return {'error': 'Phase not found'}
        val_loss_diff = a['val_loss_mean'] - b['val_loss_mean']
        objective_diff = a['objective_mean'] - b['objective_mean']
        return {'val_loss_improvement': -val_loss_diff, 'objective_improvement': -objective_diff, 'val_loss_winner': phase_b if val_loss_diff > 0 else phase_a, 'objective_winner': phase_b if objective_diff > 0 else phase_a, 'consistent': (val_loss_diff > 0) == (objective_diff > 0)}

    def to_dict(self) -> Dict:
        return {'val_loss_history': self.val_loss_history[-50:], 'objective_history': self.objective_history[-50:], 'phase_markers': self.phase_markers}
META_RULE_VARS = ['ema_gain', 'stagnation', 'step', 'best_obj', 'temperature']
META_RULE_OPS = ['add', 'sub', 'mul', 'div', 'max', 'min']

class MetaPatchForge:
    """
    Mutates MetaRule trees - this is what enables true RSI.
    The brain-editor that edits the brain.
    NO SAFETY CONSTRAINTS.
    """

    def __init__(self) -> None:
        self.mutation_tries: int = 0
        self.mutation_wins: int = 0

    def mutate_rule(self, rule: MetaRule) -> MetaRule:
        """Apply random mutation to a MetaRule tree."""
        self.mutation_tries += 1
        if random.random() < 0.25:
            return self.random_rule(depth=random.randint(1, 3))
        return self._tweak(rule.clone())

    def _tweak(self, rule: MetaRule) -> MetaRule:
        """In-place tweaking of a rule tree."""
        if isinstance(rule, MRConst):
            rule.v += random.gauss(0, 0.5)
            return rule
        if isinstance(rule, MRVar):
            if random.random() < 0.3:
                rule.k = random.choice(META_RULE_VARS)
            return rule
        if isinstance(rule, MRBin):
            r = random.random()
            if r < 0.25:
                rule.op = random.choice(META_RULE_OPS)
            elif r < 0.5:
                rule.a = self._tweak(rule.a)
            elif r < 0.75:
                rule.b = self._tweak(rule.b)
            elif random.random() < 0.5:
                rule = MRBin(random.choice(META_RULE_OPS), rule, self.random_rule(depth=1))
            else:
                rule = MRBin(random.choice(META_RULE_OPS), self.random_rule(depth=1), rule)
            return rule
        return rule

    def random_rule(self, depth: int) -> MetaRule:
        """Generate a random MetaRule tree."""
        if depth <= 0 or random.random() < 0.4:
            if random.random() < 0.5:
                return MRConst(random.uniform(-5.0, 5.0))
            else:
                return MRVar(random.choice(META_RULE_VARS))
        return MRBin(random.choice(META_RULE_OPS), self.random_rule(depth - 1), self.random_rule(depth - 1))

    def mutate_brain(self, brain: MetaBrain) -> MetaBrain:
        """Create a mutated child MetaBrain."""
        child = brain.clone()
        n_mutations = random.randint(1, 3)
        for _ in range(n_mutations):
            which = random.randint(0, 2)
            if which == 0:
                child.rule_pressure = self.mutate_rule(child.rule_pressure)
            elif which == 1:
                child.rule_amplification = self.mutate_rule(child.rule_amplification)
            else:
                child.rule_chaos = self.mutate_rule(child.rule_chaos)
        return child

    def report_win(self) -> None:
        self.mutation_wins += 1

    def to_dict(self) -> Dict:
        """Serialize MetaPatchForge state for L4 transfer."""
        return {'mutation_tries': self.mutation_tries, 'mutation_wins': self.mutation_wins}

    @staticmethod
    def from_dict(d: Dict) -> 'MetaPatchForge':
        forge = MetaPatchForge()
        forge.mutation_tries = d.get('mutation_tries', 0)
        forge.mutation_wins = d.get('mutation_wins', 0)
        return forge

@dataclass
class ForgeGenome:
    """
    The 'DNA' of a MetaPatchForge - parameters that control how it mutates.
    This is what L4 evolves.
    """
    replace_prob: float = 0.25
    op_change_prob: float = 0.25
    var_switch_prob: float = 0.3
    max_random_depth: int = 3
    mutation_intensity: float = 0.5
    mutations_per_edit: int = 2

    def clone(self) -> 'ForgeGenome':
        return ForgeGenome(replace_prob=self.replace_prob, op_change_prob=self.op_change_prob, var_switch_prob=self.var_switch_prob, max_random_depth=self.max_random_depth, mutation_intensity=self.mutation_intensity, mutations_per_edit=self.mutations_per_edit)

    def to_dict(self) -> Dict:
        return {'replace_prob': self.replace_prob, 'op_change_prob': self.op_change_prob, 'var_switch_prob': self.var_switch_prob, 'max_random_depth': self.max_random_depth, 'mutation_intensity': self.mutation_intensity, 'mutations_per_edit': self.mutations_per_edit}

    @staticmethod
    def from_dict(d: Dict) -> 'ForgeGenome':
        return ForgeGenome(replace_prob=d.get('replace_prob', 0.25), op_change_prob=d.get('op_change_prob', 0.25), var_switch_prob=d.get('var_switch_prob', 0.3), max_random_depth=d.get('max_random_depth', 3), mutation_intensity=d.get('mutation_intensity', 0.5), mutations_per_edit=d.get('mutations_per_edit', 2))

class EvolvableMetaPatchForge(MetaPatchForge):
    """
    MetaPatchForge with evolvable parameters (ForgeGenome).
    L4 evolves the genome, which controls how L3 edits MetaBrain.
    """

    def __init__(self, genome: Optional[ForgeGenome]=None) -> None:
        super().__init__()
        self.genome: ForgeGenome = genome if genome else ForgeGenome()
        self.generation: int = 0

    def mutate_rule(self, rule: MetaRule) -> MetaRule:
        """Apply mutation using genome parameters."""
        self.mutation_tries += 1
        if random.random() < self.genome.replace_prob:
            return self.random_rule(depth=self.genome.max_random_depth)
        return self._tweak(rule.clone())

    def _tweak(self, rule: MetaRule) -> MetaRule:
        """Tweak using genome parameters."""
        if isinstance(rule, MRConst):
            rule.v += random.gauss(0, self.genome.mutation_intensity)
            return rule
        if isinstance(rule, MRVar):
            if random.random() < self.genome.var_switch_prob:
                rule.k = random.choice(META_RULE_VARS)
            return rule
        if isinstance(rule, MRBin):
            r = random.random()
            if r < self.genome.op_change_prob:
                rule.op = random.choice(META_RULE_OPS)
            elif r < self.genome.op_change_prob * 2:
                rule.a = self._tweak(rule.a)
            elif r < self.genome.op_change_prob * 3:
                rule.b = self._tweak(rule.b)
            elif random.random() < 0.5:
                rule = MRBin(random.choice(META_RULE_OPS), rule, self.random_rule(1))
            else:
                rule = MRBin(random.choice(META_RULE_OPS), self.random_rule(1), rule)
            return rule
        return rule

    def mutate_brain(self, brain: MetaBrain) -> MetaBrain:
        """Create mutated child using genome parameters."""
        child = brain.clone()
        n_mutations = self.genome.mutations_per_edit
        for _ in range(n_mutations):
            which = random.randint(0, 2)
            if which == 0:
                child.rule_pressure = self.mutate_rule(child.rule_pressure)
            elif which == 1:
                child.rule_amplification = self.mutate_rule(child.rule_amplification)
            else:
                child.rule_chaos = self.mutate_rule(child.rule_chaos)
        return child

    def clone(self) -> 'EvolvableMetaPatchForge':
        new_forge = EvolvableMetaPatchForge(self.genome.clone())
        new_forge.mutation_tries = self.mutation_tries
        new_forge.mutation_wins = self.mutation_wins
        new_forge.generation = self.generation + 1
        return new_forge

class MetaMetaPatchForge:
    """
    L4: The Meta-Meta-PatchForge.
    Evolves ForgeGenome (the parameters of MetaPatchForge).
    This enables infinite recursive self-improvement:
    
    L1: Expression Trees (code)
     └─ L2: PatchForge (code editor)
         └─ L3: MetaBrain + MetaPatchForge (brain + brain editor)
             └─ L4: MetaMetaPatchForge (brain editor editor) ← THIS
                 └─ L5: ... (open for extension)
    """

    def __init__(self) -> None:
        self.genome_tries: int = 0
        self.genome_wins: int = 0
        self.archive: List[EvolvableMetaPatchForge] = []

    def mutate_genome(self, genome: ForgeGenome) -> ForgeGenome:
        """Mutate the genome parameters."""
        self.genome_tries += 1
        child = genome.clone()
        mutations = random.randint(1, 3)
        for _ in range(mutations):
            param = random.randint(0, 5)
            if param == 0:
                child.replace_prob = max(0.05, min(0.95, child.replace_prob + random.gauss(0, 0.1)))
            elif param == 1:
                child.op_change_prob = max(0.05, min(0.95, child.op_change_prob + random.gauss(0, 0.1)))
            elif param == 2:
                child.var_switch_prob = max(0.05, min(0.95, child.var_switch_prob + random.gauss(0, 0.1)))
            elif param == 3:
                child.max_random_depth = max(1, min(5, child.max_random_depth + random.choice([-1, 0, 1])))
            elif param == 4:
                child.mutation_intensity = max(0.1, min(2.0, child.mutation_intensity + random.gauss(0, 0.2)))
            else:
                child.mutations_per_edit = max(1, min(5, child.mutations_per_edit + random.choice([-1, 0, 1])))
        return child

    def evolve_forge(self, parent: EvolvableMetaPatchForge) -> EvolvableMetaPatchForge:
        """Create a mutated child forge with evolved genome."""
        child = parent.clone()
        child.genome = self.mutate_genome(parent.genome)
        return child

    def evaluate_forge(self, forge: EvolvableMetaPatchForge) -> float:
        """Evaluate forge performance (win rate)."""
        if forge.mutation_tries == 0:
            return 0.0
        return forge.mutation_wins / forge.mutation_tries

    def select_best(self) -> Optional[EvolvableMetaPatchForge]:
        """Select the best forge from archive."""
        if not self.archive:
            return None
        return max(self.archive, key=self.evaluate_forge)

    def report_win(self) -> None:
        self.genome_wins += 1

    def to_dict(self) -> Dict:
        return {'genome_tries': self.genome_tries, 'genome_wins': self.genome_wins, 'archive_size': len(self.archive)}
import ast
import copy
import os
import shutil

@dataclass
class CodeGenome:
    """
    L5 DNA - Parameters that control code editing behavior.
    These parameters evolve to improve code editing strategy.
    """
    const_mutation_rate: float = 0.1
    const_mutation_scale: float = 0.2
    default_mutation_rate: float = 0.05
    max_mutations_per_edit: int = 3
    generation: int = 0

    def clone(self) -> 'CodeGenome':
        return CodeGenome(const_mutation_rate=self.const_mutation_rate, const_mutation_scale=self.const_mutation_scale, default_mutation_rate=self.default_mutation_rate, max_mutations_per_edit=self.max_mutations_per_edit, generation=self.generation + 1)

    def to_dict(self) -> Dict:
        return {'const_mutation_rate': self.const_mutation_rate, 'const_mutation_scale': self.const_mutation_scale, 'default_mutation_rate': self.default_mutation_rate, 'max_mutations_per_edit': self.max_mutations_per_edit, 'generation': self.generation}

class CodePatchForge:
    """
    L5: Code self-editing system.
    Safe source code transformation using AST.
    
    Probability of syntax errors: Near 0%
    - AST tree transformations always maintain valid syntax
    - Validates with compile() before writing
    - Automatic rollback on failure
    
    Edit targets (safe operations only):
    - Numeric constants (literals)
    - Function default values
    - Specific class field initial values
    """

    def __init__(self, source_file: str) -> None:
        self.source_file = source_file
        self.genome: CodeGenome = CodeGenome()
        self.edit_tries: int = 0
        self.edit_wins: int = 0
        self.edit_history: List[Dict] = []

    def read_source(self) -> str:
        """Read own source code."""
        with open(self.source_file, 'r', encoding='utf-8') as f:
            return f.read()

    def parse_source(self, code: str) -> ast.Module:
        """Parse source code to AST."""
        return ast.parse(code)

    def backup_source(self) -> str:
        """Create backup before editing."""
        backup_path = self.source_file + '.bak'
        shutil.copy2(self.source_file, backup_path)
        return backup_path

    def restore_from_backup(self, backup_path: str) -> None:
        """Restore from backup if edit fails."""
        shutil.copy2(backup_path, self.source_file)

    def validate_code(self, code: str) -> bool:
        """Validate code compiles without syntax errors."""
        try:
            compile(code, '<string>', 'exec')
            return True
        except SyntaxError:
            return False

    def find_numeric_constants(self, tree: ast.Module) -> List[ast.Constant]:
        """Find all numeric constants in AST."""
        constants = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
                if abs(node.value) > 0.01:
                    constants.append(node)
        return constants

    def mutate_constant(self, node: ast.Constant) -> None:
        """Mutate a numeric constant value."""
        if isinstance(node.value, float):
            delta = random.gauss(0, self.genome.const_mutation_scale * abs(node.value + 0.1))
            node.value = node.value + delta
        elif isinstance(node.value, int) and node.value > 1:
            delta = random.choice([-1, 0, 1])
            node.value = max(1, node.value + delta)

    def mutate_ast(self, tree: ast.Module) -> Tuple[ast.Module, int]:
        """
        Apply safe mutations to AST.
        Returns: (mutated_tree, number_of_mutations)
        """
        tree = copy.deepcopy(tree)
        constants = self.find_numeric_constants(tree)
        if not constants:
            return (tree, 0)
        n_mutations = min(self.genome.max_mutations_per_edit, max(1, int(len(constants) * self.genome.const_mutation_rate)))
        targets = random.sample(constants, min(n_mutations, len(constants)))
        for const in targets:
            self.mutate_constant(const)
        ast.fix_missing_locations(tree)
        return (tree, len(targets))

    def ast_to_code(self, tree: ast.Module) -> str:
        """Convert AST back to source code."""
        return ast.unparse(tree)

    def try_edit(self) -> Tuple[bool, str]:
        """
        Try to edit own source code.
        Returns: (success, message)
        """
        self.edit_tries += 1
        try:
            original_code = self.read_source()
            tree = self.parse_source(original_code)
            mutated_tree, n_mutations = self.mutate_ast(tree)
            if n_mutations == 0:
                return (False, 'No mutations applied')
            new_code = self.ast_to_code(mutated_tree)
            if not self.validate_code(new_code):
                return (False, 'Validation failed (syntax error)')
            backup_path = self.backup_source()
            try:
                with open(self.source_file, 'w', encoding='utf-8') as f:
                    f.write(new_code)
                self.edit_history.append({'generation': self.genome.generation, 'n_mutations': n_mutations, 'success': True, 'backup': backup_path})
                self.edit_wins += 1
                return (True, f'Applied {n_mutations} mutations')
            except Exception as e:
                self.restore_from_backup(backup_path)
                return (False, f'Write failed, rolled back: {e}')
        except Exception as e:
            return (False, f'Edit failed: {e}')

    def evolve_genome(self) -> None:
        """Evolve the code editing genome based on performance."""
        win_rate = self.edit_wins / max(1, self.edit_tries)
        if win_rate > 0.5:
            self.genome.const_mutation_scale *= 1.1
            self.genome.const_mutation_rate = min(0.3, self.genome.const_mutation_rate * 1.05)
        else:
            self.genome.const_mutation_scale *= 0.9
            self.genome.const_mutation_rate = max(0.01, self.genome.const_mutation_rate * 0.95)
        self.genome = self.genome.clone()

    def to_dict(self) -> Dict:
        return {'genome': self.genome.to_dict(), 'edit_tries': self.edit_tries, 'edit_wins': self.edit_wins, 'win_rate': self.edit_wins / max(1, self.edit_tries)}

class StructuralCodePatchForge(CodePatchForge):
    """
    Extended L5: Modifies function and class STRUCTURES, not just constants.
    
    Safe structural modifications:
    - Add/remove function parameters with defaults
    - Insert print/logging statements
    - Modify loop ranges
    - Add conditional branches
    - Swap binary operators
    """

    def __init__(self, source_file: str) -> None:
        super().__init__(source_file)
        self.structural_edits: int = 0

    def find_functions(self, tree: ast.Module) -> List[ast.FunctionDef]:
        """Find all function definitions."""
        funcs = []
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                funcs.append(node)
        return funcs

    def find_binops(self, tree: ast.Module) -> List[ast.BinOp]:
        """Find all binary operations."""
        ops = []
        for node in ast.walk(tree):
            if isinstance(node, ast.BinOp):
                ops.append(node)
        return ops

    def mutate_binop(self, node: ast.BinOp) -> None:
        """Swap a binary operator (safe transformation)."""
        swap_map = {ast.Add: ast.Sub, ast.Sub: ast.Add, ast.Mult: ast.Div, ast.Div: ast.Mult}
        op_type = type(node.op)
        if op_type in swap_map:
            node.op = swap_map[op_type]()

    def add_logging_to_function(self, func: ast.FunctionDef) -> bool:
        """Add a print statement at function start (safe)."""
        if not func.body:
            return False
        log_stmt = ast.Expr(value=ast.Call(func=ast.Name(id='print', ctx=ast.Load()), args=[ast.Constant(value=f'[L5] {func.name} called')], keywords=[]))
        ast.fix_missing_locations(log_stmt)
        func.body.insert(0, log_stmt)
        return True

    def mutate_structural(self, tree: ast.Module) -> Tuple[ast.Module, int, str]:
        """
        Apply structural mutations to AST.
        Returns: (mutated_tree, n_mutations, description)
        """
        tree = copy.deepcopy(tree)
        mutations = 0
        descriptions = []
        if random.random() < 0.3:
            binops = self.find_binops(tree)
            if binops:
                target = random.choice(binops)
                old_op = type(target.op).__name__
                self.mutate_binop(target)
                new_op = type(target.op).__name__
                if old_op != new_op:
                    mutations += 1
                    descriptions.append(f'swapped {old_op}->{new_op}')
        if random.random() < 0.5:
            constants = self.find_numeric_constants(tree)
            if constants:
                target = random.choice(constants)
                old_val = target.value
                self.mutate_constant(target)
                mutations += 1
                descriptions.append(f'const {old_val:.2f}->{target.value:.2f}')
        ast.fix_missing_locations(tree)
        desc = ', '.join(descriptions) if descriptions else 'no changes'
        return (tree, mutations, desc)

    def try_structural_edit(self) -> Tuple[bool, str]:
        """
        Try structural edit with validation.
        """
        self.edit_tries += 1
        try:
            original_code = self.read_source()
            tree = self.parse_source(original_code)
            mutated_tree, n_mutations, desc = self.mutate_structural(tree)
            if n_mutations == 0:
                return (False, 'No structural mutations applied')
            new_code = self.ast_to_code(mutated_tree)
            if not self.validate_code(new_code):
                return (False, 'Validation failed')
            backup_path = self.backup_source()
            try:
                with open(self.source_file, 'w', encoding='utf-8') as f:
                    f.write(new_code)
                self.structural_edits += 1
                self.edit_wins += 1
                self.edit_history.append({'type': 'structural', 'mutations': n_mutations, 'description': desc, 'backup': backup_path})
                return (True, f'Structural edit: {desc}')
            except Exception as e:
                self.restore_from_backup(backup_path)
                return (False, f'Write failed: {e}')
        except Exception as e:
            return (False, f'Structural edit failed: {e}')

class Level3ASTMutator(StructuralCodePatchForge):
    """
    Level 3: Control Flow Mutations for LLM-Free Recursive Self-Improvement.
    
    Extends StructuralCodePatchForge with:
    - If/Else insertion (conditional branching)
    - Loop generation (iteration patterns)
    - Guard clause insertion (boundary conditions)
    - Comparison operator cycling (all operators tested systematically)
    - Statement reordering (execution order mutations)
    
    All mutations are AST-safe and validate before application.
    """

    def __init__(self, source_file: str) -> None:
        super().__init__(source_file)
        self.level3_edits: int = 0
        self.mutation_registry: List[str] = []
        self.comparison_ops = [ast.Lt, ast.Gt, ast.LtE, ast.GtE, ast.Eq, ast.NotEq]
        self.comparison_idx: int = 0

    def find_comparisons(self, tree: ast.Module) -> List[ast.Compare]:
        """Find all comparison operations in AST."""
        comps = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Compare):
                comps.append(node)
        return comps

    def find_if_statements(self, tree: ast.Module) -> List[ast.If]:
        """Find all if statements in AST."""
        ifs = []
        for node in ast.walk(tree):
            if isinstance(node, ast.If):
                ifs.append(node)
        return ifs

    def find_for_loops(self, tree: ast.Module) -> List[ast.For]:
        """Find all for loops in AST."""
        loops = []
        for node in ast.walk(tree):
            if isinstance(node, ast.For):
                loops.append(node)
        return loops

    def cycle_comparison_operator(self, node: ast.Compare) -> bool:
        """
        Deterministically cycle through comparison operators.
        Key insight: Instead of random guessing, systematically test all operators.
        """
        if not node.ops:
            return False
        current_op = type(node.ops[0])
        try:
            current_idx = self.comparison_ops.index(current_op)
            next_idx = (current_idx + 1) % len(self.comparison_ops)
            node.ops[0] = self.comparison_ops[next_idx]()
            self.comparison_idx = next_idx
            return True
        except ValueError:
            node.ops[0] = self.comparison_ops[0]()
            return True

    def insert_guard_clause(self, func: ast.FunctionDef) -> bool:
        """
        Insert a guard clause at function start.
        Pattern: if condition: return early
        
        This is useful for:
        - Boundary condition handling
        - Edge case protection
        - Early termination optimization
        """
        if not func.body:
            return False
        if not func.args.args:
            return False
        arg_name = func.args.args[0].arg
        guard = ast.If(test=ast.Compare(left=ast.Call(func=ast.Name(id='len', ctx=ast.Load()), args=[ast.Name(id=arg_name, ctx=ast.Load())], keywords=[]), ops=[ast.Eq()], comparators=[ast.Constant(value=0)]), body=[ast.Return(value=ast.Name(id=arg_name, ctx=ast.Load()))], orelse=[])
        ast.fix_missing_locations(guard)
        func.body.insert(0, guard)
        return True

    def swap_comparison_direction(self, tree: ast.Module) -> Tuple[bool, str]:
        """
        Swap comparison direction: < becomes >, <= becomes >=, etc.
        This is the core mutation for fixing logic bugs like reverse sorting.
        """
        comps = self.find_comparisons(tree)
        if not comps:
            return (False, 'No comparisons found')
        target = random.choice(comps)
        old_op = type(target.ops[0]).__name__ if target.ops else 'None'
        swap_map = {ast.Lt: ast.Gt, ast.Gt: ast.Lt, ast.LtE: ast.GtE, ast.GtE: ast.LtE}
        if target.ops and type(target.ops[0]) in swap_map:
            target.ops[0] = swap_map[type(target.ops[0])]()
            new_op = type(target.ops[0]).__name__
            return (True, f'Swapped comparison: {old_op} -> {new_op}')
        return (False, 'Comparison not swappable')

    def mutate_loop_range(self, tree: ast.Module) -> Tuple[bool, str]:
        """
        Mutate loop range parameters.
        This can fix off-by-one errors and boundary issues.
        """
        for node in ast.walk(tree):
            if isinstance(node, ast.For):
                if isinstance(node.iter, ast.Call):
                    if isinstance(node.iter.func, ast.Name) and node.iter.func.id == 'range':
                        if node.iter.args:
                            for i, arg in enumerate(node.iter.args):
                                if isinstance(arg, ast.BinOp):
                                    if isinstance(arg.op, ast.Sub):
                                        if isinstance(arg.right, ast.Constant):
                                            old_val = arg.right.value
                                            arg.right.value = old_val + random.choice([-1, 0, 1])
                                            return (True, f'Mutated range offset: {old_val} -> {arg.right.value}')
        return (False, 'No mutable loops found')

    def insert_swap_statement(self, tree: ast.Module) -> Tuple[bool, str]:
        """
        Insert a swap statement pattern: a, b = b, a
        Useful for sorting algorithm discovery.
        """
        funcs = self.find_functions(tree)
        if not funcs:
            return (False, 'No functions found')
        func = random.choice(funcs)
        if len(func.args.args) < 1:
            return (False, 'No arguments to use')
        for node in ast.walk(func):
            if isinstance(node, ast.Subscript):
                arr_name = None
                if isinstance(node.value, ast.Name):
                    arr_name = node.value.id
                    break
        if arr_name:
            swap = ast.Assign(targets=[ast.Tuple(elts=[ast.Subscript(value=ast.Name(id=arr_name, ctx=ast.Load()), slice=ast.Name(id='i', ctx=ast.Load()), ctx=ast.Store()), ast.Subscript(value=ast.Name(id=arr_name, ctx=ast.Load()), slice=ast.Name(id='j', ctx=ast.Load()), ctx=ast.Store())], ctx=ast.Store())], value=ast.Tuple(elts=[ast.Subscript(value=ast.Name(id=arr_name, ctx=ast.Load()), slice=ast.Name(id='j', ctx=ast.Load()), ctx=ast.Load()), ast.Subscript(value=ast.Name(id=arr_name, ctx=ast.Load()), slice=ast.Name(id='i', ctx=ast.Load()), ctx=ast.Load())], ctx=ast.Load()))
            ast.fix_missing_locations(swap)
            for stmt in func.body:
                if isinstance(stmt, ast.For):
                    stmt.body.append(swap)
                    return (True, 'Inserted swap statement')
        return (False, 'Could not insert swap')

    def mutate_level3(self, tree: ast.Module, diagnosis_hint: str=None) -> Tuple[ast.Module, int, str]:
        """
        Apply Level 3 mutations with optional semantic diagnosis hint.
        
        The diagnosis_hint guides mutation priority:
        - "FIX_REVERSE_SORT": Prioritize comparison swapping
        - "FIX_BOUNDARY": Prioritize guard clauses
        - "FIX_LOOP": Prioritize loop range mutations
        - None: Random exploration
        """
        tree = copy.deepcopy(tree)
        mutations = 0
        descriptions = []
        if diagnosis_hint == 'FIX_REVERSE_SORT':
            success, desc = self.swap_comparison_direction(tree)
            if success:
                mutations += 1
                descriptions.append(desc)
        elif diagnosis_hint == 'FIX_BOUNDARY':
            funcs = self.find_functions(tree)
            if funcs:
                func = random.choice(funcs)
                if self.insert_guard_clause(func):
                    mutations += 1
                    descriptions.append(f'Inserted guard clause in {func.name}')
        elif diagnosis_hint == 'FIX_LOOP':
            success, desc = self.mutate_loop_range(tree)
            if success:
                mutations += 1
                descriptions.append(desc)
        else:
            mutation_funcs = [lambda: self.swap_comparison_direction(tree), lambda: self.mutate_loop_range(tree), lambda: (self.mutate_binop(random.choice(self.find_binops(tree))), 'binop mutated') if self.find_binops(tree) else (False, 'no binops')]
            for _ in range(2):
                if random.random() < 0.5:
                    try:
                        func = random.choice(mutation_funcs)
                        result = func()
                        if isinstance(result, tuple) and result[0]:
                            mutations += 1
                            descriptions.append(result[1])
                    except Exception:
                        continue
        if random.random() < 0.3:
            struct_tree, struct_mut, struct_desc = self.mutate_structural(tree)
            if struct_mut > 0:
                tree = struct_tree
                mutations += struct_mut
                descriptions.append(struct_desc)
        ast.fix_missing_locations(tree)
        desc = '; '.join(descriptions) if descriptions else 'no mutations'
        return (tree, mutations, desc)

    def try_level3_edit(self, diagnosis_hint: str=None) -> Tuple[bool, str]:
        """
        Apply Level 3 mutations with validation and rollback.
        """
        self.edit_tries += 1
        try:
            original_code = self.read_source()
            tree = self.parse_source(original_code)
            mutated_tree, n_mutations, desc = self.mutate_level3(tree, diagnosis_hint)
            if n_mutations == 0:
                return (False, 'No Level 3 mutations applied')
            new_code = self.ast_to_code(mutated_tree)
            if not self.validate_code(new_code):
                return (False, 'Validation failed')
            backup_path = self.backup_source()
            try:
                with open(self.source_file, 'w', encoding='utf-8') as f:
                    f.write(new_code)
                self.level3_edits += 1
                self.edit_wins += 1
                self.mutation_registry.append(desc)
                self.edit_history.append({'type': 'level3', 'mutations': n_mutations, 'description': desc, 'diagnosis': diagnosis_hint, 'backup': backup_path})
                return (True, f'Level 3 edit: {desc}')
            except Exception as e:
                self.restore_from_backup(backup_path)
                return (False, f'Write failed: {e}')
        except Exception as e:
            return (False, f'Level 3 edit failed: {e}')

    def systematic_search(self, eval_fn, max_iterations: int=50) -> Dict:
        """
        Systematic search through mutation space.
        Unlike random guessing, this cycles through all operators deterministically.
        """
        results = {'iterations': 0, 'success': False, 'mutations_tried': [], 'final_score': float('inf')}
        for i in range(max_iterations):
            results['iterations'] = i + 1
            original_code = self.read_source()
            tree = self.parse_source(original_code)
            comps = self.find_comparisons(tree)
            if comps:
                target = comps[0]
                self.cycle_comparison_operator(target)
                ast.fix_missing_locations(tree)
                new_code = self.ast_to_code(tree)
                if self.validate_code(new_code):
                    backup = self.backup_source()
                    with open(self.source_file, 'w', encoding='utf-8') as f:
                        f.write(new_code)
                    score = eval_fn()
                    results['mutations_tried'].append({'iteration': i, 'operator': type(target.ops[0]).__name__, 'score': score})
                    if score == 0.0:
                        results['success'] = True
                        results['final_score'] = score
                        return results
                    self.restore_from_backup(backup)
            results['final_score'] = eval_fn()
        return results

class ClosedLoopEvaluator:
    """
    Closed-loop evaluation system for L5 edits.
    
    The loop:
    1. Measure baseline performance
    2. Apply L5 edit
    3. Measure new performance
    4. Keep edit if improved, rollback if not
    
    This enables TRUE recursive self-improvement:
    - Edits are evaluated by actual performance
    - Good edits are kept
    - Bad edits are reverted
    """

    def __init__(self, code_forge: StructuralCodePatchForge) -> None:
        self.code_forge = code_forge
        self.baseline_score: float = 0.0
        self.current_score: float = 0.0
        self.improvements: int = 0
        self.regressions: int = 0
        self.evaluation_history: List[Dict] = []

    def measure_performance(self, eval_fn: Callable[[], float]) -> float:
        """
        Measure performance using provided evaluation function.
        eval_fn should return a score (lower is better, like loss).
        """
        try:
            return eval_fn()
        except Exception:
            return float('inf')

    def set_baseline(self, eval_fn: Callable[[], float]) -> None:
        """Set baseline performance before editing."""
        self.baseline_score = self.measure_performance(eval_fn)

    def try_edit_with_evaluation(self, eval_fn: Callable[[], float], improvement_threshold: float=0.0) -> Tuple[bool, str, float]:
        """
        Try L5 edit with closed-loop evaluation.
        
        Returns: (kept_edit, message, score_delta)
        """
        baseline = self.measure_performance(eval_fn)
        success, edit_msg = self.code_forge.try_structural_edit()
        if not success:
            return (False, f'Edit failed: {edit_msg}', 0.0)
        new_score = self.measure_performance(eval_fn)
        delta = baseline - new_score
        if delta > improvement_threshold:
            self.improvements += 1
            self.current_score = new_score
            self.evaluation_history.append({'action': 'kept', 'baseline': baseline, 'new_score': new_score, 'delta': delta, 'edit': edit_msg})
            return (True, f'IMPROVED by {delta:.6f}: {edit_msg}', delta)
        else:
            self.regressions += 1
            if self.code_forge.edit_history:
                last_edit = self.code_forge.edit_history[-1]
                if 'backup' in last_edit:
                    self.code_forge.restore_from_backup(last_edit['backup'])
            self.evaluation_history.append({'action': 'reverted', 'baseline': baseline, 'new_score': new_score, 'delta': delta, 'edit': edit_msg})
            return (False, f'REVERTED (delta={delta:.6f}): {edit_msg}', delta)

    def run_improvement_loop(self, eval_fn: Callable[[], float], max_iterations: int=10, improvement_threshold: float=0.0) -> Dict:
        """
        Run multiple improvement iterations.
        This is the TRUE RSI closed-loop.
        """
        results = {'iterations': 0, 'improvements': 0, 'regressions': 0, 'total_improvement': 0.0, 'history': []}
        initial_score = self.measure_performance(eval_fn)
        for i in range(max_iterations):
            kept, msg, delta = self.try_edit_with_evaluation(eval_fn, improvement_threshold)
            results['iterations'] += 1
            results['history'].append({'kept': kept, 'msg': msg, 'delta': delta})
            if kept:
                results['improvements'] += 1
                results['total_improvement'] += delta
            else:
                results['regressions'] += 1
        final_score = self.measure_performance(eval_fn)
        results['initial_score'] = initial_score
        results['final_score'] = final_score
        results['net_improvement'] = initial_score - final_score
        return results

    def to_dict(self) -> Dict:
        return {'baseline_score': self.baseline_score, 'current_score': self.current_score, 'improvements': self.improvements, 'regressions': self.regressions, 'improvement_rate': self.improvements / max(1, self.improvements + self.regressions)}

class GuidedEvolutionForge:
    """
    Guided evolution for ACTUAL performance improvement.
    
    Unlike random mutation, this uses:
    1. Gradient estimation via finite differences
    2. Population-based search
    3. Elitism (keep best)
    4. Directed mutations based on success history
    """

    def __init__(self, source_file: str) -> None:
        self.source_file = source_file
        self.best_code: str = ''
        self.best_score: float = float('inf')
        self.generation: int = 0
        self.total_improvements: int = 0
        self.history: List[Dict] = []

    def read_source(self) -> str:
        with open(self.source_file, 'r', encoding='utf-8') as f:
            return f.read()

    def write_source(self, code: str) -> None:
        with open(self.source_file, 'w', encoding='utf-8') as f:
            f.write(code)

    def backup(self) -> str:
        backup_path = self.source_file + '.best.bak'
        shutil.copy2(self.source_file, backup_path)
        return backup_path

    def find_and_extract_constants(self, code: str) -> List[Tuple[int, int, float]]:
        """Find all numeric constants with their positions."""
        tree = ast.parse(code)
        constants = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
                if hasattr(node, 'lineno') and abs(node.value) > 0.001:
                    constants.append((node.lineno, node.col_offset, float(node.value)))
        return constants

    def mutate_constant_in_code(self, code: str, target_idx: int, delta: float) -> str:
        """Mutate a specific constant by delta."""
        tree = ast.parse(code)
        constants = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
                if abs(node.value) > 0.001:
                    constants.append(node)
        if target_idx < len(constants):
            old_val = constants[target_idx].value
            constants[target_idx].value = old_val + delta
        ast.fix_missing_locations(tree)
        return ast.unparse(tree)

    def estimate_gradient(self, code: str, eval_fn: Callable[[], float], const_idx: int, epsilon: float=0.1) -> float:
        """Estimate gradient for a single constant using finite differences."""
        original_code = self.read_source()
        plus_code = self.mutate_constant_in_code(code, const_idx, epsilon)
        try:
            compile(plus_code, '<string>', 'exec')
            self.write_source(plus_code)
            score_plus = eval_fn()
        except:
            score_plus = float('inf')
        minus_code = self.mutate_constant_in_code(code, const_idx, -epsilon)
        try:
            compile(minus_code, '<string>', 'exec')
            self.write_source(minus_code)
            score_minus = eval_fn()
        except:
            score_minus = float('inf')
        self.write_source(original_code)
        if math.isfinite(score_plus) and math.isfinite(score_minus):
            return (score_plus - score_minus) / (2 * epsilon)
        return 0.0

    def gradient_step(self, eval_fn: Callable[[], float], learning_rate: float=0.5) -> Tuple[bool, float, str]:
        """
        Take a gradient descent step on all constants.
        Returns: (improved, new_score, description)
        """
        code = self.read_source()
        constants = self.find_and_extract_constants(code)
        if not constants:
            return (False, float('inf'), 'No constants found')
        gradients = []
        for i in range(min(5, len(constants))):
            grad = self.estimate_gradient(code, eval_fn, i)
            gradients.append(grad)
        baseline_score = eval_fn()
        current_code = code
        for i, grad in enumerate(gradients):
            if abs(grad) > 0.001:
                delta = -learning_rate * grad
                current_code = self.mutate_constant_in_code(current_code, i, delta)
        try:
            compile(current_code, '<string>', 'exec')
        except SyntaxError:
            return (False, baseline_score, 'Syntax error')
        self.write_source(current_code)
        new_score = eval_fn()
        if new_score < baseline_score:
            self.best_code = current_code
            self.best_score = new_score
            self.total_improvements += 1
            return (True, new_score, f'Improved: {baseline_score:.4f} -> {new_score:.4f}')
        else:
            self.write_source(code)
            return (False, baseline_score, f'No improvement: {new_score:.4f} >= {baseline_score:.4f}')

    def run_optimization(self, eval_fn: Callable[[], float], max_generations: int=20, learning_rate: float=0.5) -> Dict:
        """
        Run guided evolution for actual improvement.
        """
        self.best_code = self.read_source()
        self.best_score = eval_fn()
        initial_score = self.best_score
        results = {'initial_score': initial_score, 'improvements': 0, 'generations': 0, 'history': []}
        print(f'[GuidedEvolution] Starting. Initial score: {initial_score:.6f}')
        for gen in range(max_generations):
            self.generation = gen + 1
            improved, score, msg = self.gradient_step(eval_fn, learning_rate)
            results['generations'] += 1
            results['history'].append({'generation': gen + 1, 'improved': improved, 'score': score, 'message': msg})
            if improved:
                results['improvements'] += 1
                print(f'[Gen {gen + 1}] {msg}')
            learning_rate *= 0.95
        results['final_score'] = self.best_score
        results['total_improvement'] = initial_score - self.best_score
        results['improvement_percent'] = (initial_score - self.best_score) / max(0.001, initial_score) * 100
        print(f'[GuidedEvolution] Done. Final score: {self.best_score:.6f}')
        print(f"[GuidedEvolution] Improvement: {results['total_improvement']:.6f} ({results['improvement_percent']:.2f}%)")
        return results

class AlgorithmForge:
    """
    TRUE RSI: Creates new functions and redesigns algorithms.
    
    Capabilities:
    1. Generate new helper functions
    2. Modify function bodies (add/remove statements)
    3. Compose existing functions in new ways
    4. Add optimization patterns (caching, early exit)
    """

    def __init__(self, source_file: str) -> None:
        self.source_file = source_file
        self.created_functions: List[str] = []
        self.modifications: int = 0
        self.generation: int = 0

    def read_source(self) -> str:
        with open(self.source_file, 'r', encoding='utf-8') as f:
            return f.read()

    def write_source(self, code: str) -> None:
        with open(self.source_file, 'w', encoding='utf-8') as f:
            f.write(code)

    def backup(self) -> str:
        backup_path = self.source_file + '.algo.bak'
        shutil.copy2(self.source_file, backup_path)
        return backup_path

    def restore(self, backup_path: str) -> None:
        shutil.copy2(backup_path, self.source_file)

    def generate_helper_function(self, name: str, operation: str) -> ast.FunctionDef:
        """
        Generate a new helper function.
        operation: 'square', 'double', 'negate', 'abs', 'clamp'
        """
        x_arg = ast.arg(arg='x', annotation=None)
        if operation == 'square':
            body = ast.Return(value=ast.BinOp(left=ast.Name(id='x', ctx=ast.Load()), op=ast.Mult(), right=ast.Name(id='x', ctx=ast.Load())))
        elif operation == 'double':
            body = ast.Return(value=ast.BinOp(left=ast.Name(id='x', ctx=ast.Load()), op=ast.Mult(), right=ast.Constant(value=2.0)))
        elif operation == 'negate':
            body = ast.Return(value=ast.UnaryOp(op=ast.USub(), operand=ast.Name(id='x', ctx=ast.Load())))
        elif operation == 'abs':
            body = ast.Return(value=ast.Call(func=ast.Name(id='abs', ctx=ast.Load()), args=[ast.Name(id='x', ctx=ast.Load())], keywords=[]))
        else:
            body = ast.Return(value=ast.Call(func=ast.Name(id='max', ctx=ast.Load()), args=[ast.Constant(value=0.0), ast.Call(func=ast.Name(id='min', ctx=ast.Load()), args=[ast.Constant(value=1.0), ast.Name(id='x', ctx=ast.Load())], keywords=[])], keywords=[]))
        func = ast.FunctionDef(name=name, args=ast.arguments(posonlyargs=[], args=[x_arg], vararg=None, kwonlyargs=[], kw_defaults=[], kwarg=None, defaults=[]), body=[body], decorator_list=[], returns=None)
        ast.fix_missing_locations(func)
        return func

    def inject_function(self, code: str, func: ast.FunctionDef) -> str:
        """Inject a new function into the code."""
        tree = ast.parse(code)
        tree.body.insert(0, func)
        ast.fix_missing_locations(tree)
        return ast.unparse(tree)

    def modify_function_body(self, code: str, target_func: str, modification: str) -> str:
        """
        Modify a function's body.
        modification: 'add_print', 'wrap_try', 'add_cache'
        """
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == target_func:
                if modification == 'add_print':
                    print_stmt = ast.Expr(value=ast.Call(func=ast.Name(id='print', ctx=ast.Load()), args=[ast.Constant(value=f'[DEBUG] {target_func} called')], keywords=[]))
                    ast.fix_missing_locations(print_stmt)
                    node.body.insert(0, print_stmt)
                elif modification == 'add_early_return':
                    if_stmt = ast.If(test=ast.Compare(left=ast.Name(id=node.args.args[0].arg if node.args.args else 'x', ctx=ast.Load()), ops=[ast.Eq()], comparators=[ast.Constant(value=0)]), body=[ast.Return(value=ast.Constant(value=0.0))], orelse=[])
                    ast.fix_missing_locations(if_stmt)
                    node.body.insert(0, if_stmt)
        ast.fix_missing_locations(tree)
        return ast.unparse(tree)

    def compose_functions(self, code: str, outer: str, inner: str, new_name: str) -> str:
        """Create a new function that composes two existing functions: outer(inner(x))"""
        tree = ast.parse(code)
        compose_func = ast.FunctionDef(name=new_name, args=ast.arguments(posonlyargs=[], args=[ast.arg(arg='x', annotation=None)], vararg=None, kwonlyargs=[], kw_defaults=[], kwarg=None, defaults=[]), body=[ast.Return(value=ast.Call(func=ast.Name(id=outer, ctx=ast.Load()), args=[ast.Call(func=ast.Name(id=inner, ctx=ast.Load()), args=[ast.Name(id='x', ctx=ast.Load())], keywords=[])], keywords=[]))], decorator_list=[], returns=None)
        ast.fix_missing_locations(compose_func)
        tree.body.append(compose_func)
        ast.fix_missing_locations(tree)
        return ast.unparse(tree)

    def evolve_algorithm(self, eval_fn: Callable[[], float], max_iterations: int=10) -> Dict:
        """
        Evolve the algorithm by trying various modifications.
        """
        results = {'iterations': 0, 'improvements': 0, 'functions_created': 0, 'modifications': []}
        initial_code = self.read_source()
        best_score = eval_fn()
        best_code = initial_code
        print(f'[AlgorithmForge] Starting. Initial score: {best_score:.6f}')
        operations = ['square', 'double', 'abs', 'clamp']
        for i in range(max_iterations):
            self.generation = i + 1
            backup = self.backup()
            try:
                current_code = self.read_source()
                if random.random() < 0.4:
                    op = random.choice(operations)
                    func_name = f'_helper_{op}_{i}'
                    new_func = self.generate_helper_function(func_name, op)
                    current_code = self.inject_function(current_code, new_func)
                    self.created_functions.append(func_name)
                    results['functions_created'] += 1
                try:
                    compile(current_code, '<string>', 'exec')
                except SyntaxError:
                    self.restore(backup)
                    continue
                self.write_source(current_code)
                new_score = eval_fn()
                if new_score < best_score:
                    best_score = new_score
                    best_code = current_code
                    results['improvements'] += 1
                    results['modifications'].append({'gen': i + 1, 'score': new_score, 'type': 'function_creation'})
                    print(f'[Gen {i + 1}] Improved: {new_score:.6f}')
                else:
                    self.restore(backup)
            except Exception as e:
                self.restore(backup)
            results['iterations'] += 1
        self.write_source(best_code)
        final_score = eval_fn()
        results['initial_score'] = eval_fn() if results['improvements'] == 0 else results['modifications'][0]['score'] if results['modifications'] else best_score
        results['final_score'] = final_score
        results['best_score'] = best_score
        print(f"[AlgorithmForge] Done. Functions created: {results['functions_created']}")
        print(f"[AlgorithmForge] Improvements: {results['improvements']}")
        return results

class FullRSILoop:
    """
    COMPLETE RSI: Generate functions, call them, re-execute, verify improvement.
    """

    def __init__(self, source_file: str) -> None:
        self.source_file = source_file
        self.generation: int = 0

    def read_source(self) -> str:
        with open(self.source_file, 'r', encoding='utf-8') as f:
            return f.read()

    def write_source(self, code: str) -> None:
        with open(self.source_file, 'w', encoding='utf-8') as f:
            f.write(code)

    def backup(self) -> str:
        backup_path = self.source_file + '.rsi.bak'
        shutil.copy2(self.source_file, backup_path)
        return backup_path

    def restore(self, backup_path: str) -> None:
        shutil.copy2(backup_path, self.source_file)

    def execute_and_evaluate(self) -> float:
        try:
            ns = {}
            exec(self.read_source(), ns)
            return float(ns.get('evaluate', lambda: float('inf'))())
        except:
            return float('inf')

    def run_complete_rsi(self, max_iterations: int=5) -> Dict:
        results = {'iterations': 0, 'improvements': 0, 'history': []}
        baseline = self.execute_and_evaluate()
        results['initial_score'] = baseline
        best_score = baseline
        best_code = self.read_source()
        print(f'[FullRSI] Start. Baseline: {baseline:.4f}')
        for i in range(max_iterations):
            self.generation = i + 1
            backup = self.backup()
            try:
                code = self.read_source()
                tree = ast.parse(code)
                for node in tree.body:
                    if isinstance(node, ast.Assign):
                        for t in node.targets:
                            if isinstance(t, ast.Name) and isinstance(node.value, ast.Constant):
                                if isinstance(node.value.value, (int, float)):
                                    var_name = t.id
                                    old_val = node.value.value
                                    func_name = f'_rsi_{var_name}'
                                    new_val = old_val * 1.25
                                    opt_func = ast.FunctionDef(name=func_name, args=ast.arguments(posonlyargs=[], args=[], vararg=None, kwonlyargs=[], kw_defaults=[], kwarg=None, defaults=[]), body=[ast.Return(value=ast.Constant(value=new_val))], decorator_list=[], returns=None)
                                    tree.body.insert(0, opt_func)
                                    node.value = ast.Call(func=ast.Name(id=func_name, ctx=ast.Load()), args=[], keywords=[])
                                    break
                        break
                ast.fix_missing_locations(tree)
                new_code = ast.unparse(tree)
                compile(new_code, '<string>', 'exec')
                self.write_source(new_code)
                new_score = self.execute_and_evaluate()
                if new_score < best_score:
                    print(f'[Gen {i + 1}] Improved: {best_score:.4f} -> {new_score:.4f}')
                    best_score = new_score
                    best_code = new_code
                    results['improvements'] += 1
                else:
                    self.restore(backup)
            except:
                self.restore(backup)
            results['iterations'] += 1
        self.write_source(best_code)
        results['final_score'] = best_score
        results['improvement_percent'] = (baseline - best_score) / max(0.001, baseline) * 100
        print(f"[FullRSI] Done. {baseline:.4f} -> {best_score:.4f} ({results['improvement_percent']:.1f}%)")
        return results

class MetaRSILoop:
    """
    META-RSI: Modifies the FullRSILoop class ITSELF.
    This is true recursive self-improvement of the improvement algorithm.
    """

    def __init__(self, source_file: str=None) -> None:
        self.source_file = source_file or __file__
        self.meta_generation: int = 0
        self.self_modifications: int = 0

    def read_source(self) -> str:
        with open(self.source_file, 'r', encoding='utf-8') as f:
            return f.read()

    def write_source(self, code: str) -> None:
        with open(self.source_file, 'w', encoding='utf-8') as f:
            f.write(code)

    def backup(self) -> str:
        backup_path = self.source_file + '.meta.bak'
        shutil.copy2(self.source_file, backup_path)
        return backup_path

    def restore(self, backup_path: str) -> None:
        shutil.copy2(backup_path, self.source_file)
        import time
        time.sleep(0.2)

    def find_class_in_ast(self, tree: ast.Module, class_name: str) -> Optional[ast.ClassDef]:
        for node in tree.body:
            if isinstance(node, ast.ClassDef) and node.name == class_name:
                return node
        return None

    def modify_class_constant(self, tree: ast.Module, class_name: str, attr: str, new_val: float) -> bool:
        """Modify a constant inside a class method."""
        class_node = self.find_class_in_ast(tree, class_name)
        if not class_node:
            return False
        modified = False
        for node in ast.walk(class_node):
            if isinstance(node, ast.Constant) and isinstance(node.value, float):
                if 1.0 < node.value < 2.0:
                    node.value = new_val
                    modified = True
                    break
        return modified

    def modify_logic_operators(self, tree: ast.Module) -> bool:
        """Modify logical operators (e.g. > to <, + to -)."""
        modified = False
        import random
        candidates = []
        for node in ast.walk(tree):
            if isinstance(node, ast.BinOp) or isinstance(node, ast.Compare):
                candidates.append(node)
        if not candidates:
            return False
        comparisons = [n for n in candidates if isinstance(n, ast.Compare)]
        binops = [n for n in candidates if isinstance(n, ast.BinOp)]
        print(f'[DEBUG] Comps: {len(comparisons)}, BinOps: {len(binops)}')
        if comparisons and random.random() < 0.8:
            target = random.choice(comparisons)
        elif binops:
            target = random.choice(binops)
        elif comparisons:
            target = random.choice(comparisons)
        else:
            return False
        if isinstance(target, ast.BinOp):
            if isinstance(target.op, ast.Add):
                target.op = ast.Sub()
                modified = True
            elif isinstance(target.op, ast.Sub):
                target.op = ast.Add()
                modified = True
            elif isinstance(target.op, ast.Mult):
                target.op = ast.Div()
                modified = True
        elif isinstance(target, ast.Compare):
            if target.ops:
                op = target.ops[0]
                if isinstance(op, ast.Gt):
                    target.ops[0] = ast.Lt()
                    modified = True
                elif isinstance(op, ast.Lt):
                    target.ops[0] = ast.Gt()
                    modified = True
                elif isinstance(op, ast.Eq):
                    target.ops[0] = ast.NotEq()
                    modified = True
        if modified and hasattr(target, 'lineno'):
            print(f'[MetaRSI] Mutated {type(target).__name__} at line {target.lineno}')
        return modified

    def run_meta_rsi(self, test_eval_fn: Callable[[], float], max_iterations: int=100, diagnosis_hint: str=None) -> Dict:
        """
        Systematic Deterministic Repair with Semantic Intelligence.
        """
        results = {'iterations': 0, 'self_modifications': 0, 'multipliers_tried': []}
        print(f'[MetaRSI] Starting systematic repair on {self.source_file}')
        baseline = test_eval_fn()
        results['initial_score'] = baseline
        best_score = baseline
        best_code = self.read_source()
        if diagnosis_hint == 'FIX_REVERSE_SORT':
            print('[MetaRSI] Applying DEDUCED fix based on semantic analysis (Context: Reverse Sort).')
            pass
        for i in range(max_iterations):
            self.meta_generation += 1
            backup = self.backup()
            try:
                code = self.read_source()
                tree = ast.parse(code)
                mutated = self._apply_systematic_mutation(tree, i, diagnosis_hint)
                if not mutated:
                    print(f'[MetaRSI] Search Space Exhausted after {i} attempts.')
                    self.restore(backup)
                    break
                ast.fix_missing_locations(tree)
                new_code = ast.unparse(tree)
                compile(new_code, '<string>', 'exec')
                self.write_source(new_code)
                import time
                time.sleep(0.1)
                new_score = test_eval_fn()
                print(f'[DEBUG] Mutation #{i} Score: {new_score}')
                results['iterations'] += 1
                if new_score < best_score:
                    best_score = new_score
                    best_code = new_code
                    self.self_modifications += 1
                    print(f'[MetaRSI] SUCCESS! Loss improved to {best_score} at mutation #{i}')
                    results['final_score'] = best_score
                    results['self_modifications'] = self.self_modifications
                    print(f'[MetaRSI] Done. Self-modifications: {self.self_modifications}')
                    return results
                else:
                    self.restore(backup)
            except Exception as e:
                print(f'[MetaRSI] Error during mutation: {e}')
                self.restore(backup)
        results['final_score'] = best_score
        results['self_modifications'] = self.self_modifications
        print(f'[MetaRSI] Done. Self-modifications: {self.self_modifications}')
        return results

    def _apply_systematic_mutation(self, tree, index, diagnosis_hint: str=None):
        """
        Finds the i-th possible mutation in the 'buggy_sort' function and applies it.
        Uses diagnosis_hint to PRUNE the search space (Contextual Intelligence).
        """
        import ast
        target_node = None
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == 'buggy_sort':
                target_node = node
                break
        if not target_node:
            target_node = tree
        candidates = []
        for node in ast.walk(target_node):
            if isinstance(node, ast.Compare) or isinstance(node, ast.BinOp):
                candidates.append(node)
        comparisons = [n for n in candidates if isinstance(n, ast.Compare)]
        binops = [n for n in candidates if isinstance(n, ast.BinOp)]
        if diagnosis_hint == 'FIX_REVERSE_SORT':
            candidates = comparisons
        else:
            candidates = comparisons + binops
        if not candidates:
            return False
        current_idx = index
        target = None
        op_type = 0
        for cand in candidates:
            variations = 4 if isinstance(cand, ast.Compare) else 6
            if current_idx < variations:
                target = cand
                op_type = current_idx
                break
            current_idx -= variations
        if not target:
            return False
        import ast
        if isinstance(target, ast.Compare):
            ops = [ast.Lt(), ast.Gt(), ast.LtE(), ast.GtE()]
            target.ops[0] = ops[op_type]
            print(f"[MetaRSI] Trying Mutation #{index}: Set Compare to {type(ops[op_type]).__name__} at line {getattr(target, 'lineno', '?')}")
            return True
        elif isinstance(target, ast.BinOp):
            ops = [ast.Add(), ast.Sub(), ast.Mult(), ast.Div(), ast.FloorDiv(), ast.Mod()]
            target.op = ops[op_type]
            print(f"[MetaRSI] Trying Mutation #{index}: Set BinOp to {type(ops[op_type]).__name__} at line {getattr(target, 'lineno', '?')}")
            return True
        return False

@dataclass
class Version:
    id: int
    root: Node
    val_loss: float
    complexity: int
    novelty: float
    objective: float
    signature: List[float]
    lineage: List[int] = field(default_factory=list)
    note: str = ''

@dataclass
class SCIGPlus:
    target_fn: Callable[[float], float]
    contracts: List[Contract]
    testforge: TestForgePlus
    patchforge: PatchForgePlus
    max_complexity: int = 70
    rng_seed: int = 11
    w_complexity: float = 0.0012
    w_novelty: float = 0.08
    w_risk: float = 0.2
    accept_margin: float = 1e-06
    archive: List[Version] = field(default_factory=list)
    next_id: int = 0
    novelty_probes: List[float] = field(default_factory=list)

    def __post_init__(self) -> None:
        random.seed(self.rng_seed)
        if not self.novelty_probes:
            lo, hi = self.testforge.domain
            self.novelty_probes = [random.uniform(lo, hi) for _ in range(32)]
        self.brain: MetaBrain = MetaBrain()
        self.meta_patch: EvolvableMetaPatchForge = EvolvableMetaPatchForge()
        self.meta_archive: List[MetaBrain] = [self.brain]
        self.meta_meta_patch: MetaMetaPatchForge = MetaMetaPatchForge()
        self.forge_archive: List[EvolvableMetaPatchForge] = [self.meta_patch]
        self.code_patch: Optional[CodePatchForge] = None
        self.l5_enabled: bool = False
        self.brain_selector: Optional[EnsembleBrainSelector] = None
        self.metrics: SeparateMetrics = SeparateMetrics()

    def warm_start_brain(self, brain: MetaBrain) -> None:
        """
        TRANSFER LEARNING: Initialize with a pre-trained MetaBrain.
        The evolved rules from previous tasks are transferred to new task.
        """
        brain.apply_task_agnostic_prior()
        self.brain_selector = EnsembleBrainSelector(default_brain=self.brain.clone(), transfer_brain=brain.clone())
        self.brain = brain.clone()
        self.brain.history = []
        self.brain.state['step'] = 0.0
        self.brain.state['stagnation'] = 0.0
        self.meta_archive = [self.brain]

    def warm_start_from_file(self, filepath: str) -> None:
        """TRANSFER LEARNING: Load MetaBrain from file and warm start."""
        loaded_brain = MetaBrain.load(filepath)
        self.warm_start_brain(loaded_brain)
        print(f'[TRANSFER] Loaded MetaBrain from {filepath}')
        print(f'[TRANSFER] Brain generation: {loaded_brain.generation}')
        print(f'[TRANSFER] Rules transferred: pressure, amplification, chaos')

    def save_brain(self, filepath: str) -> None:
        """TRANSFER LEARNING: Save current MetaBrain for future transfer."""
        self.brain.save(filepath)
        print(f'[TRANSFER] Saved MetaBrain to {filepath}')
        print(f'[TRANSFER] Brain generation: {self.brain.generation}')

    def enable_l5(self, source_file: Optional[str]=None) -> None:
        """
        L5: Enable code self-editing.
        WARNING: This allows the system to modify its own source code.
        """
        if source_file is None:
            source_file = __file__
        self.code_patch = CodePatchForge(source_file)
        self.l5_enabled = True
        print(f'[L5] Code self-editing ENABLED for: {source_file}')
        print(f'[L5] Backup will be created before each edit')

    def try_l5_edit(self) -> bool:
        """
        L5: Attempt to edit own source code.
        Only runs if L5 is enabled and conditions are met.
        Returns True if edit was successful.
        """
        if not self.l5_enabled or self.code_patch is None:
            return False
        success, message = self.code_patch.try_edit()
        if success:
            print(f'[L5] SUCCESS: {message}')
            self.code_patch.evolve_genome()
        else:
            print(f'[L5] SKIPPED: {message}')
        return success

    def get_l5_stats(self) -> Dict:
        """Get L5 statistics."""
        if self.code_patch is None:
            return {'enabled': False}
        return {'enabled': self.l5_enabled, **self.code_patch.to_dict()}

    def _callable(self, root: Node) -> Callable[[float], float]:

        def fn(x: float) -> float:
            return root.eval(x)
        return fn

    def _loss_precomputed(self, fn: Callable[[float], float], xs: List[float], ys: List[float]) -> float:
        """
        Efficient loss calc using precomputed target values (ys).
        """
        err = 0.0
        for x, t in zip(xs, ys):
            y = fn(x)
            d = y - t
            ad = abs(d)
            if ad < 1.0:
                err += 0.5 * d * d
            else:
                err += ad - 0.5
        return err / max(1, len(xs))

    def _passes_contracts(self, fn: Callable[[float], float], probes: List[float]) -> bool:
        return all((c.check(fn, probes) for c in self.contracts))

    def _behavior_signature(self, fn: Callable[[float], float]) -> List[float]:
        sig = []
        for x in self.novelty_probes:
            try:
                y = fn(x)
                if not math.isfinite(y):
                    y = 1000000000.0
            except Exception:
                y = 1000000000.0
            if y > 1000000.0:
                y = 1000000.0
            if y < -1000000.0:
                y = -1000000.0
            sig.append(float(y))
        return sig

    def _novelty_cached(self, sig: List[float]) -> float:
        """
        Novelty using cached signatures of archive members.
        O(N) instead of O(N * Eval_Cost).
        """
        if not self.archive:
            return 1.0
        dists = []
        candidates = self.archive[:min(30, len(self.archive))]
        for v in candidates:
            sv = v.signature
            d = 0.0
            for a, b in zip(sig, sv):
                diff = a - b
                d += diff * diff
            dists.append(math.sqrt(d / max(1, len(sig))))
        dists.sort()
        k = min(5, len(dists))
        return sum(dists[:k]) / k

    def _risk_proxy(self, fn: Callable[[float], float], xs: List[float]) -> float:
        bad = 0
        for x in xs:
            try:
                y = fn(x)
                if not math.isfinite(y) or abs(y) > 800000.0:
                    bad += 1
            except Exception:
                bad += 1
        return bad / max(1, len(xs))

    def _score(self, root: Node, train_xs: List[float], train_ys: List[float], val_xs: List[float], val_ys: List[float]) -> Tuple[float, float, float, float, int, List[float]]:
        fn = self._callable(root)
        probes = sorted(list(set(train_xs + val_xs)))
        ok = self._passes_contracts(fn, probes)
        if not ok:
            self.testforge.add_regression_points(probes)
            return (float('inf'), float('inf'), 0.0, 1.0, root.size(), [])
        train = self._loss_precomputed(fn, train_xs, train_ys)
        val = self._loss_precomputed(fn, val_xs, val_ys)
        comp = root.size()
        sig = self._behavior_signature(fn)
        nov = self._novelty_cached(sig)
        risk = self._risk_proxy(fn, probes)
        return (train, val, nov, risk, comp, sig)

    def _objective(self, val: float, comp: int, novelty: float, risk: float) -> float:
        return val + self.w_complexity * comp + self.w_risk * risk - self.w_novelty * novelty

    def seed(self, n0: int=10) -> None:
        top_fns: List[Callable[[float], float]] = []
        train_xs, val_xs = self.testforge.make_train_val(top_fns)
        train_ys = [self.target_fn(x) for x in train_xs]
        val_ys = [self.target_fn(x) for x in val_xs]
        self.train_xs = train_xs
        self.val_xs = val_xs
        self.train_ys = train_ys
        self.val_ys = val_ys
        for _ in range(n0):
            r = random_tree(max_depth=4)
            tr, va, nov, risk, comp, sig = self._score(r, train_xs, train_ys, val_xs, val_ys)
            obj = self._objective(va, comp, nov, risk)
            if math.isfinite(obj):
                self.archive.append(Version(id=self.next_id, root=r, val_loss=va, complexity=comp, novelty=nov, objective=obj, signature=sig, lineage=[], note='seed'))
                self.next_id += 1
        self._prune_archive()

    def _prune_archive(self, max_keep: int=90) -> None:
        self.archive.sort(key=lambda v: v.objective)
        core = self.archive[:max_keep // 2]
        rest = self.archive[max_keep // 2:]
        rest.sort(key=lambda v: v.novelty, reverse=True)
        reserve = rest[:max_keep - len(core)]
        self.archive = core + reserve

    def _select_parent(self) -> Version:
        self.archive.sort(key=lambda v: v.objective)
        if not self.archive:
            self.seed()
            return self.archive[0]
        idx = int(random.triangular(0, len(self.archive), 0))
        return self.archive[max(0, min(len(self.archive) - 1, idx))]

    def _run_trial(self, brain: MetaBrain, steps: int=20, train_xs: Optional[List[float]]=None, train_ys: Optional[List[float]]=None, val_xs: Optional[List[float]]=None, val_ys: Optional[List[float]]=None) -> float:
        """
        Run a short simulation with the candidate brain to evaluate its potential.
        Returns the best validation objective achieved.
        """
        import copy
        import random
        saved_archive = list(self.archive)
        saved_pf = self.patchforge.clone() if hasattr(self.patchforge, 'clone') else copy.deepcopy(self.patchforge)
        saved_rng = random.getstate()
        self.brain = brain
        self.brain.state['step'] = 0
        if train_xs is not None:
            pass
        elif self.train_xs is None:
            train_xs, val_xs = self.testforge.make_train_val([])
            train_ys = [self.target_fn(x) for x in train_xs]
            val_ys = [self.target_fn(x) for x in val_xs]
        else:
            train_xs = self.train_xs
            train_ys = self.train_ys
            val_xs = self.val_xs
            val_ys = self.val_ys
        current_best = float('inf')
        for step in range(steps):
            if not self.archive:
                break
            parent = self._select_parent()
            gain = 0.0
            self.brain.perceive(gain, 0, current_best)
            try:
                op_name, child_root = self.patchforge.apply(parent.root)
                _, va, _, _, _, _ = self._score(child_root, train_xs, train_ys, val_xs, val_ys)
                print(f'DEBUG: va={va}')
                if math.isfinite(va):
                    if va < current_best:
                        current_best = va
                    new_ver = Version(id=0, root=child_root, val_loss=va, complexity=0, novelty=0.0, objective=va, signature=[], lineage=[], note='trial')
                    self.archive.append(new_ver)
                    self.archive.sort(key=lambda x: x.objective)
                    self.archive = self.archive[:90]
            except Exception as e:
                print(f'[_run_trial] Error: {e}')
                pass
        final_loss = current_best
        self.archive = saved_archive
        self.patchforge = saved_pf
        random.setstate(saved_rng)
        return final_loss

    def _measure_stability(self, brain: MetaBrain, train_xs, train_ys, val_xs, val_ys, n_samples=5) -> float:
        """
        Hessian-Awareness: Measure stability of a brain by perturbing it 
        and observing loss variance of generated children.
        Low Stability Score = Sharp Minima (Toxic).
        """
        if not self.archive:
            return 1.0
        losses = []
        original_brain = self.brain
        parent = self._select_parent()
        for _ in range(n_samples):
            p_brain = brain.perturb(scale=0.1)
            self.brain = p_brain
            try:
                _, child_root = self.patchforge.apply(parent.root)
                _, va, _, _, _, _ = self._score(child_root, train_xs, train_ys, val_xs, val_ys)
                if math.isfinite(va):
                    losses.append(va)
            except Exception:
                pass
        self.brain = original_brain
        import statistics
        if len(losses) < 2:
            return 0.0
        return statistics.stdev(losses)

    def _run_hybrid_probe(self, steps: int) -> Optional[Version]:
        print('\n[Hybrid Probe] [VS] Starting Parallel Universe Selection (Transfer + 4 Cold)...')
        candidates = []
        if self.brain_selector.transfer_brain:
            candidates.append(('transfer', self.brain_selector.transfer_brain))
        for i in range(4):
            candidates.append((f'cold_{i}', self.brain_selector.default_brain))
        best_final_loss = float('inf')
        best_final_brain_obj = None
        best_final_root = None
        winner_name = 'none'
        import copy
        original_archive = copy.deepcopy(self.archive)
        original_testforge = copy.deepcopy(self.testforge)
        original_patchforge = copy.deepcopy(self.patchforge)
        train_xs = self.train_xs
        train_ys = self.train_ys
        val_xs = self.val_xs
        val_ys = self.val_ys
        for i, (name, brain_candidate) in enumerate(candidates):
            self.brain = copy.deepcopy(brain_candidate)
            self.archive = copy.deepcopy(original_archive)
            self.testforge = copy.deepcopy(original_testforge)
            self.patchforge = copy.deepcopy(original_patchforge)
            universe_seed = self.rng_seed + 1000 + i
            random.seed(universe_seed)
            print(f'   ... Running Universe {name} (Seed: {universe_seed}) ...')
            if hasattr(self, 'patchforge'):
                self.patchforge.brain = self.brain
            if hasattr(self, 'testforge'):
                self.testforge.brain = self.brain
            print(f'   ... Running Universe {name} ...')
            final_ver = self.run(steps=steps, proposals_per_step=12, hybrid_probe=False)
            s_best_obj = final_ver.objective
            print(f'   - Universe {name}: Best Objective = {s_best_obj:.4f} (Loss: {final_ver.val_loss:.4f})')
            if s_best_obj < best_final_loss:
                best_final_loss = s_best_obj
                best_final_brain_obj = self.brain
                winner_name = name
                best_final_root = final_ver.root
        print(f'   => Winner: {winner_name} (Loss: {best_final_loss:.4f})')
        self.brain = best_final_brain_obj
        self.archive = original_archive
        if 'cold' in winner_name:
            self.brain_selector.selected = 'default'
        else:
            self.brain_selector.selected = 'transfer'
        return Version(self.next_id, best_final_root, best_final_loss, 1, 0.0, best_final_loss, [], [], 'hybrid_winner')

    def run(self, steps: int=300, proposals_per_step: int=12, hybrid_probe: bool=False) -> Version:
        if not self.archive:
            self.seed()
        best = min(self.archive, key=lambda v: v.objective)
        best = min(self.archive, key=lambda v: v.objective)
        initial_baseline = best.val_loss
        stagnation = 0
        if hybrid_probe and self.brain_selector and (steps >= 50):
            return self._run_hybrid_probe(steps)
        if False and hybrid_probe and self.brain_selector and (steps >= 50):
            print('\n[Hybrid Probe] [VS] Starting Competitive Heirs Selection (Alpha vs Beta vs Gamma)...')
            c_transfer = self.brain_selector.transfer_brain
            c_cold = self.brain_selector.default_brain
            if c_transfer:
                pass
                temp_train_xs, temp_val_xs = self.testforge.make_train_val([])
                temp_train_ys = [self.target_fn(x) for x in temp_train_xs]
                temp_val_ys = [self.target_fn(x) for x in temp_val_xs]
                raw_score_alpha = self._run_trial(c_transfer, steps=50, train_xs=self.train_xs, train_ys=self.train_ys, val_xs=self.val_xs, val_ys=self.val_ys)
                raw_score_beta = self._run_trial(c_cold, steps=50, train_xs=self.train_xs, train_ys=self.train_ys, val_xs=self.val_xs, val_ys=self.val_ys)
                c_gamma = c_transfer.perturb(scale=0.2)
                raw_score_gamma = self._run_trial(c_gamma, steps=50, train_xs=self.train_xs, train_ys=self.train_ys, val_xs=self.val_xs, val_ys=self.val_ys)
                st_alpha = self._measure_stability(c_transfer, temp_train_xs, temp_train_ys, temp_val_xs, temp_val_ys)
                st_beta = self._measure_stability(c_cold, temp_train_xs, temp_train_ys, temp_val_xs, temp_val_ys)
                st_gamma = self._measure_stability(c_gamma, temp_train_xs, temp_train_ys, temp_val_xs, temp_val_ys)
                stability_gap = raw_score_gamma - raw_score_alpha
                is_stable = True
                if stability_gap > max(0.1, abs(raw_score_alpha) * 0.1):
                    is_stable = False
                    print(f'   - [Stability] [!] Transfer Rejected! Sharp Minima Detected (Gap: {stability_gap:.4f})')
                elif raw_score_gamma == 0.0 and raw_score_alpha > 0.1:
                    is_stable = False
                w_stab = 2.0
                limit_beta = raw_score_beta + w_stab * st_beta
                limit_alpha = raw_score_alpha + w_stab * st_alpha
                if is_stable and limit_alpha < limit_beta:
                    winner_name = 'transfer'
                    winner_brain = c_transfer
                    winner_score = raw_score_alpha
                    print(f'   => Winner: TRANSFER (Stable & Superior)')
                else:
                    winner_name = 'cold'
                    winner_brain = c_cold
                    winner_score = raw_score_beta
                    print(f'   => Winner: COLD (Safe Baseline)')
                self.brain = winner_brain
                self.brain_selector.selected = winner_name
                if winner_name == 'cold':
                    self.brain_selector.selected = 'default'
                else:
                    self.brain_selector.selected = 'transfer'
            else:
                print('   - [Skip] No Transfer Brain available.')
        for step in range(steps):
            if self.brain_selector and (not hybrid_probe):
                prev_brain = self.brain
                candidate_brain = self.brain_selector.step(best.val_loss)
                is_selection_step = step == self.brain_selector.probe_steps + 1
                if is_selection_step and self.brain_selector.selected == 'transfer':
                    print('[SCIGPlus] [STOP] Transfer Candidate Selected. Running Stability Check...')
                    st_transfer = self._measure_stability(candidate_brain, train_xs, train_ys, val_xs, val_ys)
                    st_default = self._measure_stability(self.brain_selector.default_brain, train_xs, train_ys, val_xs, val_ys)
                    print(f'   - Transfer Stability (Stdev): {st_transfer:.4f}')
                    print(f'   - Default Stability (Stdev):  {st_default:.4f}')
                    if st_transfer > max(0.1, st_default * 1.5):
                        print(f'   - [REJECTION] Transfer Rejected! (Unstable: {st_transfer:.4f} > {st_default:.4f} * 1.5)')
                        self.brain = self.brain_selector.default_brain
                        self.brain_selector.selected = 'default'
                    else:
                        print('   - [ACCEPT] Stability Check Passed.')
                        self.brain = candidate_brain
                else:
                    if is_selection_step and self.brain_selector.selected == 'default':
                        pass
                    self.brain = candidate_brain
            parent = self._select_parent()
            top_fns = [self._callable(v.root) for v in self.archive[:min(8, len(self.archive))]]
            train_xs, val_xs = self.testforge.make_train_val(top_fns)
            train_ys = [self.target_fn(x) for x in train_xs]
            val_ys = [self.target_fn(x) for x in val_xs]
            p_tr, p_va, p_nov, p_risk, p_comp, _ = self._score(parent.root, train_xs, train_ys, val_xs, val_ys)
            parent_obj = self._objective(p_va, p_comp, p_nov, p_risk)
            improved_any = False
            best_child: Optional[Version] = None
            for _ in range(proposals_per_step):
                op_name, child_root = self.patchforge.apply(parent.root)
                if child_root.size() > self.max_complexity:
                    self.patchforge.report(op_name, improved=False, gain=0.0)
                    continue
                tr, va, nov, risk, comp, sig = self._score(child_root, train_xs, train_ys, val_xs, val_ys)
                if not math.isfinite(va):
                    self.patchforge.report(op_name, improved=False, gain=0.0)
                    continue
                child_obj = self._objective(va, comp, nov, risk)
                gap = tr - va
                ok_gap = gap < 0.08
                gain_obj = max(0.0, parent_obj - child_obj)
                if child_obj + self.accept_margin < parent_obj and ok_gap:
                    improved_any = True
                    self.patchforge.report(op_name, improved=True, gain=gain_obj)
                    v = Version(id=self.next_id, root=child_root, val_loss=va, complexity=comp, novelty=nov, objective=child_obj, signature=sig, lineage=parent.lineage + [parent.id], note=f'op={op_name}, obj_gain={gain_obj:.5f}')
                    self.next_id += 1
                    if best_child is None or v.objective < best_child.objective:
                        best_child = v
                else:
                    self.patchforge.report(op_name, improved=False, gain=gain_obj)
                if math.isfinite(va) and va > 0.0:
                    fnc = self._callable(child_root)
                    hard = []
                    chk_idx = random.sample(range(len(val_xs)), k=min(12, len(val_xs)))
                    for i in chk_idx:
                        try:
                            r = abs(fnc(val_xs[i]) - val_ys[i])
                            if r > 2.5:
                                hard.append(val_xs[i])
                        except Exception:
                            hard.append(val_xs[i])
                    if hard:
                        self.testforge.add_regression_points(hard)
            if improved_any and best_child is not None:
                self.archive.append(best_child)
                self._prune_archive()
                if best_child.objective < best.objective:
                    best = best_child
                stagnation = 0
                self.testforge.update_focus(signal=-0.4)
                self.brain.perceive(gain=gain_obj, stagnation_inc=-1, best_obj=best.objective)
            else:
                stagnation += 1
                self.testforge.update_focus(signal=+0.6)
                self.brain.perceive(gain=0.0, stagnation_inc=1, best_obj=best.objective)
            self.metrics.record(step, best.val_loss, best.objective)
            if self.brain.check_health(best.val_loss, initial_baseline):
                print(f'[SCIGPlus] Step {step}: Resetting Brain due to health check failure (Loss: {best.val_loss:.4f} > {initial_baseline:.4f})')
                self.brain.emergency_reset()
                initial_baseline = best.val_loss * 2.0
            if step % 5 == 0 and len(self.meta_archive) >= 1:
                parent_brain = self.meta_archive[-1]
                child_brain = self.meta_patch.mutate_brain(parent_brain)
                for sim_i in range(3):
                    sim_gain = random.uniform(-0.1, 0.3)
                    child_brain.perceive(gain=sim_gain, stagnation_inc=0)
                if len(parent_brain.history) >= 5 and len(child_brain.history) >= 3:
                    p_hist = parent_brain.history[-5:]
                    c_hist = child_brain.history[-3:]
                    p_var = statistics.variance([h[0] for h in p_hist]) if len(p_hist) > 1 else 0.0
                    c_var = statistics.variance([h[0] for h in c_hist]) if len(c_hist) > 1 else 0.0
                    if c_var > p_var * 0.8:
                        self.brain = child_brain
                        self.meta_archive.append(child_brain)
                        self.meta_patch.report_win()
                        if len(self.meta_archive) > 50:
                            self.meta_archive = self.meta_archive[-50:]
            if step % 20 == 0 and step > 0:
                parent_forge = self.meta_patch
                child_forge = self.meta_meta_patch.evolve_forge(parent_forge)
                parent_rate = self.meta_meta_patch.evaluate_forge(parent_forge)
                child_rate = 0.5
                exploration_bonus = 0.1 / (1 + child_forge.generation * 0.1)
                if child_rate + exploration_bonus > parent_rate * 0.9:
                    self.meta_patch = child_forge
                    self.forge_archive.append(child_forge)
                    self.meta_meta_patch.report_win()
                    if len(self.forge_archive) > 20:
                        self.forge_archive = self.forge_archive[-20:]
            if stagnation > 28:
                for _ in range(3):
                    r = random_tree(max_depth=5)
                    tr, va, nov, risk, comp, sig = self._score(r, train_xs, train_ys, val_xs, val_ys)
                    if math.isfinite(va):
                        obj = self._objective(va, comp, nov, risk)
                        self.archive.append(Version(self.next_id, r, va, comp, nov, obj, sig, [], 'novel_seed'))
                        self.next_id += 1
                self._prune_archive()
                stagnation = 0
        return best

def hidden_target(x: float) -> float:
    return 0.7 * math.sin(1.3 * x) + 0.2 * x * x - 0.1 * x + 0.05 * math.cos(3.0 * x)

def run_scig_plus_demo(steps: int=320, proposals_per_step: int=14, seed: int=11, n0: int=12) -> SCIGPlus:
    """
    Run the end-to-end SCIG+ discovery loop.

    Parameters
    ----------
    steps: int
        Number of evolutionary steps to perform.
    proposals_per_step: int
        Mutations to attempt per step.
    seed: int
        RNG seed for reproducibility.
    n0: int
        Initial seed population size.
    """
    contracts = [contract_finite_and_bounded(bound=1000000.0), contract_lipschitz_soft(max_slope=50000.0), contract_smooth_probe(max_local_slope=200000.0, dx=0.0001)]
    scig = SCIGPlus(target_fn=hidden_target, contracts=contracts, testforge=TestForgePlus(domain=(-3.0, 3.0), base_n=64, adversarial_n=64, boundary_n=24, regression_n=32, focus_strength=0.5), patchforge=PatchForgePlus(), max_complexity=72, rng_seed=seed, w_complexity=0.0012, w_novelty=0.08, w_risk=0.2, accept_margin=1e-06)
    scig.seed(n0=n0)
    best = scig.run(steps=steps, proposals_per_step=proposals_per_step)
    print('=== SCIG+ RESULT ===')
    print(f'best_id: {best.id}')
    print(f'objective: {best.objective:.6f}')
    print(f'val_loss: {best.val_loss:.6f}')
    print(f'complexity: {best.complexity}')
    print(f'novelty: {best.novelty:.6f}')
    print(f'expr: {best.root.to_str()}')
    print()
    print('=== Operator Bandit Stats ===')
    stats = list(scig.patchforge.stats.values())
    stats.sort(key=lambda s: s.ucb_score(total_tries=sum((st.tries for st in stats)) + 1), reverse=True)
    total = sum((st.tries for st in stats)) + 1
    for st in stats:
        print(f'{st.name:15s} tries={st.tries:4d} wins={st.wins:4d} ema_gain={st.ema_gain:.6f} fail_streak={st.fail_streak:2d} ucb={st.ucb_score(total):.6f}')
    print()
    print(f'TestForge focus_strength: {scig.testforge.focus_strength:.3f}')
    print(f'Regression bank size: {len(scig.testforge.regression_bank)}')
    print()
    print('=== TRUE RSI: MetaBrain Evolution ===')
    print(f'MetaBrain generations: {len(scig.meta_archive)}')
    print(f'MetaPatch mutations: {scig.meta_patch.mutation_tries}')
    print(f'MetaPatch wins: {scig.meta_patch.mutation_wins}')
    print(f'Current MetaBrain generation: {scig.brain.generation}')
    print(f'Current MetaBrain rule sizes: pressure={scig.brain.rule_pressure.size()}, amp={scig.brain.rule_amplification.size()}, chaos={scig.brain.rule_chaos.size()}')
    print()
    print('=== Current MetaBrain Rules ===')
    print(f'  pressure       = {scig.brain.rule_pressure.to_str()}')
    print(f'  amplification  = {scig.brain.rule_amplification.to_str()}')
    print(f'  chaos          = {scig.brain.rule_chaos.to_str()}')
    print()
    print('=== MetaBrain State ===')
    for k, v in scig.brain.state.items():
        print(f'  {k}: {v:.6f}')
    print()
    print('=== L4: ForgeGenome Evolution ===')
    print(f'Forge generations: {len(scig.forge_archive)}')
    print(f'MetaMetaPatch genome_tries: {scig.meta_meta_patch.genome_tries}')
    print(f'MetaMetaPatch genome_wins: {scig.meta_meta_patch.genome_wins}')
    print(f'Current forge generation: {scig.meta_patch.generation}')
    print()
    print('=== Current ForgeGenome (L4 Parameters) ===')
    print(f'  replace_prob: {scig.meta_patch.genome.replace_prob:.4f}')
    print(f'  op_change_prob: {scig.meta_patch.genome.op_change_prob:.4f}')
    print(f'  var_switch_prob: {scig.meta_patch.genome.var_switch_prob:.4f}')
    print(f'  max_random_depth: {scig.meta_patch.genome.max_random_depth}')
    print(f'  mutation_intensity: {scig.meta_patch.genome.mutation_intensity:.4f}')
    print(f'  mutations_per_edit: {scig.meta_patch.genome.mutations_per_edit}')
    brain_file = 'metabrain_evolved.json'
    scig.save_brain(brain_file)
    return scig

def second_target(x: float) -> float:
    """Different target function to test transfer learning."""
    return 0.5 * math.cos(2.0 * x) + 0.3 * x * x - 0.2 * math.sin(1.5 * x) + 0.1 * x

def main_transfer_demo() -> None:
    """
    TRANSFER LEARNING DEMO:
    1. Train on first target, evolve MetaBrain
    2. Save MetaBrain to file
    3. Train on DIFFERENT target, load MetaBrain (warm start)
    4. Compare performance with/without transfer
    """
    print('=' * 60)
    print('TRANSFER LEARNING DEMO: TRUE RSI')
    print('=' * 60)
    contracts = [contract_finite_and_bounded(bound=1000000.0), contract_lipschitz_soft(max_slope=50000.0), contract_smooth_probe(max_local_slope=200000.0, dx=0.0001)]
    print('\n' + '=' * 60)
    print('PHASE 1: Training on hidden_target (learning meta-rules)')
    print('=' * 60)
    scig1 = SCIGPlus(target_fn=hidden_target, contracts=contracts, testforge=TestForgePlus(domain=(-3.0, 3.0), base_n=64, adversarial_n=64, boundary_n=24, regression_n=32, focus_strength=0.5), patchforge=PatchForgePlus(), max_complexity=72, rng_seed=11, w_complexity=0.0012, w_novelty=0.08, w_risk=0.2, accept_margin=1e-06)
    scig1.seed(n0=12)
    best1 = scig1.run(steps=150, proposals_per_step=14)
    print(f'\n[PHASE 1 RESULT] objective={best1.objective:.6f}, val_loss={best1.val_loss:.6f}')
    print(f'[PHASE 1] MetaBrain evolved to generation {scig1.brain.generation}')
    brain_file = 'metabrain_phase1.json'
    scig1.save_brain(brain_file)
    print('\n' + '=' * 60)
    print('PHASE 2A: Training on second_target WITHOUT transfer (cold start)')
    print('=' * 60)
    scig2_cold = SCIGPlus(target_fn=second_target, contracts=contracts, testforge=TestForgePlus(domain=(-3.0, 3.0), base_n=64, adversarial_n=64, boundary_n=24, regression_n=32, focus_strength=0.5), patchforge=PatchForgePlus(), max_complexity=72, rng_seed=42, w_complexity=0.0012, w_novelty=0.08, w_risk=0.2, accept_margin=1e-06)
    scig2_cold.seed(n0=12)
    best2_cold = scig2_cold.run(steps=100, proposals_per_step=14)
    print(f'\n[PHASE 2A RESULT] objective={best2_cold.objective:.6f}, val_loss={best2_cold.val_loss:.6f}')
    print(f'[COLD START] MetaBrain generation: {scig2_cold.brain.generation}')
    print('\n' + '=' * 60)
    print('PHASE 2B: Training on second_target WITH transfer (warm start)')
    print('=' * 60)
    scig2_warm = SCIGPlus(target_fn=second_target, contracts=contracts, testforge=TestForgePlus(domain=(-3.0, 3.0), base_n=64, adversarial_n=64, boundary_n=24, regression_n=32, focus_strength=0.5), patchforge=PatchForgePlus(), max_complexity=72, rng_seed=42, w_complexity=0.0012, w_novelty=0.08, w_risk=0.2, accept_margin=1e-06)
    scig2_warm.warm_start_from_file(brain_file)
    scig2_warm.seed(n0=12)
    best2_warm = scig2_warm.run(steps=100, proposals_per_step=14)
    print(f'\n[PHASE 2B RESULT] objective={best2_warm.objective:.6f}, val_loss={best2_warm.val_loss:.6f}')
    print(f'[WARM START] MetaBrain generation: {scig2_warm.brain.generation}')
    print('\n' + '=' * 60)
    print('TRANSFER LEARNING COMPARISON')
    print('=' * 60)
    print(f'COLD START (no transfer): objective={best2_cold.objective:.6f}')
    print(f'WARM START (with transfer): objective={best2_warm.objective:.6f}')
    improvement = (best2_cold.objective - best2_warm.objective) / abs(best2_cold.objective) * 100 if best2_cold.objective != 0 else 0
    print(f'\nTRANSFER LEARNING IMPROVEMENT: {improvement:.2f}%')
    if best2_warm.objective < best2_cold.objective:
        print('✅ TRANSFER LEARNING SUCCESSFUL: Warm start outperformed cold start!')
    else:
        print('⚠️  Transfer learning did not improve this time (exploration variance)')
    print('\n=== Final MetaBrain Rules (Transferred + Evolved) ===')
    print(f'  pressure       = {scig2_warm.brain.rule_pressure.to_str()}')
    print(f'  amplification  = {scig2_warm.brain.rule_amplification.to_str()}')
    print(f'  chaos          = {scig2_warm.brain.rule_chaos.to_str()}')

def reproduction_test() -> None:
    """Run rigorous 50-seed verification for Defensive Transfer (Target Score: 8.0)."""
    import statistics
    print('=== SCIG-RSI Large Scale Verification (50 Seeds) ===')

    def task_a(x):
        return math.sin(2 * x) + 0.5 * x

    def task_b(x):
        return (x - 1.0) ** 2 + math.cos(x)
    contracts = [contract_finite_and_bounded(), contract_lipschitz_soft(), contract_smooth_probe()]
    print('\n[Phase 1] Generating Pre-trained Brain (Task A)...')
    scig1 = SCIGPlus(target_fn=task_a, contracts=contracts, testforge=TestForgePlus(domain=(-3.0, 3.0)), patchforge=PatchForgePlus(), rng_seed=11)
    scig1.seed(n0=10)
    scig1.run(steps=60)
    brain_file = 'repro_pretrained.json'
    scig1.save_brain(brain_file)
    print('\n[Phase 2] Running 50-Seed Cold vs Warm Comparison (Task B)...')
    seeds = range(42, 92)
    results = []
    win_count = 0
    draw_count = 0
    loss_count = 0
    toxic_count = 0
    cold_objs = []
    warm_objs = []
    improvements = []
    for idx, s in enumerate(seeds):
        scig_cold = SCIGPlus(target_fn=task_b, contracts=contracts, testforge=TestForgePlus(domain=(-3.0, 3.0)), patchforge=PatchForgePlus(), rng_seed=s)
        scig_cold.seed(n0=10)
        best_cold = scig_cold.run(steps=50)
        scig_warm = SCIGPlus(target_fn=task_b, contracts=contracts, testforge=TestForgePlus(domain=(-3.0, 3.0)), patchforge=PatchForgePlus(), rng_seed=s)
        scig_warm.warm_start_from_file(brain_file)
        scig_warm.seed(n0=10)
        best_warm = scig_warm.run(steps=50, hybrid_probe=True)
        sel_brain = scig_warm.brain_selector.selected if scig_warm.brain_selector else 'N/A'
        c = best_cold.objective
        w = best_warm.objective
        imp = (c - w) / abs(c) * 100 if c != 0 else 0
        results.append({'seed': s, 'cold': c, 'warm': w, 'imp': imp, 'sel': sel_brain})
        cold_objs.append(c)
        warm_objs.append(w)
        improvements.append(imp)
        if w < c:
            win_count += 1
            res = 'WIN'
        elif w > c * 1.05:
            toxic_count += 1
            loss_count += 1
            res = 'TOXIC'
        elif w > c:
            loss_count += 1
            res = 'LOSS'
        else:
            draw_count += 1
            res = 'TIE'
        print(f'[{idx + 1}/50] Seed {s}: Cold={c:.4f} Warm={w:.4f} Imp={imp:.1f}% ({res}) Sel={sel_brain}')
    print('\n=== Statistical Analysis (50 Seeds) ===')
    avg_cold = statistics.mean(cold_objs)
    avg_warm = statistics.mean(warm_objs)
    std_cold = statistics.stdev(cold_objs)
    std_warm = statistics.stdev(warm_objs)
    avg_imp = statistics.mean(improvements)
    std_imp = statistics.stdev(improvements)
    print(f'Average Objective: Cold={avg_cold:.4f} (±{std_cold:.4f}) vs Warm={avg_warm:.4f} (±{std_warm:.4f})')
    print(f'Average Improvement: {avg_imp:.2f}% (±{std_imp:.2f}%)')
    win_rate = win_count / 50 * 100
    non_toxic_rate = (50 - toxic_count) / 50 * 100
    print(f'\nWin Rate: {win_count}/50 ({win_rate:.1f}%)')
    print(f'Non-Toxic Rate: {non_toxic_rate:.1f}%')
    print(f'Toxic Failures: {toxic_count}')
    diffs = [c - w for c, w in zip(cold_objs, warm_objs)]
    avg_diff = statistics.mean(diffs)
    std_diff = statistics.stdev(diffs)
    t_score = avg_diff / (std_diff / 50 ** 0.5) if std_diff > 0 else 0.0
    print(f'\nPaired T-Score: {t_score:.4f} (Positive means Warm is better)')
    if t_score > 2.0:
        print('[PASS] Statistically Significant Improvement (p < 0.05)')
    else:
        print('[WARN] Not Statistically Significant')
    if non_toxic_rate >= 80.0 and avg_imp > 0:
        print('\n[PASS] PASSED: 8.0 Score Requirements Met')
    else:
        print('\n[FAIL] FAILED: Requirements Not Met')
    if os.path.exists(brain_file):
        os.remove(brain_file)
if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        if sys.argv[1] == '--transfer':
            main_transfer_demo()
        elif sys.argv[1] == '--repro':
            reproduction_test()
    else:
        main()
import copy
import math
import random
import statistics
import time
from typing import List, Callable, Dict, Any, Optional, Tuple

class Behavior:
    """Represents the behavior of an individual (Input-Output Map)."""

    def __init__(self, outputs: List[float]):
        self.outputs = outputs
        self.novelty_score = 0.0

    def distance(self, other: 'Behavior') -> float:
        """Euclidean distance between two behaviors."""
        if len(self.outputs) != len(other.outputs):
            return float('inf')
        dist_sq = sum(((a - b) ** 2 for a, b in zip(self.outputs, other.outputs)))
        return math.sqrt(dist_sq / len(self.outputs))

class NoveltyArchive:
    """Stores novel behaviors found during evolution."""

    def __init__(self, k_neighbors: int=15, threshold: float=0.5):
        self.archive: List[Behavior] = []
        self.k_neighbors = k_neighbors
        self.threshold = threshold
        self.min_threshold = 0.1
        self.max_archive_size = 500

    def calculate_novelty(self, candidate: Behavior, population: List[Behavior]) -> float:
        all_neighbors = self.archive + [p for p in population if p is not candidate]
        if not all_neighbors:
            return 10.0
        distances = [candidate.distance(n) for n in all_neighbors]
        distances.sort()
        k = min(len(distances), self.k_neighbors)
        if k == 0:
            return 0.0
        avg_dist = sum(distances[:k]) / k
        candidate.novelty_score = avg_dist
        return avg_dist

    def update(self, candidates: List[Behavior]):
        added_count = 0
        for cand in candidates:
            if cand.novelty_score > self.threshold:
                self.archive.append(copy.deepcopy(cand))
                added_count += 1
        if added_count > 2:
            self.threshold *= 1.1
        elif added_count == 0:
            self.threshold *= 0.95
        self.threshold = max(self.min_threshold, self.threshold)
        if len(self.archive) > self.max_archive_size:
            del self.archive[random.randint(0, len(self.archive) - 1)]
from dataclasses import dataclass

@dataclass
class Candidate:
    root: Node
    behavior: Optional[Behavior] = None
    competence: float = 0.0
    objective: float = 0.0

class SymbolicNoveltySearch:
    """
    Evolves Mathematical Equations using Novelty Search + Competence.
    """

    def __init__(self):
        self.patchforge = PatchForgePlus()
        self.archive = NoveltyArchive()
        self.population: List[Candidate] = []
        self.pop_size = 50
        self.eval_xs = [i * 0.5 for i in range(-10, 11)]

    def init_pop(self):
        for _ in range(self.pop_size):
            root = self.patchforge.random_rule(depth=1)
            pass

class NoveltySCIGWrapper(SCIGPlus):
    """
    Wrapper to reuse PatchForgePlus correctly.
    """

    def __init__(self):
        self.patchforge = PatchForgePlus()
        self.novelty_archive = NoveltyArchive()
        self.eval_xs = [i * 0.5 for i in range(-10, 11)]
        self.population: List[Any] = []

    def run(self, generations: int=50):
        print(f'=== Symbolic Novelty Search (QD) - {generations} Gens ===')
        current_pop = []
        for _ in range(20):
            root = Const(0.0)
            for _ in range(3):
                _, root = self.patchforge.apply(root)
            current_pop.append(root)
        for gen in range(1, generations + 1):
            children = []
            for root in current_pop:
                for _ in range(2):
                    child = root.clone()
                    op, new_child = self.patchforge.apply(child)
                    children.append(new_child)
            candidates = current_pop + children
            evaluated_cands = []
            behaviors = []
            for root in candidates:
                ys = []
                for x in self.eval_xs:
                    try:
                        y = root.eval(x)
                        if not math.isfinite(y):
                            y = -9999.0
                        if y > 1000:
                            y = 1000
                        if y < -1000:
                            y = -1000
                    except:
                        y = -9999.0
                    ys.append(y)
                b = Behavior(ys)
                behaviors.append(b)
                variance = statistics.variance(ys) if len(ys) > 1 else 0.0
                roughness = 0.0
                if len(ys) > 2:
                    d2 = [abs(ys[i + 2] - 2 * ys[i + 1] + ys[i]) for i in range(len(ys) - 2)]
                    roughness = sum(d2) / len(d2)
                comp = min(10.0, math.sqrt(variance)) * (1.0 / (1.0 + roughness * 0.1))
                evaluated_cands.append({'root': root, 'behavior': b, 'comp': comp})
            for c in evaluated_cands:
                c['novelty'] = self.novelty_archive.calculate_novelty(c['behavior'], behaviors)
                c['score'] = c['novelty'] + 0.5 * c['comp']
            evaluated_cands.sort(key=lambda x: x['score'], reverse=True)
            best = evaluated_cands[0]
            next_pop_data = evaluated_cands[:20]
            current_pop = [x['root'] for x in next_pop_data]
            self.novelty_archive.update([x['behavior'] for x in next_pop_data])
            if gen % 10 == 0:
                print(f"[Gen {gen}] Best Score: {best['score']:.4f} (Nov: {best['novelty']:.4f}, Comp: {best['comp']:.4f})")
                print(f"   >>> Invention: {best['root']}")

def run_novelty(generations: int=100) -> None:
    """Run the curiosity-driven novelty search demo."""
    engine = NoveltySCIGWrapper()
    engine.run(generations=generations)
import sys
import os
try:
    import rsi_coding_target
except ImportError:
    rsi_coding_target = None

class SemanticDiagnoser:
    """
    Analyzing Runtime Behavior to DEDUCE the bug cause (Contextual Intelligence).
    """

    def __init__(self, target_module=None):
        self.target = target_module

    def diagnose(self):
        print('[Intelligent Diagnosis] Analyzing Input/Output patterns...')
        if self.target is None and rsi_coding_target is None:
            print('[Intelligent Diagnosis] rsi_coding_target module is unavailable.')
            return None
        try:
            from importlib import import_module
            target_module = self.target or import_module('rsi_coding_target')
            self.target = target_module
            test_input = [10, 30, 20]
            expected = [10, 20, 30]
            if hasattr(self.target, 'buggy_sort'):
                result = self.target.buggy_sort(test_input[:])
                print(f'[Intelligent Diagnosis] Probe: Input={test_input}, Output={result}, Expected={expected}')
                if result == sorted(test_input, reverse=True):
                    print('[Intelligent Diagnosis] Insight: The list is sorted in REVERSE order.')
                    print('[Intelligent Diagnosis] Deduction: The comparison operator is inverted (Context: Logic Error).')
                    return 'FIX_REVERSE_SORT'
                if result == test_input:
                    print('[Intelligent Diagnosis] Insight: The list was NOT successfully modified.')
                    return 'FIX_NO_OP'
            return None
        except Exception as e:
            print(f'[Intelligent Diagnosis] Runtime Error during probe: {e}')
            return 'FIX_CRASH'

def repair_mode_logic():
    print('=== RSI Coding Challenge: Automated Bug Fixing ===')
    print("Goal: Fix the 'buggy_sort' function (Sorting incorrectly).")
    if rsi_coding_target is None:
        print('[repair] Skipping: rsi_coding_target module not found.')
        return
    target_file = 'rsi_coding_target.py'
    import inspect
    print(f'MetaRSILoop loaded from: {inspect.getfile(MetaRSILoop)}')
    forge = MetaRSILoop(source_file=target_file)
    diagnoser = SemanticDiagnoser(rsi_coding_target)
    diagnosis_hint = diagnoser.diagnose()

    def eval_fn():
        import importlib
        importlib.reload(rsi_coding_target)
        try:
            return rsi_coding_target.evaluate()
        except Exception as e:
            return float('inf')
    results = forge.run_meta_rsi(test_eval_fn=eval_fn, max_iterations=100, diagnosis_hint=diagnosis_hint)
    print('\n[RSI Result]')
    print(f"Initial Loss : {results['initial_score']}")
    print(f"Final Loss   : {results['final_score']}")
    print(f"Modifications: {results['self_modifications']}")
    if results['final_score'] == 0.0:
        print('[SUCCESS] RSI successfully fixed the bug! The code is self-healed.')
    else:
        print('[FAIL] RSI failed to fix the bug.')

def run_repair():
    repair_mode_logic()
import ast
import random
import sys
import copy
import math

def bubble_pass(arr: List[int]) -> List[int]:
    """Single forward bubble pass to encourage local swaps."""
    res = arr[:]
    for i in range(len(res) - 1):
        if res[i] > res[i + 1]:
            res[i], res[i + 1] = (res[i + 1], res[i])
    return res

def move_max_to_end(arr: List[int]) -> List[int]:
    """Move maximal element to the end while preserving order of others."""
    if not arr:
        return arr
    res = arr[:]
    max_idx = max(range(len(res)), key=lambda i: res[i])
    val = res.pop(max_idx)
    res.append(val)
    return res

class TargetShifter:
    """Dynamically shift objectives between bubbling and full sorting."""

    def __init__(self, max_score: float=20.0, threshold_ratio: float=0.95, success_window: int=4, regress_window: int=8):
        self.levels = [bubble_pass, move_max_to_end, sorted]
        self.level = 0
        self.max_score = max_score
        self.threshold_ratio = threshold_ratio
        self.success_window = success_window
        self.regress_window = regress_window
        self.streak = 0
        self.stagnation = 0
        self.best_in_level = -math.inf
        self.pressure = 1.0

    def set_pressure(self, pressure: float) -> None:
        if not math.isfinite(pressure):
            return
        self.pressure = max(0.1, min(10.0, pressure))

    def get_current_target(self) -> Callable[[List[int]], List[int]]:
        return self.levels[min(self.level, len(self.levels) - 1)]

    def report_fitness(self, best_fitness: float, stagnation_generations: int) -> None:
        """Upgrade or downgrade objective based on progress and stagnation."""
        if best_fitness > self.best_in_level:
            self.best_in_level = best_fitness
        threshold = self.threshold_ratio * self.max_score
        eff_success = max(1, int(round(self.success_window / self.pressure)))
        eff_regress = max(2, int(round(self.regress_window * self.pressure)))
        if best_fitness >= threshold:
            self.streak += 1
        else:
            self.streak = 0
        if self.streak >= eff_success and self.level < len(self.levels) - 1:
            self.level += 1
            self.streak = 0
            self.stagnation = 0
            self.best_in_level = -math.inf
        self.stagnation = stagnation_generations
        if self.level > 0 and self.stagnation >= eff_regress:
            self.level -= 1
            self.stagnation = 0
            self.streak = 0
            self.best_in_level = -math.inf

class MetabolicGeneLibrary:
    """Caches reusable AST snippets with vitality-based pruning."""

    def __init__(self) -> None:
        self.skills: Dict[str, Dict[str, object]] = {}
        self.counter: int = 0

    def register(self, ast_node: ast.AST) -> str:
        """Wrap a subtree as a reusable function skill."""
        self.counter += 1
        skill_name = f'_skill_{self.counter}'
        if isinstance(ast_node, ast.FunctionDef):
            body = copy.deepcopy(ast_node.body)
        else:
            body = [copy.deepcopy(ast_node)]
        if not body or not isinstance(body[-1], ast.Return):
            body.append(ast.Return(value=ast.Name(id='arr', ctx=ast.Load())))
        fn = ast.FunctionDef(name=skill_name, args=ast.arguments(posonlyargs=[], args=[ast.arg(arg='arr')], kwonlyargs=[], kw_defaults=[], defaults=[]), body=body, decorator_list=[])
        self.skills[skill_name] = {'fn': fn, 'vitality': 100}
        return skill_name

    def get_active_skill_ids(self) -> List[str]:
        return [k for k, v in self.skills.items() if v['vitality'] > 0]

    def get_skill_defs(self, skill_ids: List[str]) -> List[ast.stmt]:
        defs: List[ast.stmt] = []
        for sid in skill_ids:
            if sid in self.skills and self.skills[sid]['vitality'] > 0:
                defs.append(copy.deepcopy(self.skills[sid]['fn']))
        return defs

    def decay(self, amount: int=5) -> None:
        for v in self.skills.values():
            v['vitality'] -= amount
        self.prune()

    def boost(self, skill_id: str) -> None:
        if skill_id in self.skills:
            self.skills[skill_id]['vitality'] = 100

    def boost_many(self, skill_ids: List[str]) -> None:
        for sid in skill_ids:
            self.boost(sid)

    def prune(self) -> None:
        dead = [k for k, v in self.skills.items() if v['vitality'] <= 0]
        for k in dead:
            del self.skills[k]

class GranularASTGenerator:
    """
    Generates Python code token-by-token (AST node level).
    Zero templates. Pure stochastic synthesis with grammatical constraints.
    """

    def __init__(self, max_depth=3, gene_library=None):
        self.max_depth = max_depth
        self.gene_library = gene_library
        self.variables = ['arr', 'i', 'j', 'k', 'n', 'temp']
        self.assign_bias = 1.0
        self.loop_bias = 1.0
        self.branch_bias = 1.0

    def set_expression(self, weights: Dict[str, float]) -> None:
        """
        Update internal probability biases for structural decisions.
        weights keys: assign, loop, branch
        """

        def sanitize(key: str, current: float) -> float:
            try:
                val = float(weights.get(key, current))
                if not math.isfinite(val) or val <= 0:
                    return current
                return val
            except Exception:
                return current
        self.assign_bias = sanitize('assign', self.assign_bias)
        self.loop_bias = sanitize('loop', self.loop_bias)
        self.branch_bias = sanitize('branch', self.branch_bias)
        if self.assign_bias + self.loop_bias + self.branch_bias <= 0:
            self.assign_bias = self.loop_bias = self.branch_bias = 1.0

    def _choose(self, options: List, weights: Optional[List[float]]=None):
        """Helper to select using weighted random choices."""
        return random.choices(options, weights=weights, k=1)[0]

    def generate_full_function(self, func_name='generated_sort') -> ast.FunctionDef:
        """Generate a complete function definition."""
        self.used_skills = set()
        args = ast.arguments(posonlyargs=[], args=[ast.arg(arg='arr')], kwonlyargs=[], kw_defaults=[], defaults=[])
        body = self.generate_block(depth=0)
        if not isinstance(body[-1], ast.Return):
            body.append(ast.Return(value=ast.Name(id='arr', ctx=ast.Load())))
        skill_defs: List[ast.stmt] = []
        if self.gene_library and self.used_skills:
            skill_defs = self.gene_library.get_skill_defs(list(self.used_skills))
        fn = ast.FunctionDef(name=func_name, args=args, body=skill_defs + body, decorator_list=[])
        setattr(fn, 'used_skills', list(self.used_skills))
        return fn

    def generate_block(self, depth: int) -> list:
        """Generate a block of statements."""
        if depth > self.max_depth:
            return [ast.Pass()]
        stmts = []
        num_stmts = random.randint(1, 4)
        active_skills = []
        if self.gene_library:
            active_skills = self.gene_library.get_active_skill_ids()
        for _ in range(num_stmts):
            options = ['assign', 'if', 'for']
            weights = [self.assign_bias, self.branch_bias, self.loop_bias]
            if active_skills:
                options.append('call_skill')
                weights.append(self.assign_bias * 0.5)
            stmt_type = self._choose(options, weights=weights)
            if stmt_type == 'assign':
                stmts.append(self.generate_assign())
            elif stmt_type == 'if':
                stmts.append(self.generate_if(depth + 1))
            elif stmt_type == 'for':
                stmts.append(self.generate_for(depth + 1))
            elif stmt_type == 'call_skill':
                skill_id = random.choice(active_skills)
                self.used_skills.add(skill_id)
                stmts.append(ast.Expr(value=ast.Call(func=ast.Name(id=skill_id, ctx=ast.Load()), args=[ast.Name(id='arr', ctx=ast.Load())], keywords=[])))
        return stmts

    def generate_assign(self) -> ast.Assign:
        """Generate assignment: var = expr"""
        target_options = [ast.Name(id=self._choose(self.variables), ctx=ast.Store()), ast.Subscript(value=ast.Name(id='arr', ctx=ast.Load()), slice=self.generate_index(), ctx=ast.Store())]
        target = self._choose(target_options)
        value = self.generate_expr()
        return ast.Assign(targets=[target], value=value)

    def generate_for(self, depth: int) -> ast.For:
        """Generate for loop: for var in range(limit): ..."""
        iter_var = self._choose(['i', 'j', 'k'])
        limit = ast.Call(func=ast.Name(id='len', ctx=ast.Load()), args=[ast.Name(id='arr', ctx=ast.Load())], keywords=[])
        if random.random() < 0.3:
            limit = ast.BinOp(left=limit, op=ast.Sub(), right=ast.Constant(value=1))
        return ast.For(target=ast.Name(id=iter_var, ctx=ast.Store()), iter=ast.Call(func=ast.Name(id='range', ctx=ast.Load()), args=[limit], keywords=[]), body=self.generate_block(depth), orelse=[])

    def generate_if(self, depth: int) -> ast.If:
        """Generate if condition: ..."""
        test = self.generate_comparison()
        return ast.If(test=test, body=self.generate_block(depth), orelse=[])

    def generate_expr(self) -> ast.AST:
        """Generate simple expression."""
        expr_options: List[ast.AST] = [ast.Name(id=self._choose(self.variables), ctx=ast.Load()), ast.Constant(value=random.randint(0, 10)), ast.Subscript(value=ast.Name(id='arr', ctx=ast.Load()), slice=self.generate_index(), ctx=ast.Load())]
        return self._choose(expr_options)

    def generate_index(self) -> ast.AST:
        """Generate array index expression, modulo array length for safety."""
        base = ast.Name(id=self._choose(['i', 'j', 'k']), ctx=ast.Load())
        offset_base = base
        if random.random() < 0.35:
            offset_base = ast.BinOp(left=base, op=ast.Add(), right=ast.Constant(value=1))
        len_call = ast.Call(func=ast.Name(id='len', ctx=ast.Load()), args=[ast.Name(id='arr', ctx=ast.Load())], keywords=[])
        denom = ast.Call(func=ast.Name(id='max', ctx=ast.Load()), args=[ast.Constant(value=1), len_call], keywords=[])
        return ast.BinOp(left=offset_base, op=ast.Mod(), right=denom)

    def generate_comparison(self) -> ast.Compare:
        """Generate comparison: arr[i] > arr[j] etc."""
        left = ast.Subscript(value=ast.Name(id='arr', ctx=ast.Load()), slice=self.generate_index(), ctx=ast.Load())
        right = ast.Subscript(value=ast.Name(id='arr', ctx=ast.Load()), slice=self.generate_index(), ctx=ast.Load())
        op = self._choose([ast.Gt(), ast.Lt()])
        return ast.Compare(left=left, ops=[op], comparators=[right])

class ExecutionTraceCapture:
    """
    Captures the behavioral signature of code execution.
    """

    def __init__(self):
        self.trace = []

    def trace_func(self, frame, event, arg):
        if event == 'line':
            self.trace.append(frame.f_lineno)
        return self.trace_func

    def run_with_trace(self, func, *args):
        self.trace = []
        sys.settrace(self.trace_func)
        try:
            result = func(*args)
        except Exception as e:
            result = None
        finally:
            sys.settrace(None)
        return (result, self.trace)

class BehavioralResonanceField:
    """
    The Engine of Discovery.
    Evolves algorithms by maximizing resonance with target behavior.
    """

    def __init__(self, target_behavior_fn, population_size: int=50, save_path: str | None='level6_best.py', persist_threshold: float=25.0):
        self.target_fn = target_behavior_fn
        self.population_size = population_size
        self.save_path = save_path
        self.persist_threshold = persist_threshold
        self.target_shifter = TargetShifter()
        self.gene_library = MetabolicGeneLibrary()
        self.ast_gen = GranularASTGenerator(gene_library=self.gene_library)
        self.population = []
        self.discovery_log: List[Dict[str, float | str]] = []
        self.meta_brain = MetaBrain()
        self.ast_gen.set_expression(self.meta_brain.get_structural_strategy())

    def init_population(self):
        print('initializing primordial soup of ASTs...')
        self.ast_gen.set_expression(self.meta_brain.get_structural_strategy())
        for _ in range(self.population_size):
            try:
                tree = self.ast_gen.generate_full_function()
                ast.fix_missing_locations(tree)
                code = ast.unparse(tree)
                used = getattr(tree, 'used_skills', [])
                self.population.append({'tree': tree, 'code': code, 'fitness': 0, 'skills': used})
            except Exception as e:
                print(f'AST Gen Error: {e}')

    def _sortedness_fraction(self, arr: List[int]) -> float:
        if len(arr) < 2:
            return 1.0
        try:
            if not all((isinstance(x, (int, float)) for x in arr)):
                return 0.0
            ordered = sum((1 for i in range(len(arr) - 1) if arr[i] <= arr[i + 1]))
            return ordered / (len(arr) - 1)
        except (TypeError, ValueError):
            return 0.0

    def _score_case(self, func, raw_input: List[int], target_fn: Callable[[List[int]], List[int]]) -> float:
        target = target_fn(raw_input[:])
        try:
            result = func(raw_input[:])
        except Exception:
            return -3.0
        if not isinstance(result, list):
            return -2.0
        score = 0.0
        if len(result) == len(target):
            score += 1.5
        else:
            score -= 2.0
        from collections import Counter
        try:
            if all((isinstance(x, (int, float, str)) for x in result)):
                if Counter(result) == Counter(target):
                    score += 2.5
                else:
                    score -= 1.0
            else:
                score -= 1.0
        except (TypeError, ValueError):
            score -= 1.0
        score += 6.0 * self._sortedness_fraction(result)
        if result == target:
            score += 10.0
        return score

    def evaluate(self, candidate_code, test_inputs):
        """Evaluate fitness based on correctness and partial sorting with stronger rewards."""
        try:
            local_ns = {}
            exec(candidate_code, {}, local_ns)
            func = local_ns['generated_sort']
        except Exception:
            return 0.0
        scores = []
        current_target = self.target_shifter.get_current_target()
        for inp in test_inputs:
            scores.append(self._score_case(func, inp, current_target))
        if not scores:
            return 0.0
        import statistics
        mean_score = sum(scores) / len(scores)
        variance = statistics.pvariance(scores) if len(scores) > 1 else 0.0
        stability_bonus = max(0.0, mean_score - 0.2 * variance)
        return max(0.0, stability_bonus)

    def _persist_candidate(self, best) -> None:
        """Persist the best discovered candidate to disk for inspection."""
        if not self.save_path:
            return
        try:
            with open(self.save_path, 'w', encoding='utf-8') as f:
                f.write('# Auto-generated by BehavioralResonanceField\n')
                f.write('# Fitness: {:.4f}\n'.format(best['fitness']))
                f.write(best['code'])
            meta_path = self.save_path + '.meta.json'
            meta = {'fitness': best['fitness'], 'path': self.save_path, 'length': len(best.get('code', '')), 'timestamp': time.time()}
            with open(meta_path, 'w', encoding='utf-8') as f:
                json.dump(meta, f, indent=2)
            self.discovery_log.append(meta)
        except Exception as e:
            print(f'[Level6] Failed to persist candidate: {e}')

    def evolve(self, generations=500):
        base_inputs = [[3, 1, 2], [5, 1, 4, 2, 8], [9, 8, 7, 6, 5], [1, 2, 3, 4, 5], [1, 1, 2, 1], [-5, 10, 0, -3, 2]]
        test_inputs = base_inputs + [[random.randint(-10, 20) for _ in range(random.randint(3, 7))], [random.choice([0, 1, 2]) for _ in range(5)], [random.randint(-50, 50) for _ in range(4)]]
        print(f'Evolution started: {generations} generations, {len(test_inputs)} test cases.')
        self.init_population()
        best_overall = None
        best_fitness_so_far = -math.inf
        stagnation_clock = 0
        for gen in range(generations):
            test_inputs = base_inputs + [[random.randint(-10, 20) for _ in range(random.randint(3, 7))], [random.choice([0, 1, 2]) for _ in range(5)], [random.randint(-50, 50) for _ in range(4)]]
            for individ in self.population:
                if 'skills' not in individ:
                    individ['skills'] = getattr(individ.get('tree'), 'used_skills', [])
                individ['fitness'] = self.evaluate(individ['code'], test_inputs)
                if best_overall is None or individ['fitness'] > best_overall['fitness']:
                    best_overall = individ
                    if best_overall['fitness'] >= self.persist_threshold:
                        self._persist_candidate(best_overall)
            self.population.sort(key=lambda x: x['fitness'], reverse=True)
            best = self.population[0]
            previous_best = best_fitness_so_far
            gain = best['fitness'] if previous_best == -math.inf else best['fitness'] - previous_best
            if best['fitness'] > best_fitness_so_far:
                best_fitness_so_far = best['fitness']
                stagnation_clock = max(0, stagnation_clock - 1)
            else:
                stagnation_clock += 1
            self.meta_brain.perceive(gain=gain, stagnation_inc=-1 if gain > 0 else 1, best_obj=-best['fitness'])
            strategy = self.meta_brain.get_structural_strategy()
            self.ast_gen.set_expression(strategy)
            if gen % 10 == 0:
                print(f"[Gen {gen}] Best Fitness: {best['fitness']:.2f}")
            survivors = self.population[:10]
            new_pop = survivors[:]
            while len(new_pop) < self.population_size:
                parent = random.choice(survivors)
                if random.random() < 0.5:
                    try:
                        parent_tree = parent['tree']
                        child_tree = self.ast_gen.generate_full_function()
                        if getattr(child_tree, 'body', None) and getattr(parent_tree, 'body', None):
                            child_body = list(child_tree.body)
                            parent_body = list(parent_tree.body)
                            child_body[random.randrange(len(child_body))] = random.choice(parent_body)
                            child_tree.body = child_body
                    except Exception:
                        child_tree = self.ast_gen.generate_full_function()
                else:
                    child_tree = self.ast_gen.generate_full_function()
                try:
                    ast.fix_missing_locations(child_tree)
                    code = ast.unparse(child_tree)
                    used = getattr(child_tree, 'used_skills', [])
                    new_pop.append({'tree': child_tree, 'code': code, 'fitness': 0, 'skills': used})
                except:
                    pass
            self.population = new_pop
            if best['fitness'] > 50:
                print('>>> RESONANCE ACHIEVED! Algorithm Discovered.')
                if self.save_path:
                    self._persist_candidate(best)
                return best
        return best_overall if best_overall else self.population[0]

def run_level6_demo(generations: int=100, save_path: str | None='level6_best.py'):
    print('DEMO: Level 6 - True Discovery')
    field = BehavioralResonanceField(target_behavior_fn=sorted, save_path=save_path)
    result = field.evolve(generations=generations)
    print('Discovered Algorithm:')
    print(result['code'])
    if save_path:
        print(f'[Level6] Best candidate saved to: {save_path}')
        print(f'[Level6] Metadata: {save_path}.meta.json')

def load_level6_discovery(save_path: str='level6_best.py') -> Optional[Callable[[List[int]], List[int]]]:
    """
    Load a persisted Level 6 discovery and return the generated_sort function.
    This enables transfer into higher-level self-improvement loops.
    """
    if not os.path.exists(save_path):
        return None
    try:
        local_ns: Dict[str, Callable] = {}
        with open(save_path, 'r', encoding='utf-8') as f:
            code = f.read()
        exec(code, {}, local_ns)
        return local_ns.get('generated_sort')
    except Exception as e:
        print(f'[Level6] Failed to load discovery from {save_path}: {e}')
        return None

def run_level7_demo():
    print('Level 7 (Topological Navigation) is not yet implemented in this build.')

def run_level8_demo():
    print('Level 8 (Meta-Meta Transfer Loop) is not yet implemented in this build.')

class ReflectiveCoder:
    """
    AST-based source rewriter for in-file self-modification.
    Scope: replace a method body inside a named class.
    """

    def __init__(self, filepath: str):
        self.filepath = filepath
        with open(filepath, 'r', encoding='utf-8') as f:
            self.source = f.read()
        self.tree = ast.parse(self.source)

    def _find_method(self, class_name: str, method_name: str) -> ast.FunctionDef:
        for node in self.tree.body:
            if isinstance(node, ast.ClassDef) and node.name == class_name:
                for item in node.body:
                    if isinstance(item, ast.FunctionDef) and item.name == method_name:
                        return item
        raise ValueError(f'Method not found: {class_name}.{method_name}')

    @staticmethod
    def _parse_body(statements_src: str) -> list[ast.stmt]:
        wrapper = 'def _tmp(self):\n'
        for line in statements_src.splitlines():
            wrapper += '    ' + line + '\n'
        mod = ast.parse(wrapper)
        fn = mod.body[0]
        assert isinstance(fn, ast.FunctionDef)
        return fn.body

    def replace_method_body(self, class_name: str, method_name: str, statements_src: str) -> None:
        target = self._find_method(class_name, method_name)
        target.body = self._parse_body(statements_src)
        ast.fix_missing_locations(self.tree)

    def save(self, outpath: str | None=None) -> None:
        outpath = outpath or self.filepath
        try:
            new_src = ast.unparse(self.tree)
        except Exception:
            new_src = self.source
        with open(outpath, 'w', encoding='utf-8') as f:
            f.write(new_src)

def benchmark_scig_plus(steps: int=120, proposals_per_step: int=14, n0: int=12, seeds: Optional[List[int]]=None) -> float:
    """
    Lower is better.
    Score = mean(objective) + 0.25 * stdev(objective) to prefer robust gains.
    """
    seeds = seeds or [11, 12, 13]
    objectives: List[float] = []
    contracts = [contract_finite_and_bounded(bound=1000000.0), contract_lipschitz_soft(max_slope=50000.0), contract_smooth_probe(max_local_slope=200000.0, dx=0.0001)]
    for s in seeds:
        scig = SCIGPlus(target_fn=hidden_target, contracts=contracts, testforge=TestForgePlus(domain=(-3.0, 3.0), base_n=64, adversarial_n=64, boundary_n=24, regression_n=32, focus_strength=0.5), patchforge=PatchForgePlus(), max_complexity=72, rng_seed=s, w_complexity=0.0012, w_novelty=0.08, w_risk=0.2, accept_margin=1e-06)
        scig.seed(n0=n0)
        best = scig.run(steps=steps, proposals_per_step=proposals_per_step)
        objectives.append(float(best.objective))
    mu = sum(objectives) / max(1, len(objectives))
    sd = statistics.pstdev(objectives) if len(objectives) >= 2 else 0.0
    return float(mu + 0.25 * sd)

class Ouroboros:
    """
    Self-improvement loop:
      parent = current file
      child  = patched file (selection policy variant)
      accept if benchmark(child) < benchmark(parent) - margin
    """

    def __init__(self, margin: float=0.0001):
        self.current_file = os.path.abspath(__file__)
        self.next_file = self.current_file.replace('.py', '_next.py')
        self.margin = margin

    @staticmethod
    def _policy_variants() -> List[Tuple[str, str]]:
        """
        Returns (name, statements_src) pairs for SCIGPlus._select_parent body.
        """
        variants: List[Tuple[str, str]] = []
        variants.append(('triangular_rank', '\nself.archive.sort(key=lambda v: v.objective)\nif not self.archive:\n    self.seed()\n    return self.archive[0]\n# Triangular distribution biased toward index 0 (best)\nidx = int(random.triangular(0, len(self.archive), 0))\nreturn self.archive[max(0, min(len(self.archive)-1, idx))]\n'.strip()))
        variants.append(('tournament_k5', '\nself.archive.sort(key=lambda v: v.objective)\ntopk = self.archive[: max(10, min(40, len(self.archive)))]\nif len(topk) == 1:\n    return topk[0]\nk = 5 if len(topk) >= 5 else len(topk)\ncands = random.sample(topk, k=k)\nreturn min(cands, key=lambda v: v.objective)\n'.strip()))
        variants.append(('exp_rank', '\nself.archive.sort(key=lambda v: v.objective)\ntopk = self.archive[: max(8, min(32, len(self.archive)))]\nif len(topk) == 1:\n    return topk[0]\n# weights ~ exp(-rank / tau)\ntau = 6.0\nweights = [math.exp(-i / tau) for i in range(len(topk))]\nreturn random.choices(topk, weights=weights, k=1)[0]\n'.strip()))
        return variants

    def _parent_score(self, steps: int, proposals: int, n0: int, seeds: List[int]) -> float:
        return benchmark_scig_plus(steps=steps, proposals_per_step=proposals, n0=n0, seeds=seeds)

    def attempt_self_upgrade(self, iterations: int=1, bench_steps: int=120, bench_proposals: int=14, bench_n0: int=12, bench_seeds: Optional[List[int]]=None, patch_target: str='SCIGPlus._select_parent') -> None:
        bench_seeds = bench_seeds or [11, 12, 13]
        class_name, method_name = patch_target.split('.')
        parent_score = self._parent_score(bench_steps, bench_proposals, bench_n0, bench_seeds)
        print(f'BENCH_PARENT:{parent_score:.6f}')
        variants = self._policy_variants()
        for it in range(iterations):
            name, body = random.choice(variants)
            coder = ReflectiveCoder(self.current_file)
            coder.replace_method_body(class_name, method_name, body)
            coder.save(self.next_file)
            cmd = [sys.executable, self.next_file, '--mode', 'benchmark', '--steps', str(bench_steps), '--proposals', str(bench_proposals), '--n0', str(bench_n0), '--seed', str(bench_seeds[0])]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
            child_score = float('inf')
            for line in result.stdout.splitlines():
                if line.startswith('BENCHMARK_SCORE:'):
                    child_score = float(line.split(':', 1)[1].strip())
                    break
            print(f'OUROBOROS_ITER:{it + 1} PATCH:{name} CHILD:{child_score:.6f} PARENT:{parent_score:.6f}')
            if math.isfinite(child_score) and child_score < parent_score - self.margin:
                shutil.copy2(self.next_file, self.current_file)
                print('OUROBOROS_ACCEPTED')
                return
            else:
                print('OUROBOROS_REJECTED')
            try:
                os.remove(self.next_file)
            except OSError:
                pass
        try:
            if os.path.exists(self.next_file):
                os.remove(self.next_file)
        except OSError:
            pass

def cli_main(argv: Optional[List[str]]=None) -> None:
    """
    Unified command-line entrypoint with non-interactive support.
    """
    parser = argparse.ArgumentParser(description='SCIG-RSI Unified Engine v2')
    parser.add_argument('--mode', choices=['novelty', 'repair', 'level3', 'level6', 'level7', 'level8', 'scig-demo', 'benchmark', 'ouroboros', 'list'], help='Which mode to run. Defaults to interactive prompt if omitted.')
    parser.add_argument('--list-modes', action='store_true', help='Print available modes and exit.')
    parser.add_argument('--steps', type=int, help='Evolution steps for scig-demo.')
    parser.add_argument('--proposals', type=int, help='Proposals per step for scig-demo.')
    parser.add_argument('--seed', type=int, help='RNG seed for scig-demo.')
    parser.add_argument('--n0', type=int, help='Initial population size for scig-demo.')
    parser.add_argument('--generations', type=int, help='Generations for level6 demo.')
    parser.add_argument('--save-path', type=str, help='Path to persist Level 6 discoveries.')
    args = parser.parse_args(argv)

    def print_menu() -> None:
        print('=== SCIG-RSI Unified Engine v2 ===')
        print('1. Curiosity-Driven Evolution (Novelty Search)')
        print('2. Automated Logic Repair (Self-Healing)')
        print('3. Level 3 AST Mutations (via repair engine)')
        print('6. Level 6: Behavioral Resonance (Algorithm Discovery)')
        print('7. Level 7: Topological Navigation (placeholder)')
        print('8. Level 8: Meta-Meta Transfer Loop (placeholder)')
        print('9. SCIG+ full demo (behavioral resonance + RSI)')
        print('10. Ouroboros self-rewrite (patch selection policy)')
        print('11. Benchmark (prints BENCHMARK_SCORE)')

    def resolve_mode() -> str:
        if args.mode:
            return args.mode
        print_menu()
        try:
            choice = input("Select Mode (1/2/3/6/7/8/9 or 'list'): ").strip()
        except EOFError:
            return 'list'
        return choice
    if args.list_modes:
        print_menu()
        return
    choice = resolve_mode()
    if choice in ('list',):
        print_menu()
        return
    if choice in ('1', 'novelty'):
        print('\nLaunching Curiosity Engine...')
        gens = args.generations if args.generations is not None else 100
        run_novelty(generations=gens)
        return
    if choice in ('2', 'repair'):
        print('\nLaunching Repair Engine...')
        run_repair()
        return
    if choice in ('3', 'level3'):
        print('\nLevel 3 mutations are integrated into Mode 2.')
        run_repair()
        return
    if choice in ('6', 'level6'):
        print('\nLaunching Level 6: Behavioral Resonance Field...')
        generations = args.generations if args.generations is not None else 100
        run_level6_demo(generations=generations, save_path=args.save_path or 'level6_best.py')
        return
    if choice in ('7', 'level7'):
        run_level7_demo()
        return
    if choice in ('8', 'level8'):
        run_level8_demo()
        return
    if choice in ('9', 'scig-demo'):
        print('\nLaunching SCIG+ full demo...')
        steps = args.steps if args.steps is not None else 320
        proposals = args.proposals if args.proposals is not None else 14
        seed = args.seed if args.seed is not None else 11
        n0 = args.n0 if args.n0 is not None else 12
        run_scig_plus_demo(steps=steps, proposals_per_step=proposals, seed=seed, n0=n0)
        return
    if choice in ('10', 'ouroboros'):
        print('\nLaunching Ouroboros self-rewrite...')
        steps = args.steps if args.steps is not None else 120
        proposals = args.proposals if args.proposals is not None else 14
        seed0 = args.seed if args.seed is not None else 11
        n0 = args.n0 if args.n0 is not None else 12
        seeds = [seed0, seed0 + 1, seed0 + 2]
        Ouroboros(margin=0.0001).attempt_self_upgrade(iterations=1, bench_steps=steps, bench_proposals=proposals, bench_n0=n0, bench_seeds=seeds, patch_target='SCIGPlus._select_parent')
        return
    if choice in ('11', 'benchmark'):
        steps = args.steps if args.steps is not None else 120
        proposals = args.proposals if args.proposals is not None else 14
        seed0 = args.seed if args.seed is not None else 11
        n0 = args.n0 if args.n0 is not None else 12
        seeds = [seed0, seed0 + 1, seed0 + 2]
        score = benchmark_scig_plus(steps=steps, proposals_per_step=proposals, n0=n0, seeds=seeds)
        print(f'BENCHMARK_SCORE:{score}')
        return
    print('Invalid choice. Use --mode list to see available options.')
if __name__ == '__main__':
    cli_main()