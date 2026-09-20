from __future__ import annotations
from dataclasses import dataclass
from fractions import Fraction
import math
from typing import Union

Number = Union[int, float, Fraction, "ConstructiveNumber"]


def _q(x) -> Fraction:
    if isinstance(x, Fraction):
        return x
    if isinstance(x, int):
        return Fraction(x, 1)
    if isinstance(x, float):
        if not math.isfinite(x):
            raise ValueError("value must be finite")
        return Fraction(str(x))
    return Fraction(x)


@dataclass(frozen=True)
class ConstructiveNumber:
    """Конструктивное число как рациональный интервал [a,b]."""

    a: Fraction
    b: Fraction

    def __post_init__(self):
        if self.a > self.b:
            raise ValueError("need a <= b")

    @classmethod
    def from_x_eps(cls, x: float, eps: float) -> "ConstructiveNumber":
        if eps < 0:
            raise ValueError("epsilon must be non-negative")
        return cls(_q(x - eps), _q(x + eps))

    @classmethod
    def from_pair(cls, a, b) -> "ConstructiveNumber":
        return cls(_q(a), _q(b))

    @property
    def epsilon(self) -> float:
        return float((self.b - self.a) / 2)

    @property
    def midpoint(self) -> float:
        return float((self.a + self.b) / 2)

    def value(self, alpha: float = 0.5) -> float:
        if not 0 <= alpha <= 1:
            raise ValueError("alpha must be in [0,1]")
        return float(self.a + (self.b - self.a) * _q(alpha))

    @staticmethod
    def _coerce(x) -> "ConstructiveNumber":
        if isinstance(x, ConstructiveNumber):
            return x
        q = _q(x)
        return ConstructiveNumber(q, q)

    def __repr__(self):
        return f"CN([{float(self.a):.6g}, {float(self.b):.6g}], eps={self.epsilon:.3e})"

    def __add__(self, other):
        o = self._coerce(other)
        return ConstructiveNumber(self.a + o.a, self.b + o.b)

    __radd__ = __add__

    def __neg__(self):
        return ConstructiveNumber(-self.b, -self.a)

    def __sub__(self, other):
        return self + (-self._coerce(other))

    def __rsub__(self, other):
        return self._coerce(other) - self

    def __mul__(self, other):
        o = self._coerce(other)
        values = (self.a*o.a, self.a*o.b, self.b*o.a, self.b*o.b)
        return ConstructiveNumber(min(values), max(values))

    __rmul__ = __mul__

    def __pow__(self, power):
        if power != 2:
            raise NotImplementedError("ConstructiveNumber currently supports power 2 only")
        values = (self.a*self.a, self.a*self.b, self.b*self.a, self.b*self.b)
        return ConstructiveNumber(min(values), max(values))

    def __truediv__(self, other):
        o = self._coerce(other)
        if o.a <= 0 <= o.b:
            raise ZeroDivisionError("divisor interval contains zero")
        values = (self.a/o.a, self.a/o.b, self.b/o.a, self.b/o.b)
        return ConstructiveNumber(min(values), max(values))

    def __rtruediv__(self, other):
        return self._coerce(other) / self

    def __lt__(self, other):
        o = self._coerce(other)
        return self.b < o.a

    def __le__(self, other):
        o = self._coerce(other)
        return self.b <= o.a

    def __gt__(self, other):
        o = self._coerce(other)
        return self.a > o.b

    def __ge__(self, other):
        o = self._coerce(other)
        return self.a >= o.b

    def __eq__(self, other):
        return isinstance(other, ConstructiveNumber) and self.a == other.a and self.b == other.b

    def __float__(self):
        return self.midpoint


def _interval_monotone(fn, x: ConstructiveNumber) -> ConstructiveNumber:
    return ConstructiveNumber(_q(fn(float(x.a))), _q(fn(float(x.b))))


def cn_sqrt(x):
    x = ConstructiveNumber._coerce(x)
    if x.a < 0:
        # Для sqrt(|u|) обычно отрицательных аргументов не возникает.
        # Здесь берём консервативную оболочку.
        lo = 0.0
    else:
        lo = math.sqrt(float(x.a))
    hi = math.sqrt(max(0.0, float(x.b)))
    return ConstructiveNumber(_q(lo), _q(hi))


def cn_abs(x):
    x = ConstructiveNumber._coerce(x)
    if x.a >= 0:
        return x
    if x.b <= 0:
        return ConstructiveNumber(-x.b, -x.a)
    return ConstructiveNumber(Fraction(0), max(-x.a, x.b))


def cn_sin(x):
    x = ConstructiveNumber._coerce(x)
    a, b = float(x.a), float(x.b)
    if a > b:
        a, b = b, a
    if b - a >= 2*math.pi:
        return ConstructiveNumber(Fraction(-1), Fraction(1))

    candidates = [math.sin(a), math.sin(b)]
    # Проверяем точки, где sin достигает +/-1.
    k0 = math.floor((a - math.pi/2) / math.pi) - 1
    k1 = math.ceil((b - math.pi/2) / math.pi) + 1
    for k in range(k0, k1 + 1):
        t = math.pi/2 + k*math.pi
        if a <= t <= b:
            candidates.append(math.sin(t))
    return ConstructiveNumber(_q(min(candidates)), _q(max(candidates)))


def cn_cos(x):
    x = ConstructiveNumber._coerce(x)
    return cn_sin(x + math.pi/2)


def _round_half_up(v: float) -> int:
    return math.floor(v + 0.5)


def cn_round(x):
    """Оболочка функции round для интервала.

    round монотонна, поэтому достаточно округлить левую и правую границы.
    Используется правило математического округления .5 вверх, как в Desmos.
    """
    x = ConstructiveNumber._coerce(x)
    lo = _round_half_up(float(x.a))
    hi = _round_half_up(float(x.b))
    return ConstructiveNumber.from_pair(lo, hi)
