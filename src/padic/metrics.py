
from .field import Qp

def padic_abs(x: Qp) -> float:
    return x.abs()

def padic_dist(x: Qp, y: Qp) -> float:
    return (x.sub(y)).abs()
