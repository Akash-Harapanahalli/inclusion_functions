import numpy as np
import interval
import sympy as sp

class InclusionFunction :
    def __init__(self, f) -> None:
        self.f = f

    def __call__ (self, x) :
        return NotImplementedError

