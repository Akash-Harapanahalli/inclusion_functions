import numpy as np
import interval
# from interval import NotIntervalException
import sympy as sp

class InclusionFunction :
    def __init__(self, f_eqn, x_vars) -> None:
        self.f_eqn = sp.Matrix(f_eqn)
        self.x_vars = tuple(x_vars)

    def __call__ (self, x) :
        return NotImplementedError

class NaturalInclusionFunction (InclusionFunction) :
    def __init__(self, f_eqn, x_vars) -> None:
        super().__init__(f_eqn, x_vars)
        self.f = sp.lambdify((x_vars,), f_eqn, 'numpy')
    
    def __call__(self, x) :
        # if x.dtype != np.interval :
        #     raise NotIntervalException(x)
        return self.f(x)

class Ordering :
    def __init__(self, ordering) -> None:
        self.ordering = tuple(ordering)

    def __getitem__ (self, key) :
        return self.ordering[key]
    
    def __setitem__ (self, key, item) :
        self.ordering[key] = item
    
    def __repr__ (self) :
        return repr(self.ordering)
        
    def __str__ (self) :
        return str(self.ordering)
    
    def __len__(self) :
        return len(self.ordering)

class OrderingIter :
    def __init__(self, ordering) :
        self.ordering = ordering
        self.idx = 0
        self.length = len(ordering)
    
    def __iter__(self) :
        return self

    def __next__ (self) :
        if self.idx < self.length :
            ret = self.ordering[self.idx]
            self.idx += 1
            return ret
        raise StopIteration

def standard_ordering (n) :
    return [Ordering(range(n))]

def two_orderings (n) :
    return [Ordering(range(n)), Ordering(reversed(range(n)))]
