import sys, os
import random
import itertools

def is_prf(x): return bool(getattr(x, '_is_prf'))

def zero(n): return 0
zero._is_prf = True
zero._arity = 1
def S(n): return n+1
S._is_prf = True
S._arity = 1

def Proj(n, i):
    src = f'''lambda { ", ".join("arg%d"%i for i in range(n)) }: arg{i}'''
    f = eval(src)
    f._is_prf = True
    f._arity = n
    return f

def C(f, *hs):
    n = len(hs)
    #args =  ", ".join("arg%d"%i for i in range(n)) 
    #src = f'''lambda {args}: '''
    def g(*a):
        return f(*( h(*a) for h in hs ))
    g._is_prf = True
    g._arity = hs[0]._arity
    assert all( h._arity == g._arity for h in hs )
    assert f._arity == len(hs)
    return g

def PR(base, step):
    assert is_prf(base) and is_prf(step)
    def f(ctr, *a):
        cur = base(*a)
        for i in range(ctr):
            cur = step(cur, i, *a)
        return cur
    f._is_prf = True
    assert base._arity + 2 == step._arity
    f._arity = 1 + base._arity
    return f

def rand_arg_vals(n, *, range_=1000):
    assert n  > 2
    return [0, 1] + [ random.randrange(range_) for _ in range(n-2) ]

def test(f, gold, nargs, *, range_=1000, values=10):
    for args in itertools.product(*[ rand_arg_vals(values, range_=range_) for _ in range(nargs) ]):
        expected = gold(*args)
        got = f(*args)
        if expected != got:
            raise ValueError(f"Test failed for {args}: expected {expected}, got {got}")

def accel(f, acc, nargs, *, test=True, range_=1000, values=10):
    if test:
        globals()['test'](f, acc, nargs, range_=range_, values=values)
    acc._is_prf = True
    acc._orig_prf = f
    acc._arity = f._arity
    return acc

def Minimize(predicate):
    assert predicate._is_prf
    def f(*args):
        for i in itertools.count():
            if predicate(i, *args):
                return i
    f._is_prf = True
    f._arity = predicate._arity - 1
    return f



