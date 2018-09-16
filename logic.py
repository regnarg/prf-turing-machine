from arith import *

### LOGIC AND COMPARISONS ###

def Zero(nargs):
    assert nargs > 0
    return C(zero, Proj(nargs, 0))
def One(nargs):
    assert nargs > 0
    return C(one, Proj(nargs, 0))
def Constant(c, nargs):
    def f(*a):
        assert len(a) == nargs
        return c
    f._is_prf = True
    f._arity = nargs
    return f
_sgn = PR(zero, C(one, Proj(3,0)))
sgn = C(_sgn, Proj(1,0), Proj(1,0))
sgn = accel(sgn, lambda x: int(x>0), 1)

not_ = C(sub, one, Proj(1,0))
or_ = C(sgn, add)
and_ = C(sgn, mul)

gt = C(sgn, sub)
gt = accel(gt, lambda x,y: int(x>y), 2)
ge = C(gt, C(S, Proj(2,0)), Proj(2,1))
ge = accel(ge, lambda x,y: int(x>=y), 2)
lt = C(gt, Proj(2,1), Proj(2,0))
le = C(ge, Proj(2,1), Proj(2,0))

eq = C(mul, ge, le)
eq = accel(eq, lambda x,y: int(x==y), 2)
ne = C(not_, eq)
ne = accel(ne, lambda x,y: int(x!=y), 2)

### CONDITIONALS ###

# cond(x,a,b) -> (x>0)*a + (x=0)*b

true_cmp = C(gt, Proj(3, 0), Zero(3))
true_half = C(mul, true_cmp, Proj(3,1))
false_cmp = C(eq, Proj(3, 0), Zero(3))
false_half = C(mul, false_cmp, Proj(3,2))
cond = C(add, true_half, false_half)
cond = accel(cond, lambda x,y,z: y if x else z, 3)

max_ = C(cond, ge, Proj(2,0), Proj(2,1))
min_ = C(cond, le, Proj(2,0), Proj(2,1))
