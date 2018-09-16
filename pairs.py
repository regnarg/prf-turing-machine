from prf import *
from arith import *
from logic import *
from division import *

### TUPLES ###

# diagonal $d$ (zero-based) has offset d*(d+1)/2
diag_off = C(div,
                C(mul,
                    Proj(1,0),       # d
                    C(S, Proj(1,0)), # d + 1
                ),
                C(S, One(1)) # /2
            )
diag_off = accel(diag_off, lambda d: d*(d+1)//2, 1)

# (x+y)*(x+y+1)/2 + y = add(y, div(mul(x+y, x+y+1), S(1)))
pair = C(add,
            Proj(2,1), # y+
            C(diag_off, C(add, Proj(2,0), Proj(2,1))), # diagonal number x+y
        )
pair = accel(pair, lambda x,y: (x+y)*(x+y+1)//2 + y, 2)

# bruteforce diagonal: look for max d: diag_off(d) ≤ enc
# step: (cur, ctr, enc) -> if diag_off(ctr) ≤ enc, retrun ctr, else return cur
diag_get_step = C(cond,
                    C(le,
                        C(diag_off, Proj(3,1)),
                        Proj(3,2)
                    ),
                    Proj(3,1),
                    Proj(3,0)
                )
diag_get_pr = PR(Zero(1), diag_get_step)
diag_get = C(diag_get_pr, C(S, Proj(1,0)), Proj(1,0))
def _diag_get_acc(enc):
    for d in range(enc+2):
        if diag_off(d) > enc: return d - 1
    raise ValueError
diag_get = accel(diag_get, _diag_get_acc, 1)

# right(enc) = enc - diag_off = enc - diag_off(diag_get(enc))
right = C(sub, Proj(1,0), C(diag_off, diag_get))

# left(enc) = diag_get(enc) - right(enc)
left = C(sub, diag_get, right)

### ACCELERATE PAIRS ###

# Encoding nested pairs (will be needed for lists below) into big numbers
# gets quicky out of hand. Let's hack it to store pairs as Python tuples,
# even tough logically they are still big integers.

# Because pair codes are often compared to zero to test (0,0), we use
# a subclass of `int` that behaves as 0 for (0,0) and 1 for any other
# value. This is hackish and we assume that code makes no other assumptions
# about the numeric codes except that (0,0) has code 0.

accel_pairs = 1
if accel_pairs:
    class PairTuple(int):
        def __repr__(self):
            return '(%r %r)'%self.data
        def __eq__(self, other):
            if isinstance(other, PairTuple):
                return self.data == other.data
            elif other == 0:
                return self.data == (0,0)
            else:
                raise TypeError
        def __ne__(self, other):
            if isinstance(other, PairTuple):
                return self.data == other.data
            elif other == 0:
                return self.data != (0,0)
            else:
                raise TypeError
        __str__ = __repr__
    def pair(x,y):
        if x==0 and y==0: return 0
        pt = PairTuple(0 if x==0 and y==0 else 1)
        pt.data = (x,y)
        return pt
    pair._is_prf = True
    pair._arity = 2
    left = lambda p: 0 if p==0 else p.data[0]
    left._is_prf = True
    left._arity = 1
    right = lambda p: 0 if p==0 else p.data[1]
    right._is_prf = True
    right._arity = 1
