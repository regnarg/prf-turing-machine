from prf import *
from arith import *
from logic import *

# see graph in exam_prep.xoj
div_mul = C(mul, Proj(4,1), Proj(4,3))
div_le = C(le, div_mul, Proj(4,2))
div_step = C(cond, div_le, Proj(4,1), Proj(4,0))
div_pr = PR(Zero(2), div_step) # takes an extra #iterations argument
div = C(div_pr, C(S, Proj(2,0)), Proj(2,0), Proj(2,1)) # bound iteration number by (dividend+1)
div = accel(div, lambda x,y: (x//y if y!=0 else x), 2)

# (cur, ctr, a, b) -> new = cur < b ? cur : cur - b
mod_step = C(cond, C(lt, Proj(4,0), Proj(4,3)), Proj(4,0), C(sub, Proj(4,0), Proj(4,3)))
mod_pr = PR(Proj(2,0), mod_step)
mod = C(mod_pr, C(S, Proj(2,0)), Proj(2,0), Proj(2,1))
mod = accel(mod, lambda x,y: (x%y if y!=0 else x), 2)
