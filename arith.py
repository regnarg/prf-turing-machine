from prf import *

id = Proj(1, 0)
one = C(S, zero)

### ARITHMETIC ###

S3 = C(S, Proj(3, 0))
add = PR(id, S3)
add = accel(add, lambda x,y: x+y, 2)

pred = C(PR(id, Proj(3,1)), Proj(1,0), Proj(1,0))
pred = accel(pred, lambda x: x - 1 if x > 0 else 0, 1)

pred3 = C(pred, Proj(3, 0))
rsub = PR(id, pred3)
sub = C(rsub, Proj(2,1), Proj(2,0))
sub = accel(sub, lambda x,y: x-y if x>=y else 0, 2)

mul_step = C(add, Proj(3, 0), Proj(3, 2))
mul = PR(zero, mul_step)
mul = accel(mul, lambda x,y: x*y, 2)

pow_step = C(mul, Proj(3, 0), Proj(3, 2))
rpow = PR(one, pow_step)
pow = C(rpow, Proj(2,1), Proj(2,0))
pow = accel(pow, lambda x,y: x**y, 2)
