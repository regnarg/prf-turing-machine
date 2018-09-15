#!/usr/bin/python3

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

### DIVISION ###

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

### LISTS ###

# Lists are stored using nested (head, tail) pairs like in List or Haskell
# with one twist: to allow primitive recursive iteration: the head of the
# first pair stores length of the list, items follow that.
# The only valid encoding of an empty list is (0,0), i.e., 0. (0,x) for x≥0
# is never allowed.

nonempty = sgn
empty = not_

length = left
head = C(left, right)

tail_data = C(right, right)
tail_length = C(pred, left)
tail = C(pair, tail_length, tail_data)

cons_length = C(S, C(length, Proj(2,1)))
cons_data = C(pair, Proj(2,0), C(right, Proj(2,1)))
cons = C(pair, cons_length, cons_data)

list2 = C(cons, Proj(2,0), C(cons, Proj(2,1), Zero(2)))
list3 = C(cons, Proj(3,0), C(list2, Proj(3,1), Proj(3,2)))
list4 = C(cons, Proj(4,0), C(list3, Proj(4,1), Proj(4,2), Proj(4,3)))

# List indexing
suffix_pr = PR(id, C(tail, Proj(3, 0)))
suffix = C(suffix_pr, Proj(2,1), Proj(2,0))
get = C(head, suffix)
def Getter(idx): return C(get, id, Constant(idx, 1))

def LIST(*a):
    r = 0
    for itm in reversed(a):
        r = cons(itm, r)
    return r

def UNLIST(l):
    r = []
    while l:
        r.append(head(l))
        l = tail(l)
    return r

def ListAccum(base, step, nargs, *, want_index=False):
    # PR args: list, *a
    # strip off the 'list' argument
    #list_base = C(base, *( Proj(nargs,i) for i in range(1, nargs) ))
    pr_base = C(pair, base, Proj(nargs+1, 0))
    # PR gives step the following arguments:
    # pair(cur_accum, cur_tail), index, list, *extra_args
    pr_step_args = 3 + nargs
    pr_step_cur_accum = C(left, Proj(pr_step_args,0))
    pr_step_cur_list = C(right, Proj(pr_step_args,0))
    pr_step_cur_head = C(head, pr_step_cur_list)
    pr_step_new_accum = C(step,
                    pr_step_cur_accum, # cur_accum
                    pr_step_cur_head,
                    *(
                        ([Proj(pr_step_args, 1)] if want_index else []) # index
                        + [Proj(pr_step_args,i) for i in range(3,3+nargs)] # extra args
                    )
                )
    pr_step_new_tail = C(tail, pr_step_cur_list)
    pr_step = C(pair, pr_step_new_accum, pr_step_new_tail)
    pr = PR(pr_base, pr_step)
    # pr2: Automatically set the iteration number to length of list
    # arguments: (list, *extra)
    pr2_list_arg = Proj(1+nargs, 0)
    pr2 = C(pr, C(length, pr2_list_arg), pr2_list_arg,
            *( Proj(1+nargs,i) for i in range(1, 1+nargs) ))
    # pr3: Extract only final accumulator value from return value.
    pr3 = C(left, pr2)
    return pr3

sum_ = ListAccum(Zero(1), add, 0)
rev = ListAccum(Zero(1), C(cons, Proj(2,1), Proj(2,0)), 0)

any_ = ListAccum(Zero(1), or_, 0)
all_ = ListAccum(One(1), and_, 0)

def Map(f, extra_args=0):
    # This creates the mapped list in reverse order
    r = ListAccum(
            Zero(1+extra_args),
            C(cons,
                C(f,
                    Proj(2+extra_args,1), # the item
                    *( Proj(2+extra_args, i) for i in range(2,2+extra_args) ) # extra args from map call
                ),
                Proj(2+extra_args,0)
            ),
            extra_args)
    return C(rev, r)

def Filter(f, extra_args=0):
    r = ListAccum(Zero(1+extra_args),
                  C(cond,
                      C(f,
                          Proj(2+extra_args,1), # the item
                          *( Proj(2+extra_args, i) for i in range(2,2+extra_args) ) # extra args from map call
                      ),
                      C(cons,  Proj(2+extra_args,1), Proj(2+extra_args,0)),
                      Proj(2+extra_args,0)
                  ),
                  extra_args)
    return C(rev, r)

contains = C(any_, Map(eq, 1))

### DICTIONARIES (as lists of (key,value) pairs) ###

filter_key = Filter(C(eq, C(left, Proj(2, 0)), Proj(2,1)), 1)
remove_key = Filter(C(ne, C(left, Proj(2, 0)), Proj(2,1)), 1)
has_key = C(sgn, filter_key)
lookup = C(right, C(head, filter_key))
dict_set = C(cons, C(pair, Proj(3,1), Proj(3,2)), C(remove_key, Proj(3,0), Proj(3,1)))

### TURING MACHINES ###

TM_STAY = 0
TM_LEFT = 1
TM_RIGHT = 2

# Machine description is
#   [start_state, final_states, trans_func]
# Transition function is a dictionary of
#   display -> action
#           i.e.
#   pair(state, headsym) -> [new_state, new_sym, move]
# Configuration is
#   [ state, left_halftape, right_halftape ]
# Tape head is on the first element of right_halftape.
# left_halftape is stored reversed (first element is field before head)

TM_PARITY = LIST(
                0,
                LIST(2,3), # terminating states: 2 for even parity, 3 for odd
                LIST(
                    pair( pair(0, 1), LIST(1, 1, TM_RIGHT) ),
                    pair( pair(0, 2), LIST(0, 2, TM_RIGHT) ),
                    pair( pair(1, 1), LIST(0, 1, TM_RIGHT) ),
                    pair( pair(1, 2), LIST(1, 2, TM_RIGHT) ),
                    pair( pair(0, 0), LIST(2, 0, TM_STAY) ), # terminate on first blank
                    pair( pair(1, 0), LIST(3, 0, TM_STAY) ),
                ),
            )

# machine, input -> initial config
tm_init_conf = C(list3, C(Getter(0), Proj(2,0)), Zero(2), Proj(2,1))

def tm_dump_conf(conf):
    state, left_tape, right_tape = UNLIST(conf)
    left_tape = UNLIST(left_tape)
    right_tape = UNLIST(right_tape)
    line1 = f'[{state}] {" ".join(reversed(list(map(str, left_tape))))} > {" ".join(map(str, right_tape))}'
    print(line1)

# get display (state, headsym) from config
tm_conf_disp = C(pair, Getter(0), C(head, Getter(2)))

# machine, conf -> action
tm_conf_action = C(lookup, C(Getter(2), Proj(2,0)), C(tm_conf_disp, Proj(2,1)))

tm_conf_move_left = C(list3,
                        Getter(0), # keep state
                        C(tail, Getter(1)), # left half loses symbol
                        C(cons, C(head, Getter(1)), Getter(2)), # right half gains symbol
                    )
tm_conf_move_right = C(list3,
                        Getter(0), # keep state
                        C(cons, C(head, Getter(2)), Getter(1)), # left half gains symbol
                        C(tail, Getter(2)), # right half loses symbol
                    )
# conf, direction -> new_conf
tm_conf_move = C(cond,
                        C(eq, Proj(2,1), Constant(TM_LEFT,2)),
                        C(tm_conf_move_left, Proj(2,0)),
                        C(cond,
                            C(eq, Proj(2,1), Constant(TM_RIGHT,2)),
                            C(tm_conf_move_right, Proj(2,0)),
                            Proj(2,0),
                        )
                     )
# conf, sym -> new_conf
tm_conf_write = C(list3,
                    C(Getter(0), Proj(2,0)), # state unchanged
                    C(Getter(1), Proj(2,0)), # left tape unchanged
                    C(cons,
                        Proj(2,1), # new symbol
                        C(tail, C(Getter(2), Proj(2,0))) # rest of original tape
                    ),
                )
# conf, state -> new_conf
tm_conf_set_state = C(list3,
                        Proj(2,1), # new state
                        C(Getter(1), Proj(2,0)), # tape unchanged
                        C(Getter(2), Proj(2,0)),
                    )
                    
# conf, action -> new_conf
# conf, [new_state, new_sym, move] -> new_conf
tm_conf_apply_action = C(tm_conf_move,
                            C(tm_conf_write,
                                C(tm_conf_set_state,
                                    Proj(2,0), # original config
                                    C(Getter(0), Proj(2,1)), # new state
                                ),
                                C(Getter(1), Proj(2,1)), # new symbol
                            ),
                            C(Getter(2), Proj(2,1))
                        )
# machine, conf -> new_conf; does step even from final config
tm_step_force = C(tm_conf_apply_action,
                    Proj(2,1), # orig_conf
                    tm_conf_action,
                )

# machine, state -> 1/0
tm_state_is_final = C(contains, C(Getter(1), Proj(2,0)), Proj(2,1))

# machine, conf -> 1/0
tm_conf_is_final = C(tm_state_is_final, Proj(2,0), C(Getter(0), Proj(2,1)))

# machine, conf -> next_conf; do not step if configuration is final
tm_step = C(cond,
                    tm_conf_is_final,
                    Proj(2,1),
                    tm_step_force,
                )

tm_steps_r = PR( Proj(2, 1), C(tm_step, Proj(4,2), Proj(4,0)) )
# machine, conf, steps -> new_conf
tm_steps = C(tm_steps_r, Proj(3,2), Proj(3,0), Proj(3,1))

### THE FUNCTIONS BELOW ARE NOT PRIMITIVE RECURSIVE BUT ONLY PARTIAL μ-RECURSIVE ###

def Minimize(predicate):
    assert predicate._is_prf
    def f(*args):
        for i in itertools.count():
            if predicate(i, *args):
                return i
    f._is_prf = True
    f._arity = predicate._arity - 1
    return f

tm_conf_steps_to_finish = Minimize(C(tm_conf_is_final, Proj(3,1), C(tm_steps, Proj(3,1), Proj(3,2), Proj(3,0))))
# machine, conf -> final_conf
tm_conf_finish = C(tm_steps, Proj(2,0), Proj(2,1), tm_conf_steps_to_finish)
# machine, input -> final_conf
tm_exec = C(tm_conf_finish, Proj(2,0), tm_init_conf)

tm_dump_conf(tm_exec(TM_PARITY, LIST(1,1,2,2,1)))

def ipy():
    """Run the IPython console in the context of the current frame.

    Useful for ad-hoc debugging."""
    frame = sys._getframe(1)
    try:
        from IPython.terminal.embed import InteractiveShellEmbed
        from IPython import embed
        import inspect
        shell = InteractiveShellEmbed.instance()
        shell(local_ns=frame.f_locals, module=inspect.getmodule(frame))
    except ImportError:
        import code
        dct={}
        dct.update(frame.f_globals)
        dct.update(frame.f_locals)
        code.interact("", None, dct)
ipy()
