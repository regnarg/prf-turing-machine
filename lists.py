from prf import *
from arith import *
from logic import *
from pairs import *

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
