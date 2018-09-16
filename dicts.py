from prf import *
from logic import *
from pairs import *
from lists import *

### DICTIONARIES (as lists of (key,value) pairs) ###

filter_key = Filter(C(eq, C(left, Proj(2, 0)), Proj(2,1)), 1)
remove_key = Filter(C(ne, C(left, Proj(2, 0)), Proj(2,1)), 1)
has_key = C(sgn, filter_key)
lookup = C(right, C(head, filter_key))
dict_set = C(cons, C(pair, Proj(3,1), Proj(3,2)), C(remove_key, Proj(3,0), Proj(3,1)))
