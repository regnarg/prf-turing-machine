#!/usr/bin/python3

from prf import *
from pairs import *
from lists import *
from dicts import *

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

tm_conf_steps_to_finish = Minimize(C(not_, C(tm_conf_is_final, Proj(3,1), C(tm_steps, Proj(3,1), Proj(3,2), Proj(3,0)))))
# machine, conf -> final_conf
tm_conf_finish = C(tm_steps, Proj(2,0), Proj(2,1), tm_conf_steps_to_finish)
# machine, input -> final_conf
tm_exec = C(tm_conf_finish, Proj(2,0), tm_init_conf)

# EXAMPLE
if __name__ == '__main__':
    inp = LIST(1,1,2,2,1)
    # Step by step execution using primitive recursive step function
    conf = tm_init_conf(TM_PARITY, inp)
    tm_dump_conf(conf)
    while not tm_conf_is_final(TM_PARITY, conf):
        conf = tm_step(TM_PARITY, conf)
        tm_dump_conf(conf)
    # Execution using one partial recursive function call
    tm_dump_conf(tm_exec(TM_PARITY, inp))

