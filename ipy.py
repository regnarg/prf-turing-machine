#!/usr/bin/python3

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
if __name__ == '__main__':
    from prf import *
    from arith import *
    from logic import *
    from division import *
    from pairs import *
    from lists import *
    from dicts import *
    from tm import *
    ipy()
