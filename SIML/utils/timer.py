"""Timer for function running"""
from functools import wraps
from time import time


def timer(func):
    """A wrapper to calculate the cpu time required by function excution

    Args:
        func (function): the name of function.

    """

    @wraps(func)
    def func_wrapper(*args, **kwargs):
        time_start = time()
        result = func(*args, **kwargs)
        time_end = time()
        time_spend = time_end - time_start
        print(f"{func.__name__} cost time {time_spend} s")
        return result

    return func_wrapper
