import numpy as np

def clip(min_val, value, max_val):
    """
    this is trivial but annoying to stop to think about
    """
    return min(max(min_val, value), max_val)