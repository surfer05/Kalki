# utils.py

"""
Utility functions for the Key Lattice Group Messaging Protocol Simulation.
"""

import hashlib

def KeyRoll(k, x):
    """
    Derive a new key from the current key and random data x.
    """
    combined = b''.join(sorted([k, x]))
    new_key = hashlib.sha256(combined).digest()
    return new_key

def increment(index, dimension):
    """
    Increment the index tuple in the specified dimension.
    """
    new_index = list(index)
    new_index[dimension] += 1
    return tuple(new_index)
