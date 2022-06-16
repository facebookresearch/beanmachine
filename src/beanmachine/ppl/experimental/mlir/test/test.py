import inspect
import math
import typing
import unittest

from to_metal import to_metal

x = math.pow

@to_metal
def foo(p1:float) -> float:
    i0 = p1*p1
    return i0


if __name__ == "__main__":
    call = foo(4)
    print("mlir returned " + str(call))
