from modules.mystats import *
from modules.mylinalg import *

def tolist(arr):

    if isinstance(arr,list):
        return arr
    else:
        try:
            return arr.values.tolist()
        except AttributeError:
            try:
                return arr.tolist()
            except AttributeError:
                raise Exception('Unrecognized data type')


def validate(*args):

    args = [tolist(arr) for arr in args]

    for arr in args:
        if not is_matrix(arr):
            print(arr)
            raise Exception('Data must be a matrix of 2 dimension')

    return tuple(args)