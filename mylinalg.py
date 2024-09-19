import random
from math import sqrt
from random import normalvariate


def is_empty(arr: list) -> bool:
    while isinstance(arr, list):
        if not arr:
            return True
        arr = arr[0]
    return False


def empty():
    raise Exception("Matrix is empty")


def is_matrix(arr: list) -> bool:
    if isinstance(arr, list):
        if isinstance(arr[0], list):
            col_len = len(arr[0])
            for r in range(len(arr)):
                if len(arr[r]) != col_len:
                    return False
            return True
    return False


def not_matrix():
    raise Exception('Input must be matrix(constant row length)')


def to2dim(arr: list) -> list:
    if is_empty(arr):
        return [[]]

    if isinstance(arr, (int, float)):
        return [[arr]]

    if isinstance(arr[0], (int, float)):
        col_mat = []
        for i in range(len(arr)):
            col_mat.append([arr[i]])
        return col_mat

    return arr


def shape(inp: [int, float, list]) -> [tuple[int, int], tuple[int,], None]:
    if is_empty(inp):
        return 0, 0

    if isinstance(inp, (int, float)):
        return None

    if isinstance(inp[0], (int, float)):
        return len(inp),

    return len(inp), len(inp[0])


def dim(inp: [int, float, list]) -> int:
    if not shape(inp):
        return 0
    return len(shape(inp))


def res_type(inp1: [int, float, list], inp2: [int, float, list]) -> str:
    inp1_dim = dim(inp1)
    inp2_dim = dim(inp2)

    if 1 in {inp1_dim, inp2_dim}:
        return 'flat'
    if max(inp1_dim, inp2_dim) == 2:
        return 'mat'
    return 'num'


def numify(inp):
    if isinstance(inp, (int, float)):
        return inp
    if isinstance(inp[0], (int, float)):
        return inp[0]
    return inp[0][0]


def flatify(inp):
    if isinstance(inp, (int, float)):
        return [inp]
    if isinstance(inp[0], (int, float)):
        return inp
    return [inp[r][c] for c in range(len(inp[0])) for r in range(len(inp))]


def matify(inp):
    if isinstance(inp, (int, float)):
        return [[inp]]
    if isinstance(inp[0], (int, float)):
        return [inp]
    return inp


def result(inp, res_type):
    res_dic = {
        'num': numify,
        'flat': flatify,
        'mat': matify
    }

    return res_dic[res_type](inp)


def transpose(mat: list) -> list:
    if is_empty(mat):
        empty()

    mat = to2dim(mat)

    transposed = []
    row, col = shape(mat)

    for col in range(col):
        transposed.append([mat[row][col] for row in range(len(mat))])

    return transposed


def scaler_mul(inp1: [int, float, list], inp2: [int, float, list]) -> [float, list]:
    scaler = inp1 if isinstance(inp1, (int, float)) else inp2
    mat = inp2 if scaler is inp1 else inp1

    if not isinstance(scaler, (int, float)):
        raise Exception("one of the input must be a scaler")

    if isinstance(mat, (int, float)):
        return scaler * mat

    if isinstance(mat[0], (int, float)):
        return [i * scaler for i in mat]

    for r in range(len(mat)):
        for c in range(len(mat[r])):
            mat[r][c] = scaler * mat[r][c]
    return mat


def scaler_add(inp1: [int, float, list], inp2: [int, float, list]) -> [float, list]:
    scaler = inp1 if isinstance(inp1, (int, float)) else inp2
    mat = inp2 if scaler is inp1 else inp1

    if not isinstance(scaler, (int, float)):
        raise Exception("one of the input must be a scaler")

    if isinstance(mat, (int, float)):
        return scaler + mat

    if isinstance(mat[0], (int, float)):
        return [i + scaler for i in mat]

    for r in range(len(mat)):
        for c in range(len(mat[r])):
            mat[r][c] = scaler + mat[r][c]
    return mat


def scaler_sub(inp1: [int, float, list], inp2: [int, float, list]) -> [float, list]:
    scaler = inp1 if isinstance(inp1, (int, float)) else inp2
    mat = inp2 if scaler is inp1 else inp1

    if not isinstance(scaler, (int, float)):
        raise Exception("one of the input must be a scaler")

    coef = 1 if isinstance(inp1, (int, float)) else -1

    if isinstance(mat, (int, float)):
        return scaler - mat

    if isinstance(mat[0], (int, float)):
        return [coef * (scaler - i) for i in mat]

    for r in range(len(mat)):
        for c in range(len(mat[r])):
            mat[r][c] = coef * (scaler - mat[r][c])
    return mat


def scaler_div(inp1: [int, float, list], inp2: [int, float, list]) -> [float, list]:
    scaler = inp1 if isinstance(inp1, (int, float)) else inp2
    mat = inp2 if scaler is inp1 else inp1

    if not isinstance(scaler, (int, float)):
        raise Exception("one of the input must be a scaler")

    if isinstance(mat, (int, float)):
        return scaler / mat

    if isinstance(mat[0], (int, float)):
        if isinstance(inp1, (int, float)):
            return [scaler / i for i in mat]
        else:
            return [i / scaler for i in mat]

    for r in range(len(mat)):
        for c in range(len(mat[r])):
            if isinstance(inp1, (int, float)):
                mat[r][c] = scaler / mat[r][c]
            else:
                mat[r][c] = mat[r][c] / scaler

    return mat


def scaler_op(inp1: [int, float, list], inp2: [int, float, list], op: str) -> [float, list]:
    if op not in {'+', '-', '*', '/'}:
        raise Exception("Operations must be between '+, - , *, /")

    dic = {
        "+": scaler_add,
        "-": scaler_sub,
        "*": scaler_mul,
        "/": scaler_div
    }

    return dic[op](inp1, inp2)


def dot(arr1: [int, float, list], arr2: [int, float, list]) -> [float, int]:
    if isinstance(arr1, (int, float)) or isinstance(arr2, (int, float)):
        scaler_mul(arr1, arr2)

    arr1, arr2 = to2dim(arr1), to2dim(arr2)

    if len(arr1[0]) > 1:
        raise Exception("Inputs must be column matrix")

    if len(arr1) != len(arr2):
        raise Exception("inputs should have similar length")

    return sum(arr1[i][0] * arr2[i][0] for i in range(len(arr1)))


def outer_prod(inp1: [int, float, list], inp2: [int, float, list]) -> list:
    inp1 = to2dim(inp1)
    inp2 = to2dim(inp2)

    if len(inp1[0]) > 1 or len(inp2[0]) > 1:
        raise Exception("Inputs must be column vector")

    res = [[0 for _ in range(len(inp2))] for _ in range(len(inp1))]

    for row in range(len(inp1)):
        for col in range(len(inp2)):
            res[row][col] = inp1[row][0] * inp2[col][0]

    return res


# Linked to: "PCA_from_scratch"
def matmul(mat1: [int, float, list], mat2: [int, float, list]) -> list:
    if is_empty(mat1) or is_empty(mat2):
        empty()

    mat1_sh = shape(mat1)
    mat2_sh = shape(mat2)

    if not mat1_sh or not mat2_sh:
        raise Exception('Better use scaler operation')

    if len(mat1_sh) == len(mat2_sh) == 1 and mat1_sh[0] == mat2_sh[0]:
        return dot(mat1, mat2)

    res_form = res_type(mat1, mat2)

    mat1 = to2dim(mat1)
    mat2 = to2dim(mat2)

    if len(mat1[0]) != len(mat2):
        raise Exception('column of 1st matrix should be same as row of 2nd matrix')

    mul = []
    for row in range(len(mat1)):
        mul_row = []
        for col in range(len(mat2[0])):
            each_multi = 0
            for i in range(len(mat1[row])):
                each_multi += mat1[row][i] * mat2[i][col]
            mul_row.append(each_multi)
        mul.append(mul_row)

    return result(mul, res_form)


def matsub(mat1: [int, float, list], mat2: [int, float, list]) -> list:
    if isinstance(mat1, (int, float)) or isinstance(mat2, (int, float)):
        return scaler_sub(mat1, mat2)

    result_form = res_type(mat1, mat2)

    mat1 = to2dim(mat1)
    mat2 = to2dim(mat2)

    if shape(mat1) != shape(mat2):
        raise Exception("both matrix should have same length")

    res = [[0 for _ in range(len(mat1[0]))] for _ in range(len(mat1))]
    for r in range(len(res)):
        for c in range(len(res[0])):
            res[r][c] = mat1[r][c] - mat2[r][c]
    return result(res, result_form)


def matdiv(mat1: [int, float, list], mat2: [int, float, list]) -> list:
    if isinstance(mat1, (int, float)) or isinstance(mat2, (int, float)):
        return scaler_div(mat1, mat2)

    result_form = res_type(mat1, mat2)

    mat1 = to2dim(mat1)
    mat2 = to2dim(mat2)

    if shape(mat1) != shape(mat2):
        raise Exception("both matrix should have same length")

    res = [[0 for _ in range(len(mat1[0]))] for _ in range(len(mat1))]
    for r in range(len(res)):
        for c in range(len(res[0])):
            res[r][c] = mat1[r][c] / mat2[r][c]
    return result(res, result_form)


def l2_norm(vec):
    vec = to2dim(vec)
    if len(vec[0]) > 1:
        raise Exception("Input should be a vector of one dimension")
    return sqrt(sum([i[0] ** 2 for i in vec]))


def random_unit_vector(size):
    random_vec = [random.random() for _ in range(size)]
    unit_vec = l2_norm(random_vec)
    return [i / unit_vec for i in random_vec]


def power_iteration(X, epsilon=1e-10, it=10000):
    m, n = shape(X)
    init_vec = random_unit_vector(n)
    curr_vec = init_vec
    XTX = matmul(transpose(X), X)
    i = 0
    while i <= it:
        new_vec = matmul(XTX, curr_vec)
        prev_vec = curr_vec
        curr_vec = new_vec
        curr_vec = matdiv(curr_vec, l2_norm(curr_vec))

        while abs(dot(prev_vec, curr_vec)) > 1 - epsilon:
            return curr_vec
    raise Exception("Failed to converge for eigenvector")


# Linked to: "PCA_from_scratch"
def svd(X: list, epsilon=1e-10, it=10000) -> tuple[list, list, list]:
    m, n = shape(X)
    basis_change = []

    for i in range(n):
        v = power_iteration(X, epsilon, it)
        us = matmul(X, v)
        s = l2_norm(us)
        u = matdiv(us, s)
        X = matsub(X, scaler_mul(s, outer_prod(u, v)))
        basis_change.append((u, s, v))

    basis_change.sort(key=lambda x:x[1], reverse=True)

    UT, S, Vh = [list(x) for x in zip(*basis_change)]

    return transpose(UT), S, Vh


def eig(mat: list) -> tuple[list, list]:
    U, S, V = svd(mat)
    return S, U
