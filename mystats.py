from mylinalg import *


def mean(array: list, axis=None):
    if isinstance(array[0], (int, float)):
        array = [array]

    means = []

    if axis is None:
        all_sum = 0
        size = 0
        for row in array:
            all_sum += sum(row)
            size += len(row)
        return all_sum / size

    elif axis == 1:
        for r in range(len(array)):
            means.append(sum(array[r]) / len(array[r]))

    elif axis == 0:

        max_row_len = 0

        for r in range(len(array)):
            row_len = len(array[r])
            max_row_len = max(max_row_len, row_len)

        for col in range(max_row_len):
            col_sum = 0
            for r in range(len(array)):
                if len(array[r]) > col:
                    col_sum += array[r][col]
            means.append(col_sum / len(array))

    return means[0] if len(means) == 1 else means


def var(array: list, axis=None):
    if isinstance(array[0], (int, float)):
        array = [array]

    var_ = []

    if axis is None:
        sqr_dev_sum = 0
        size = 0
        array_mean = mean(array)
        for row in array:
            for e in row:
                sqr_dev_sum += (e - array_mean) ** 2
            size += len(row)

        return sqr_dev_sum / size

    if axis == 1:
        for r in range(len(array)):
            row_mean = mean(array[r])
            sqr_dev_sum = 0
            for x in array[r]:
                sqr_dev_sum += (x - row_mean) ** 2
            var_.append(sqr_dev_sum / len(array[r]))

    elif axis == 0:

        max_row_len = 0

        for row in array:
            row_len = len(row)
            max_row_len = max(max_row_len, row_len)

        for col in range(max_row_len):
            col_mean = mean([array[r][col] for r in range(len(array))])
            sqr_dev_sum = 0
            for row in range(len(array)):
                if len(array[row]) > col:
                    sqr_dev_sum += (array[row][col] - col_mean) ** 2
            var_.append(sqr_dev_sum / len(array))
    return var_[0] if len(var_) == 1 else var_


def std(array, axis=None):
    var_ = var(array, axis)
    if isinstance(var_, (int, float)):
        return sqrt(var_)
    for i in range(len(var_)):
        var_[i] = sqrt(var_[i])
    return var_


def standardize(array, axis=0, with_mean=False, with_std=False):
    means = mean(array, axis) if with_mean else 0
    stds = std(array, axis) if with_std else 1

    standardized = [[0 for _ in range(len(array[0]))] for _ in range(len(array))]

    for row in range(len(array)):
        m = means[row] if axis == 1 and with_mean else 0
        s = stds[row] if axis == 1 and with_std else 1
        for col in range(len(array[0])):
            m = means[col] if axis == 0 and with_mean else m
            s = stds[col] if axis == 0 and with_std else s
            standardized[row][col] = (array[row][col] - m) / s
    return standardized


def cov(array):
    centered_array = standardize(array, axis=0, with_mean=True)
    numerator = matmul(transpose(centered_array), centered_array)
    cov_mat = scaler_div(numerator, len(array) - 1)
    return cov_mat
