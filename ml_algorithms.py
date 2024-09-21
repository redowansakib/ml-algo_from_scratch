from modules.mystats import *


def myPCA(data, n_components=1):

    cov_mat = cov(data)
    U, S, Vh = svd(cov_mat)
    components = [U[i][:n_components] for i in range(len(U))]
    reduced = matmul(data, components)

    return reduced


def my_simple_linear_regression(x: [list], y: [list]) -> tuple[float, float]:
    """This is a simple version of linear regression for only one variable and one output"""

    if len(x) != len(y):
        raise Exception("x and y both should have same row number")

    if len(x[0]) > 1 or len(y[0]) > 1:
        raise Exception("x and y both should have 1 dimension")

    cov_mat = cov(x, y)
    coef = cov_mat[0][1] / cov_mat[0][0]
    intercept = mean(y) - coef * mean(x)

    return coef, intercept









