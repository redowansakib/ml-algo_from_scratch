from modules.mylinalg import *
from modules.mystats import *
from modules.myprocessing import *
from modules.mymetrics import r2_score
import random


def nnls(X, y, L2_penalty=0, gram=None):
    X, y = validate(X, y)

    if shape(y)[1] != 1:
        raise Exception('please provide only one target variable')

    max_iter = 10000
    tol = 0.0001
    warm_start = False
    random.seed(10)
    b = None
    selection = 'cyclic'
    b = [0 for i in range(len(X[0]))]

    gram = matmul(transpose(X), X) if not gram else gram

    loop_max_step = 100

    while loop_max_step > tol:
        max_step = 0

        idx_set = set(range(len(b)))

        j = 0
        iter = 0
        if iter > max_iter:
            break

        while idx_set:
            iter += 1
            if selection == 'random':
                j = random.choice(idx_set)
            idx_set.discard(j)

            b_j = b[j]
            X_j = [x[j] for x in X]
            XT_jY = dot(X_j, y)

            b_k = []
            XT_jX_k = []
            for k in range(len(b)):
                if k != j:
                    b_k.append(b[k])
                    XT_jX_k.append(gram[j][k])

            XT_jX_kb_k = dot(XT_jX_k, b_k)
            rho_j = XT_jY - XT_jX_kb_k

            z_j = gram[j][j]

            if j == 0 or rho_j > 0:
                b_j_hat = rho_j / (L2_penalty + z_j)
            else:
                b_j_hat = 0

            step = abs(b_j_hat - b_j)
            max_step = max(step, max_step)
            b[j] = b_j_hat

            if selection == 'cyclic':
                j += 1
        loop_max_step = max_step
    return b


class MyLinearRegression:
    def __init__(self, fit_intercept=True, copy_X=True, positive=False):
        self._set_intercept = fit_intercept
        self._positive = positive
        self._copy_X = copy_X

    def fit(self, X, y):

        try:
            self.feature_names_in_ = X.columns
        except AttributeError:
            self.feature_names_in_ = None

        X, y = validate(X, y)

        if self._copy_X:
            X = X.copy()

        if len(X) != len(y):
            raise Exception('length of X and y must be same')

        self.rank_ = matrix_rank(X)
        self.singular_ = svdvals(X)
        self.n_features_in_ = len(X[0])

        if self._set_intercept:
            X = [[1] + x for x in X]

        if self._positive:
            self._b = []
            for i in range(len(y[0])):
                yi = [[t[i]] for t in y]
                b = nnls(X, yi)
                self._b.append(b)
                self._b = transpose(self._b)
        else:
            self._b = matmul(matmul(cholesky_inverse(matmul(transpose(X), X)), transpose(X)), y)
        self.coef_ = transpose(self._b[1:])
        self.intercept_ = self._b[0]

        return self

    def predict(self, X):

        X, = validate(X)
        X = to2dim(X)
        X = [[1] + x for x in X]

        y_hat = matmul(X, self._b)

        return y_hat

    def score(self, X, y):
        y_pred = self.predict(X)
        return r2_score(y, y_pred)
