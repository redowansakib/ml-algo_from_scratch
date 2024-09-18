from mystats import *

def myPCA(data, n_components=1):
    cov_mat = cov(data)
    U, S, Vh = svd(cov_mat)
    components = [U[i][:n_components] for i in range(len(U))]
    reduced = matmul(data, components)
    return reduced











