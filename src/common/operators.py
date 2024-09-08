import numpy as np
import scipy.sparse as sp


def identity_operator(img):
    return sp.eye(np.prod(img.shape))


def dx_operator(img, boundary="periodic"):
    flat_dims = np.prod(img.shape)
    vals = np.ones((2, flat_dims)) * np.array([[-1], [1]])

    if boundary == "valid":
        Dx = sp.spdiags(vals, [0, 1], flat_dims - 1, flat_dims)

    elif boundary == "periodic":
        Dx = sp.spdiags(vals, [0, 1], flat_dims, flat_dims).tolil()
        Dx[-1, 0] = 1

    return Dx.tocsr()


def dy_operator(img, boundary="periodic"):
    dims = img.shape
    flat_dims = np.prod(dims)
    vals = np.ones((2, flat_dims)) * np.array([[-1], [1]])

    if boundary == "valid":
        Dy = sp.spdiags(vals, [0, dims[1]], flat_dims - dims[1], flat_dims)

    elif boundary == "periodic":
        Dy = sp.spdiags(vals, [0, dims[1]], flat_dims, flat_dims).tolil()
        Dy[-dims[1] :, 0 : dims[1]] = sp.eye(dims[1])

    return Dy.tocsr()


def derivative_operator(img, boundary="periodic"):
    Dx = dx_operator(img, boundary)
    Dy = dy_operator(img, boundary)
    D = sp.vstack([Dx, Dy])

    u = np.random.random(D.shape[1])
    v = np.random.random(D.shape[0])
    assert np.isclose(((D @ u).T @ v), u @ (D.T @ v))

    return D


def laplacian_operator(img, boundary="periodic"):
    D = derivative_operator(img, boundary)

    return -D.T @ D


def perona_malik_operator(img):
    Dx = dx_operator(img, boundary="periodic")
    Dy = dy_operator(img, boundary="periodic")

    img = img.reshape([-1, 1])
    img_dx = Dx @ img
    img_dy = Dy @ img

    abs_derivative = np.sqrt(np.square(img_dx) + np.square(img_dy))
    threshold = 0.5 * np.max(abs_derivative)
    gamma = np.exp(-abs_derivative / threshold)
    gamma_diag_hat = np.sqrt(sp.diags(gamma.ravel()))

    return sp.vstack([gamma_diag_hat @ Dx, gamma_diag_hat @ Dy])
