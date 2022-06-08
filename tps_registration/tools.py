from math import pi

import torch


def transpose(mat):
    return torch.transpose(mat, dim0=-2, dim1=-1)


class Sigma2d(torch.autograd.Function):
    """Biharmonic function, differentiable at 0."""

    @staticmethod
    def forward(ctx, *args, **kwargs):
        x, = args
        x_2 = x ** 2
        sq_norm = torch.sum(x_2, dim=-1)
        log = torch.log(sq_norm)
        ctx.save_for_backward(x, x_2, sq_norm, log)
        return torch.where(
            torch.greater(sq_norm, 0), sq_norm * log, torch.zeros_like(sq_norm))

    @staticmethod
    def backward(ctx, *grad_outputs):
        grad_output, = grad_outputs
        x, x_2, sq_norm, log = ctx.saved_tensors
        log_dev = torch.where(
            torch.greater(sq_norm, 0), torch.log(sq_norm) / 2. + 1, torch.zeros_like(sq_norm))
        return 2 * grad_output[..., None] * x * log_dev[..., None]


class Sigma3d(torch.autograd.Function):
    """Inverse of the norm, with derivative arbitrarely set to 0 at 0."""

    @staticmethod
    def forward(ctx, *args, **kwargs):
        x, = args
        sq_norm = torch.sum(x ** 2, dim=-1)
        ctx.save_for_backward(x, sq_norm)
        return - sq_norm ** .5

    @staticmethod
    def backward(ctx, *grad_outputs):
        grad_output, = grad_outputs
        x, sq_norm = ctx.saved_tensors
        safe = torch.where(torch.greater(sq_norm, 0.), sq_norm, torch.ones_like(sq_norm))
        grad = torch.where(torch.greater(sq_norm, 0.), 1. / safe ** .5, torch.zeros_like(sq_norm))
        return - grad_output[..., None] * x * grad[..., None]


def sigma(x, dim=2):
    if dim == 2:
        return Sigma2d.apply(x)
    return Sigma3d.apply(x)


def coupling(x, dim=2, helmert=None):
    r"""Coupling matrix with entries S_ij = \sigma(x_i - x_j)."""
    if helmert is None:
        k = x.shape[-2]
        helmert = helmert_mat(k, dtype=x.dtype)
    s = sigma(x[..., :, None, :] - x[..., None, :, :], dim=dim)
    return helmert @ s @ transpose(helmert)


def helmert_mat(k, dtype=torch.float64):
    h = torch.zeros((k, k), dtype=dtype)
    for j in range(1, k):
        h[j, :j] = -(j * (j + 1)) ** -.5
        h[j, j] = j * (j * (j + 1)) ** -.5
    return h[1:]


def energy_matrices(x, dim=2, helmert=None):
    """Compute the coupling and bending energy matrices"""
    if helmert is None:
        k = x.shape[-2]
        helmert = helmert_mat(k, dtype=x.dtype)
    s_x = coupling(x, dim, helmert)
    s_inverse = torch.linalg.inv(s_x)
    x_h = helmert @ x
    x_s = transpose(x_h) @ s_inverse
    projection = x_s @ x_h
    coupling_matrix = torch.linalg.inv(projection) @ x_s
    bending_matrix = s_inverse - transpose(x_s) @ coupling_matrix
    return coupling_matrix, bending_matrix


def _get_triangle_mask(polydata):
    """Get the triangles from a vtk.polydata mesh."""
    n = polydata.GetNumberOfCells()
    triangles = []
    for i in range(n):
        cell = polydata.GetCell(i)
        p0 = cell.GetPointId(0)
        p1 = cell.GetPointId(1)
        p2 = cell.GetPointId(2)
        triangle = torch.tensor([p0, p1, p2])
        triangles.append(triangle)
    return torch.stack(triangles)


def volume(tensor, triangles=None, dim=2):
    """Compute the area or volume of a shape."""
    if dim == 2:
        intermediate = torch.concat([tensor, tensor[..., 0, :][..., None, :]], dim=-2)
        shape_stacked = torch.stack([intermediate[..., :-1, :], intermediate[..., 1:, :]], dim=-2)
        return torch.det(shape_stacked).sum(dim=-1) / 2
    shape_stacked = torch.stack(
        [tensor[triangles[:, 0]], tensor[triangles[:, 1]],
         tensor[triangles[:, 2]]], dim=1)
    return torch.det(shape_stacked).sum() / 6


def intertia_2d(contour):
    """Compute the polar inertia coefficient of a contour."""
    centre = torch.mean(contour, dim=-2)
    centred = contour - centre[..., None, :]
    intermediate = torch.concat([centred, centred[..., 0, :][..., None, :]], dim=-2)
    shape_stacked = torch.stack([intermediate[..., :-1, :], intermediate[..., 1:, :]], dim=-2)
    dets = torch.abs(torch.det(shape_stacked)[..., None])
    lengths = torch.sum(shape_stacked, dim=-2) ** 2 - torch.prod(shape_stacked, dim=-2)
    return torch.sum(dets * lengths, dim=(-2, -1)) / 12


def mu(x, dim=2, rho=1.,value=0.013):
    #return  2. * torch.ones_like(x.sum((-2, -1)))

    #return  0.013 * torch.ones_like(x.sum((-2, -1)))

    return  value * torch.ones_like(x.sum((-2, -1)))

    #if dim == 2:
    #    return 4 * pi * intertia_2d(x) / rho / volume(x, dim=dim) ** 2
    #return .4 * torch.ones_like(x.sum((-2, -1)))


def metric_matrix(x, dim=2, rho=1., helmert=None, muvalue=0.013):
    """Compute the metric matrix at a shape."""
    coupling_matrix, bending_matrix = energy_matrices(x, dim=dim, helmert=helmert)
    affine = transpose(coupling_matrix) @ coupling_matrix / 2
    bending = mu(x, dim=dim, rho=rho,value=muvalue)[..., None, None] * bending_matrix
    return affine + bending


def squared_norm(tangent_vec, base_point, dim=2, rho=1., helmert=None, muvalue=0.013):
    """Compute the metric squared norm."""
    if helmert is None:
        k = tangent_vec.shape[-2]
        helmert = helmert_mat(k, dtype=tangent_vec.dtype)
    metric_mat = metric_matrix(base_point, dim, rho, helmert, muvalue)
    tangent_vec_h = helmert @ tangent_vec
    return torch.sum(tangent_vec_h * (metric_mat @ tangent_vec_h), dim=(-2, -1))


def aspect_ratio(x):
    """Compute the aspect ratio."""
    return (x[..., :, 0].max(-1)[0] - x[..., :, 0].min(-1)[0]) / (x[..., :, 1].max(-1)[0] - x[..., :, 1].min(-1)[0])
