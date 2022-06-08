import logging
import time
import numpy as np
import torch
from os.path import join
from torch.utils.tensorboard import SummaryWriter

from .tools import squared_norm, helmert_mat

def centra(matrice,n_steps):
    for i in range(n_steps):
        mat=matrice[i]
        c=[0.0,0.0]
        for p in range(0,mat.size()[0]):
            c[0]+=mat[p][0]
            c[1]+=mat[p][1]

        c[0]/=mat.size()[0]
        c[1]/=mat.size()[1]

        for p in range(0,mat.size()[0]):
            #centro la figura
            matrice[i][p][0]-=c[0]
            matrice[i][p][1]-=c[1]

        #Ricalcolo le decentrature
        c=[0.0,0.0]
        for p in range(0,mat.size()[0]):
            c[0]+=mat[p][0]
            c[1]+=mat[p][1]

        c[0]/=mat.size()[0]
        c[1]/=mat.size()[1]

        print("------------------------")
        print("Print del centroide:",c)
    return matrice

def loss_function(source, target, path, kappa_1=1., kappa_2=1., dim=2, rho=1., helmert=None, mu=0.013):
    distances = squared_norm(path[1:] - path[:-1], path[:-1], dim, rho, helmert, mu)
    distances = torch.Tensor.sqrt(distances)
    equality_constraint = kappa_1 * (
            torch.sum((target - path[-1]) ** 2) + torch.sum((source - path[0]) ** 2))
    equirepartition_constraint = kappa_2 * torch.sum((distances[1:] ** .5 - distances[:-1] ** .5) ** 2)
    return torch.sum(distances) + equality_constraint + equirepartition_constraint
    


def initialisation(source, target, n):
    """Linear interpolation."""
    times = torch.linspace(0, 1, n)
    output=times[:, None, None] * target[None, :, :] + (1. - times)[:, None, None] * source[None, :, :]
    return centra(output,n)


def registration(
        source,
        target,
        n,
        log_dir,
        dim=2,
        rho=1.,
        lr=1e-3,
        max_iter=100,
        eps=1e-6, kappa_1=1., kappa_2=1., muvalue=0.013):

    k = source.shape[-2]
    helmert = helmert_mat(k, source.dtype)

    initial = initialisation(source, target, n)
    params = initial.clone().requires_grad_(True)
    kappa_1_param = torch.tensor(kappa_1).requires_grad_(True)
    kappa_2_param = torch.tensor(kappa_2).requires_grad_(True)

    optimizer = torch.optim.Adam([params, kappa_1_param, kappa_2_param], lr=lr)

    log_wall_time = time.time()
    tb = SummaryWriter(log_dir + f'/runs/{log_wall_time}')

    for iter_count in range(max_iter):
        loss = loss_function(source, target, params, kappa_1_param, kappa_2_param, dim, rho, helmert, muvalue)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        tb.add_scalar("Loss", loss.detach(), iter_count)

        grad_norm = torch.sum(params.grad ** 2)
        tb.add_scalar("Grad Norm", grad_norm, iter_count)
        tb.add_scalar('kappa', kappa_1_param.detach(), iter_count)
        tb.add_scalar('kappa_2', kappa_2_param.detach(), iter_count)

        if torch.isnan(grad_norm).any():
            logging.warning('NaN detected in the gradient.')

        if grad_norm.detach().numpy() < eps:
            logging.info(f'Tolerance threshold reached with grad norm: {grad_norm}')
            break

    tb.close()
    logging.warning(f'run time: {time.time() - log_wall_time}')
    return params.detach()
