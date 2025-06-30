import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.nn import Module
from models import KAN

def ql_loss(y_pred, y_true):
    taus = np.array([0.5, 0.025, 0.05, 0.075, 0.925, 0.95, 0.975])
    y_pred = y_pred.flatten(1)
    e = torch.mean(
        (1.0 - taus[0]) * torch.relu(torch.flatten(y_pred[:, 0]) - torch.flatten(y_true)) + taus[0] * torch.relu(
            torch.flatten(y_true) - torch.flatten(y_pred[:, 0])), dim=-1)
    for i, tau in enumerate(taus[1:]):
        e += torch.mean(
            (1.0 - tau) * torch.relu(torch.flatten(y_pred[:, i + 1]) - torch.flatten(y_true)) + tau * torch.relu(
                torch.flatten(y_true) - torch.flatten(y_pred[:, i + 1])), dim=-1)
    return e

def lpls_loss(y_predict, y_ture):
    u = y_predict[:, :, 0]
    sigma = y_predict[:, :, 1]
    sigma_min = torch.full_like(sigma, 1e-6)
    sigma = torch.maximum(sigma, sigma_min)
    loss = torch.sum(torch.log(2 * sigma) + torch.abs((y_ture - u) / (sigma)), dim=-1)
    loss = torch.sum(loss)
    if torch.isnan(loss):
        print('nan_train')
    return loss

def gauss_loss(y_predict, y_true):
    # Gaussian likelihood
    u = y_predict[:, :, 0]
    sigma = y_predict[:, :, 1]
    sigma_min = torch.full_like(sigma, 1e-6)
    sigma = torch.maximum(sigma, sigma_min)
    taus = np.array([0.5])
    e = torch.sum(torch.log(sigma) + taus[0] * torch.square((y_true - u) / sigma))
    if torch.isnan(e):
        print('nan_train')
    return e

def student_t_loss(y_true, y_pred, nu, sigma):
    term1 = torch.lgamma((nu + 1) / 2) - torch.lgamma(nu / 2)
    term2 = 0.5 * torch.log(nu * torch.pi) + torch.log(sigma)
    term3 = (nu + 1) / 2 * torch.log(1 + ((y_true - y_pred) ** 2) / (nu * sigma ** 2))
    loss = term1 + term2 + term3
    return loss.mean()


class Dense_Gauss(nn.Module):
    def __init__(self, n_input, n_out_tasks=1):
        super(Dense_Gauss, self).__init__()
        self.n_in = n_input
        self.n_out = 2 * n_out_tasks
        self.n_tasks = n_out_tasks
        # self.l1 = nn.Linear(self.n_in, self.n_out)
        self.l1=KAN([self.n_in, self.n_out])

    def forward(self, x):
        x = self.l1(x)
        if len(x.shape) == 1:
            gamma, lognu = torch.split(x, self.n_tasks, dim=0)
        else:
            gamma, lognu = torch.split(x, self.n_tasks, dim=-1)

        nu = F.softplus(lognu)
              # + 1e-6)
        y=torch.stack([gamma, nu], dim=2).to(x.device)
        y=y.squeeze(1)
        y=y.transpose(2,1)
        return y


def reduce(val, reduction):
    if reduction == 'mean':
        val = val.mean()
    elif reduction == 'sum':
        val = val.sum()
    elif reduction == 'none':
        pass
    else:
        raise ValueError(f"Invalid reduction argument: {reduction}")
    return val

def NIG_NLL(y: torch.Tensor,
            gamma: torch.Tensor,
            nu: torch.Tensor,
            alpha: torch.Tensor,
            beta: torch.Tensor, reduction='mean'):
    inter = 2 * beta * (1 + nu)

    nll = 0.5 * (np.pi / nu).log() \
          - alpha * inter.log() \
          + (alpha + 0.5) * (nu * (y - gamma) ** 2 + inter).log() \
          + torch.lgamma(alpha) \
          - torch.lgamma(alpha + 0.5)
    return torch.mean(nll)


# def NIG_NLL(y: torch.Tensor,
#             gamma: torch.Tensor,
#             nu: torch.Tensor,
#             alpha: torch.Tensor,
#             beta: torch.Tensor, reduction='mean'):
#     # student_var = beta * (1. + alpha) / (nu * alpha)
#     student_var = torch.sqrt(beta * (1 + nu) / (nu * alpha))
#     dist = StudentT(loc=gamma, scale=student_var, df=2 * alpha)
#     nll = -1 * dist.log_prob(y)
#     return reduce(nll, reduction=reduction)

def NIG_Reg(y, gamma, nu, alpha, reduction='mean'):
    error = (y - gamma).abs()
    # evidence = 2 * nu + alpha
    evidence = nu + 2 * alpha
    return reduce(error * evidence, reduction='mean')

# def EvidentialRegression(y: torch.Tensor, evidential_output: torch.Tensor, lmbda=1.):
#     gamma, nu, alpha, beta = evidential_output
#     loss_nll = NIG_NLL(y, gamma, nu, alpha, beta)
#     loss_reg = NIG_Reg(y, gamma, nu, alpha)
#     return loss_nll, lmbda * loss_reg

def DER_Loss(y: torch.Tensor, evidential_output: torch.Tensor, lmbda=0.01):
    # gamma, nu, alpha, beta = evidential_output[:, :, 0], evidential_output[:, :, 1], evidential_output[:, :, 2], evidential_output[:, :, 3]
    gamma, nu, alpha, beta = torch.split(evidential_output,1,dim=-1)
    loss_nll = NIG_NLL(y, gamma, nu, alpha, beta)
    loss_reg = NIG_Reg(y, gamma, nu, alpha)
    #
    return loss_nll + lmbda * loss_reg


def SDER_REG(y, gamma, nu, alpha, beta, reduction='mean'):
    error = (y - gamma) ** 2 * (alpha * nu) / (beta * (1 + nu))
    # evidence = 2 * nu + alpha
    evidence = nu + 2 * alpha
    return torch.mean(error * evidence)


def SDER_Loss(y_pred: torch.Tensor,y: torch.Tensor, coeff=0.005):
    gamma, nu, alpha, beta = y_pred[:, :, 0], y_pred[:, :, 1], y_pred[:, :, 2], y_pred[:, :, 3]
    # gamma, nu, alpha, beta = torch.split(evidential_output,1,dim=-1)
    loss_nll = NIG_NLL(y, gamma, nu, alpha, beta)
    loss_reg = SDER_REG(y, gamma, nu, alpha, beta)

    return loss_nll + coeff * loss_reg

class DenseNormalGamma(Module):
    def __init__(self, n_input, n_out_tasks=1):
        super(DenseNormalGamma, self).__init__()
        self.n_in = n_input
        self.n_out = 4 * n_out_tasks
        self.n_tasks = n_out_tasks
        self.l1 = nn.Linear(self.n_in, self.n_out)

    def forward(self, x):
        x = self.l1(x)
        if len(x.shape) == 1:
            gamma, lognu, logalpha, logbeta = torch.split(x, self.n_tasks, dim=0)
        else:
            gamma, lognu, logalpha, logbeta = torch.split(x, self.n_tasks, dim=1)

        nu = F.softplus(lognu)
        alpha = F.softplus(logalpha) + 1
        beta = F.softplus(logbeta)
        y=torch.stack([gamma, nu, alpha, beta], dim=2).to(x.device)

        # y=y.unsqueeze(1)
        return y


class SDERLayer(nn.Module):
    def __init__(self, n_input, n_out_tasks=1):
        super(SDERLayer, self).__init__()
        self.n_in = n_input
        self.n_out = 4 * n_out_tasks
        self.n_tasks = n_out_tasks
        # self.l1 = nn.Linear(self.n_in, self.n_out)
        self.l1=KAN([self.n_in, self.n_out])

    def forward(self, x):
        x = self.l1(x)
        if len(x.shape) == 1:
            gamma, lognu, logalpha, logbeta = torch.split(x, self.n_tasks, dim=0)
        else:
            gamma, lognu, logalpha, logbeta = torch.split(x, self.n_tasks, dim=-1)
        gamma = gamma
        nu = nn.functional.softplus(lognu)
        alpha = nn.functional.softplus(logalpha) + 1.0
        beta = nn.functional.softplus(logbeta)
        y = torch.stack([gamma, nu, alpha, beta], dim=2).to(x.device)
        y=y.squeeze(1)
        y=y.transpose(2,1)
        return y