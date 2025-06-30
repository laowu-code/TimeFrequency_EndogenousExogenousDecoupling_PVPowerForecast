import os

import pandas as pd
import torch
from matplotlib import pyplot as plt
import dill
from tqdm import tqdm
import numpy as np
import math
import properscoring as ps
import scipy.stats
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from math import sqrt


def metrics_point(y_pred, y_true):
    """
    Calculate MAE, R2, RMSE, and MMAPE for each prediction step.

    Args:
        y_pred (np.ndarray): Predicted values, shape (N, P, 7).
            The first column is the point prediction.
        y_true (np.ndarray): True values, shape (N, P).

    Returns:
        dict: A dictionary containing metrics for each step:
            - 'MAE': Mean Absolute Error, shape (P,).
            - 'R2': R-squared, shape (P,).
            - 'RMSE': Root Mean Squared Error, shape (P,).
            - 'MMAPE': Mean Maximum Absolute Percentage Error, shape (P,).
    """
    # Extract the point predictions from y_pred (first column)
    point_preds = y_pred[:, :, 0]  # Shape (N, P)

    # Number of prediction steps
    N, P = y_true.shape

    # Initialize metrics
    mae = np.zeros(P)
    r2 = np.zeros(P)
    rmse = np.zeros(P)
    # mmape = np.zeros(P)
    mbe=np.zeros(P)

    # Loop through each step
    for p in range(P):
        y_pred_p = point_preds[:, p]  # Predicted values for step p
        y_true_p = y_true[:, p]  # True values for step p

        # MAE
        mae[p] = np.mean(np.abs(y_pred_p - y_true_p))

        # R2
        ss_total = np.sum((y_true_p - np.mean(y_true_p)) ** 2)
        ss_residual = np.sum((y_true_p - y_pred_p) ** 2)
        r2[p] = 1 - ss_residual / ss_total if ss_total != 0 else 0

        # RMSE
        rmse[p] = np.sqrt(np.mean((y_pred_p - y_true_p) ** 2))
        mbe[p] = np.mean(y_pred_p - y_true_p)

        # MMAPE
        # mmape[p] = np.mean(np.abs(y_pred_p - y_true_p) / np.maximum(np.abs(y_true_p), 1e-8))


    # Compile results into a dictionary
    metrics = {
        'MAE': np.round(mae, 4),
        'RMSE': np.round(rmse, 4),
        'R2': np.round(r2, 4),
        'MBE': np.round(mbe, 4),
    }

    return metrics


class prob_metric(object):

    def get_metrics(self, quantile_low, quantile_high, lower, upper, label, isList=False):
        # quantile_low下边界对应的分位数；quantile_high为上边界对应的置信度；lower为下边界；upper为上边界；label:标签
        # label_data = label.flatten()
        # lower = lower.flatten()
        # upper = upper.flatten()
        label_data = label
        lower = lower
        upper = upper
        u = quantile_high - quantile_low
        PL = self.pinball_loss(label_data, upper, lower, quantile_low, quantile_high)
        # WQL = self.wQL(label_data, upper, lower, quantile_low, quantile_high)
        PICP = self.cp(label_data, upper, lower)  # PICP越接近置信度越好（0，1）
        # MWP = self.mwp(label_data, upper, lower)
        # MC = self.mc(MWP, CP)
        PINAW = self.pinaw(label_data, upper, lower)
        # PINRW = self.pinrw(label_data, upper, lower)
        CWC = self.cwc(PICP, PINAW, 50, u)
        IS, IS_list = self.skill_score(label_data, upper, lower, 1 - u)
        ACE = np.round((PICP - u), 4)

        return PICP, ACE, PINAW, CWC, IS, PL

    def pinball_loss(self, label, up, low, quantile_low, quantile_high):

        PL = (np.sum(
            (label - low) * (quantile_low * (low <= label) - (1 - quantile_low) * (low > label)), axis=0) + np.sum(
            (label - up) * (quantile_high * (up <= label) - (1 - quantile_high) * (up > label)), axis=0)) / (
                     2 * label.shape[0])

        return np.round(PL, 4)

    def wQL(self, label, up, low, quantile_low, quantile_high):
        PL = (np.sum(
            (label - low) * (quantile_low * (low <= label) - (1 - quantile_low) * (low > label)), axis=0) + np.sum(
            (label - up) * (quantile_high * (up <= label) - (1 - quantile_high) * (up > label)), axis=0))
        wql = 2 * PL / np.sum(label)
        return np.round(wql, 4)

    def cp(self, label, up, low):  # PICP‘s algorithm is the same
        l, n = label.shape
        picp = np.zeros((n))
        for i in range(n):
            result1 = (label[:, i] <= up[:, i]).astype(int)
            result2 = (label[:, i] >= low[:, i]).astype(int)
            result = result1 + result2
            picp[i] = np.round(np.sum(result == 2, axis=0) / l, 4)
        return picp

    def mwp(self, label, up, low):
        mwp = np.mean(np.abs(up - low) / label)
        return round(mwp, 4)

    def mc(self, mwp, cp):
        mc = mwp / cp
        return np.round(mc, 4)

    def pinaw(self, label, up, low):
        l, n = label.shape
        pinaw = np.zeros((n))
        for i in range(n):
            pinaw[i] = np.round(
                np.mean(np.abs(up[:, i] - low[:, i]), axis=0) / (np.max(label[:, i]) - np.min(label[:, i])), 4)
        return pinaw

    def pinrw(self, label, up, low):
        PINRW = np.sqrt(np.mean(np.square(up - low))) / (np.max(label) - np.min(label))
        return np.round(PINRW, 4)

    # def cwc(self, index_picp, index_pinaw, n, u):
    #     e = math.exp(-n * (index_picp - [u]*12))
    #     if index_picp >= u:
    #         r = 0
    #     else:
    #         r = 1
    #     index = index_pinaw + r * e
    #     return np.round(index, 4)

    def cwc(self, index_picp, index_pinaw, n, u):
        l = (-n * (index_picp - u))
        e = np.zeros(l.shape[0])
        r = np.zeros(l.shape[0])
        for i in range(l.shape[0]):
            e[i] = math.exp(l[i])
            if index_picp[i] >= u:
                r[i] = 0
            else:
                r[i] = 1
        index = index_pinaw * (1 + r * e)
        return np.round(index, 4)

    def skill_score(self, label, up, low, alpha):
        """
        Calculate skill score for inputs with shape (N, L).

        Args:
            label (np.ndarray): True values, shape (N, L).
            up (np.ndarray): Upper bounds, shape (N, L).
            low (np.ndarray): Lower bounds, shape (N, L).
            alpha (float): Weight parameter.

        Returns:
            tuple:
                - Average skill score for each column (L-dimensional array).
                - Skill scores for all elements (N, L array).
        """
        # Calculate coverage (cc)
        cc = up - low
        # Case 1: label > up
        case1_mask = label > up
        sc1 = np.where(case1_mask, -2 * alpha * cc - 4 * (label - up), 0)
        # Case 2: low <= label <= up
        case2_mask = (label >= low) & (label <= up)
        sc2 = np.where(case2_mask, -2 * alpha * cc, 0)
        # Case 3: label < low
        case3_mask = label < low
        sc3 = np.where(case3_mask, -2 * alpha * cc - 4 * (low - label), 0)

        # Combine all cases
        sc = sc1 + sc2 + sc3
        # Average skill score for each column
        avg_skill_score = np.round(np.mean(sc, axis=0), 4)
        # Return results
        return avg_skill_score, np.round(sc, 4)

    # pred_result:[样本个数，每个样本的各个分位点的条件分位数]；quantiles：一个array向量，存储的是各个分为点的值
    def ProbabilityPredictionMetricCalculation(self, pred_result, quantiles, label):
        label_data = label.flatten()
        crps = self.CRPS(pred_result, label_data, quantiles)
        print("CRPS:{}".format(crps))
        return crps

    def cdf(self, pred_result, quantiles):
        y_cdf = np.zeros((pred_result.shape[0], quantiles.size + 2))
        y_cdf[:, 1:-1] = pred_result
        y_cdf[:, 0] = 2.0 * pred_result[:, 1] - pred_result[:, 2]
        y_cdf[:, -1] = 2.0 * pred_result[:, -2] - pred_result[:, -3]
        qs = np.zeros((1, quantiles.size + 2))
        qs[0, 1:-1] = quantiles
        qs[0, 0] = 0.0
        qs[0, -1] = 1.0
        return y_cdf, qs

    def CRPS(self, pred_result, label, quantiles):
        y_cdf, qs = self.cdf(pred_result, quantiles)
        ind = np.zeros(y_cdf.shape)
        ind[y_cdf > label.reshape(-1, 1)] = 1.0
        CRPS = np.trapz((qs - ind) ** 2.0, y_cdf)
        CRPS = np.mean(CRPS)
        return round(CRPS, 4)

    def CRPS_LPLS(selfself, pred_result, label):

        return None


def probabilistic_metrics(y_pred, y_true):
    """
    y_pred: N*l*n_tau
    y_true: N*l
    """
    metrics = prob_metric()


    PICP_95, ACE_95, PINAW_95, CWC_95, IS_95, PL_95 = metrics.get_metrics(0.025, 0.975, y_pred[:, :, 1],
                                                                          y_pred[:, :, -1], y_true)
    prob_95={'PICP_95': PICP_95, 'ACE_95': ACE_95, 'PINAW_95': PINAW_95, 'CWC_95': CWC_95, 'IS_95': IS_95, 'PL_95': PL_95}
    PICP_90, ACE_90, PINAW_90, CWC_90, IS_90, PL_90 = metrics.get_metrics(0.05, 0.95, y_pred[:, :, 2],
                                                                          y_pred[:, :, -2], y_true)
    prob_90={'PICP_90': PICP_90, 'ACE_90': ACE_90, 'PINAW_90': PINAW_90, 'CWC_90': CWC_90, 'IS_90': IS_90, 'PL_90': PL_90}
    PICP_85, ACE_85, PINAW_85, CWC_85, IS_85, PL_85 = metrics.get_metrics(0.075, 0.925, y_pred[:, :, 3],
                                                                          y_pred[:, :, -3], y_true)
    prob_85={'PICP_85': PICP_85, 'ACE_85': ACE_85, 'PINAW_85': PINAW_85, 'CWC_85': CWC_85, 'IS_85': IS_85, 'PL_85': PL_85}
    return {**prob_95, **prob_90, **prob_85}


def convert_numpy_to_list(obj):
    if isinstance(obj, np.ndarray):
        # 使用 np.around 函数对数组元素保留四位小数，再转换为列表
        return np.around(obj, decimals=4).tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_to_list(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_numpy_to_list(item) for item in obj]
    return obj


def get_average_metrics(y_pred, y_true):
    point_dict = metrics_point(y_pred, y_true)
    prob_dict = probabilistic_metrics(y_pred, y_true)
    point_ = {key: np.round(np.mean(value), 4) for key, value in point_dict.items()}
    prob_ = {key: np.round(np.mean(value), 4) for key, value in prob_dict.items()}
    metrics_dict = {**point_, **prob_}

    return metrics_dict
