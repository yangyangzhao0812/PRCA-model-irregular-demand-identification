import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
import math
import scipy.stats as stats
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams['font.size'] = 10

def build_gmm(y):
    com_num = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    cov_type = ["full"]
    aic_list = np.zeros((len(com_num), len(cov_type)))
    bic_list = np.zeros((len(com_num), len(cov_type)))
    for i, num in enumerate(com_num):
        for j, type in enumerate(cov_type):
            gmm = GaussianMixture(n_components=num, random_state=0, max_iter=500, covariance_type=type, init_params="kmeans").fit(y)
            aic = gmm.aic(y)
            bic = gmm.bic(y)
            aic_list[i][j] = int(aic)
            bic_list[i][j] = int(bic)
    return aic_list.reshape(-1), bic_list.reshape(-1)

def AIC_BIC_plot(aic_list1, bic_list1, aic_list2, bic_list2):
    fig = plt.figure(figsize=(5, 2.3))
    plt.subplots_adjust(left=0.13, right=0.92, top=0.9, bottom=0.2, wspace=0.4)
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)
    x = np.arange(1, 16, 1)
    ax1.plot(x, aic_list1, label="AIC", c="b", ls="-", lw=0.8, marker=".", markersize=3)
    ax1.plot(x, bic_list1, label="BIC", c="g", ls="-", lw=0.8, marker=".", markersize=3)
    maior_tick = np.array(np.arange(1, 16, 2))
    ax1.set_xticks(maior_tick)
    ax1.set_xticklabels(np.arange(1, 16, 2))
    ax1.axvline(x=3, ymin=0, ymax=280000, color="red", lw=0.5, ls="--")
    ax1.set_xlabel("n\n (a)", labelpad=0.15)
    ax1.set_ylabel("AIC/BIC Value", labelpad=0.5)
    ax1.ticklabel_format(axis='y', style='sci', scilimits=(4, 4))
    ax1.legend(loc=1, frameon=False, prop={'size': 8.5}, labelspacing=0.2)
    ax1.annotate("n=3", (3.5, 278750), color='black')

    #######################
    ax2.plot(x, aic_list2, label="AIC", c="b", ls="-", lw=0.8, marker=".", markersize=3)
    ax2.plot(x, bic_list2, label="BIC", c="g", ls="-", lw=0.8, marker=".", markersize=3)
    major_tick = np.array(np.arange(1, 16, 2))
    ax2.set_xticks(maior_tick)
    ax2.set_xticklabels(major_tick)
    ax2.set_xlabel("n\n (b)", labelpad=0.1)
    ax2.set_ylabel("AIC/BIC Value", labelpad=0.5)
    ax2.ticklabel_format(axis='y', style='sci', scilimits=(4, 4))
    ax2.axvline(x=3, ymin=0, ymax=280000, color="red", lw=0.5, ls="--")
    ax2.annotate("n=3", (3.5, 271500), color='black')
    ax2.legend(loc=1, frameon=False, prop={'size': 8.5}, labelspacing=0.2)
    # plt.savefig(r"AIC_BIC.svg")
    plt.show()
    return None

def gmm_plot(y1, y2):
    fig = plt.figure(figsize=(6, 2.3))
    plt.subplots_adjust(left=0.13, right=0.92, top=0.9, bottom=0.2, wspace=0.3)
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)
    n1_comp = 3
    n2_comp = 3
    gmm1 = GaussianMixture(n_components=n1_comp, random_state=0, max_iter=500, covariance_type="full",
                           init_params="kmeans").fit(y1)
    gmm2 = GaussianMixture(n_components=n2_comp, random_state=0, max_iter=500, covariance_type="full",
                           init_params="kmeans").fit(y2)
    mu1 = np.array(gmm1.means_).reshape(-1)
    mu1_double = np.square(mu1)
    var1 = np.array(gmm1.covariances_).reshape(-1)
    wei1 = np.array(gmm1.weights_).reshape(-1)
    sigma1 = np.sqrt(var1.reshape(-1))
    variance1 = np.dot(wei1, var1) + np.dot(wei1, mu1_double) - np.square(np.dot(wei1, mu1))

    mu2 = np.array(gmm2.means_).reshape(-1)
    mu2_double = np.square(mu2)
    var2 = np.array(gmm2.covariances_).reshape(-1)
    wei2 = np.array(gmm2.weights_).reshape(-1)
    sigma2 = np.sqrt(var2.reshape(-1))
    variance2 = np.dot(wei2, var2) + np.dot(wei2, mu2_double) - np.square(np.dot(wei2, mu2))

    mu1_upper_max = 0
    mu2_upper_max = 0
    mu1_lower_min = 0
    mu2_lower_min = 0
    for i in range(n1_comp):
        mu1_lower = mu1[i] - 3 * sigma1[i]
        mu1_upper = mu1[i] + 3 * sigma1[i]
        if mu1_upper <= mu1_upper_max:
            pass
        else:
            mu1_upper_max = mu1_upper
        if mu1_lower <= mu1_lower_min:
            mu1_lower_min = mu1_lower
        else:
            pass
        wei1_ = wei1[i]
        mu1_ = mu1[i]
        sigma1_ = sigma1[i]
        x1 = np.linspace(mu1_lower, mu1_upper, 500)
        x_whole = np.linspace(mu1_lower_min, mu1_upper_max, 500)
        color_list = ["blue", "green", "black"]
        ax1.plot(x1, wei1_ * stats.norm.pdf(x1, mu1_, sigma1_), color=color_list[i], lw=0.8, ls="-",
                 label="Component "+(i+1)*str("I"))

    logprob = gmm1.score_samples(x_whole.reshape(-1, 1))
    pdf = np.exp(logprob)
    ax1.plot(x_whole, pdf, lw=0.8, ls="-", color="red", label="GMM")
    ax1.hist(y1, bins=60, density=True, facecolor='darkgrey', edgecolor="darkgrey", rwidth=0.4, linewidth=0.4,
             label="Observed demand")  # darkgrey
    ax1.set_xlabel("Entry demand noise\n(a)", labelpad=1.5)
    ax1.set_ylabel("Probability density distribution", labelpad=1.5)
    ax1.legend(frameon=False, loc=2)
    for i in range(n2_comp):
        mu2_lower = mu2[i] - 3 * sigma2[i]
        mu2_upper = mu2[i] + 3 * sigma2[i]
        if mu2_upper <= mu2_upper_max:
            pass
        else:
            mu2_upper_max = mu2_upper
        if mu2_lower <= mu2_lower_min:
            mu2_lower_min = mu2_lower
        else:
            pass
        wei2_ = wei2[i]
        mu2_ = mu2[i]
        sigma2_ = sigma2[i]
        x2 = np.linspace(mu2_lower, mu2_upper, 500)
        x_whole = np.linspace(mu2_lower_min, mu2_upper_max, 500)
        color_list = ["blue", "green", "black"]
        ax2.plot(x2, wei2_ * stats.norm.pdf(x2, mu2_, sigma2_), color=color_list[i], lw=0.8, ls="-",
                 label="Component "+(i+1)*str("I"))
        ax2.plot(x2, wei2_ * stats.norm.pdf(x2, mu2_, sigma2_), color=color_list[i], lw=0.8, ls="-")
    logprob = gmm2.score_samples(x_whole.reshape(-1, 1))
    pdf = np.exp(logprob)
    ax2.plot(x_whole, pdf, lw=0.8, ls="-", color="red", label="GMM")
    ax2.hist(y2, bins=60, density=True, facecolor='darkgrey', edgecolor="darkgrey", rwidth=0.4, linewidth=0.4,
             label="Observed demand")
    ax2.set_xlabel("Exit demand noise\n(b)", labelpad=1.5)
    ax2.set_ylabel("Probability density distribution", labelpad=1.5)
    ax2.legend(frameon=False, loc=2)
    # fig.savefig(r"gmm_noise.svg")
    plt.show()
    return variance1, variance2




