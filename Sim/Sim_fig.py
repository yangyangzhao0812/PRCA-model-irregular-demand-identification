import numpy as np
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams['font.size'] = 8
plt.rcParams['text.usetex'] = False
greek_letterz=[chr(code) for code in range(945, 970)]
print(chr(947))
print(chr(948))

def fig1():
    with open('error_ma1.npy', 'rb') as f:
        error_ma = np.load(f)
    with open('error_rpca1.npy', 'rb') as f:
        error_rpca = np.load(f)
    sigma = [5, 10, 20, 40, 60, 80, 100, 120]
    n_sigma = len(sigma)
    p = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
    ##############
    MAPE_L_ma1 = error_ma[:len(sigma), :]
    MAPE_S_ma1= error_ma[len(sigma):len(sigma)*2, :]
    RMSE_L_ma1 = error_ma[len(sigma)*2:len(sigma)*3, :]
    RMSE_S_ma1 = error_ma[len(sigma)*3:len(sigma)*4, :]
    MAPE_L_rpca1 = error_rpca[:len(sigma), :]
    MAPE_S_rpca1 = error_rpca[len(sigma):len(sigma) * 2, :]
    RMSE_L_rpca1 = error_rpca[len(sigma) * 2:len(sigma) * 3, :]
    RMSE_S_rpca1 = error_rpca[len(sigma) * 3:len(sigma) * 4, :]
    ##############
    fig = plt.figure(figsize=(5, 2.3))
    plt.subplots_adjust(left=0.1, right=0.92, top=0.9, bottom=0.18, wspace=0.3)
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)
    for i in range(4):
        if i == 0:
            MAPE_L_ma1 = error_ma[i*n_sigma:(i+1)*n_sigma, 1:2].reshape(-1)
            MAPE_L_rpca1 = error_rpca[i * n_sigma:(i + 1) * n_sigma, 1:2].reshape(-1)
        elif i == 1:
            MAPE_S_ma1 = error_ma[i*n_sigma:(i+1)*n_sigma, 1:2].reshape(-1)
            MAPE_S_rpca1 = error_rpca[i * n_sigma:(i + 1) * n_sigma, 1:2].reshape(-1)
        elif i == 2:
            RMSE_L_ma1 = error_ma[i*n_sigma:(i+1)*n_sigma, 1:2].reshape(-1)
            RMSE_L_rpca1 = error_rpca[i * n_sigma:(i + 1) * n_sigma, 1:2].reshape(-1)
        else:
            RMSE_S_ma1 = error_ma[i*n_sigma:(i+1)*n_sigma, 1:2].reshape(-1)
            RMSE_S_rpca1 = error_rpca[i * n_sigma:(i + 1) * n_sigma, 1:2].reshape(-1)
    x = np.arange(8)
    xticklabel = [5, 10, 20, 40, 60, 80, 100, 120]
    ##### fig 1.1
    ax1.plot(x, MAPE_L_rpca1, label="L-RPCA", c="r", ls="--", lw=0.8, marker="8", markersize=3, markerfacecolor='none')
    ax1.plot(x, MAPE_S_rpca1, label="S-RPCA", c="b", ls="--", lw=0.8, marker="s", markersize=3)
    ax1.plot(x, MAPE_L_ma1, label="L-MA", c="r", ls="-", lw=0.8, marker="8", markersize=3, markerfacecolor='none') # markerfacecolor='none'
    ax1.plot(x, MAPE_S_ma1, label="S-MA", c="b", ls="-", lw=0.8, marker="s", markersize=3)
    ax1.set_xticks(np.arange(8))
    ax1.set_xticklabels(xticklabel)
    ax1.set_xlabel(chr(948)+"\n(a)", labelpad=0.05)
    ax1.set_ylabel("MAPE (%)", labelpad=1)
    ax1.legend(frameon=False, prop={'size': 8}, labelspacing=0.1)
    for i in range(4):
        if i == 0:
            MAPE_L_ma2 = error_ma[4:5, :].reshape(-1)
            MAPE_L_rpca2 = error_rpca[4:5, :].reshape(-1)
        elif i == 1:
            MAPE_S_ma2 = error_ma[4+n_sigma * i:5 + n_sigma * i, :].reshape(-1)
            MAPE_S_rpca2 = error_rpca[4 + n_sigma * i:5 + n_sigma * i, :].reshape(-1)
        elif i == 2:
            RMSE_L_ma2 = error_ma[4 + n_sigma * i:5 + n_sigma * i, :].reshape(-1)
            RMSE_L_rpca2 = error_rpca[4 + n_sigma * i:5 + n_sigma * i, :].reshape(-1)
        else:
            RMSE_S_ma2 = error_ma[4 + n_sigma * i:5 + n_sigma * i, :].reshape(-1)
            RMSE_S_rpca2 = error_rpca[4 + n_sigma * i:5 + n_sigma * i, :].reshape(-1)
    x = np.arange(6)
    ax2.plot(x, MAPE_L_rpca2, label="L-RPCA", c="r", ls="--", lw=0.8, marker="8", markersize=3, markerfacecolor='none')
    ax2.plot(x, MAPE_S_rpca2, label="S-RPCA", c="b", ls="--", lw=0.8, marker="s", markersize=3)
    ax2.plot(x, MAPE_L_ma2, label="L-MA", c="r", ls="-", lw=0.8, marker="8", markersize=3, markerfacecolor='none')
    ax2.plot(x, MAPE_S_ma2, label="S-MA", c="b", ls="-", lw=0.8, marker="s", markersize=3)
    ax2.set_xticks(np.arange(6))
    ax2.set_xticklabels(p)
    ax2.set_xlabel("p\n(b)", labelpad=0.05)
    ax2.set_ylabel("MAPE (%)", labelpad=1)
    ax2.legend(frameon=False, prop={'size': 8}, bbox_to_anchor=(0.50, 0.72), labelspacing=0.1)
    fig.savefig(r"mape_1-2.svg")

    ##### fig 1.2
    fig = plt.figure(figsize=(5, 2.3))
    plt.subplots_adjust(left=0.1, right=0.92, top=0.9, bottom=0.18, wspace=0.3)
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)
    x = np.arange(8)
    ax1.plot(x, RMSE_L_rpca1, label="L-RPCA", c="r", ls="--", lw=0.8, marker="8", markersize=3, markerfacecolor='none')
    ax1.plot(x, RMSE_S_rpca1, label="S-RPCA", c="b", ls="--", lw=0.8, marker="s", markersize=3)
    ax1.plot(x, RMSE_L_ma1, label="L-MA", c="r", ls="-", lw=0.8, marker="8", markersize=3, markerfacecolor='none')
    ax1.plot(x, RMSE_S_ma1, label="S-MA", c="b", ls="-", lw=0.8, marker="s", markersize=3)
    ax1.set_xticks(x)
    ax1.set_xticklabels(xticklabel)
    ax1.set_xlabel(chr(948)+"\n(a)", labelpad=0.05)
    ax1.set_ylabel("RMSE", labelpad=3)
    ax1.legend(frameon=False, prop={'size': 8}, labelspacing=0.1)
    x = np.arange(6)
    ax2.plot(x, RMSE_L_rpca2, label="L-RPCA", c="r", ls="--", lw=0.8, marker="8", markersize=3, markerfacecolor='none')
    ax2.plot(x, RMSE_S_rpca2, label="S-RPCA", c="b", ls="--", lw=0.8, marker="s", markersize=3)
    ax2.plot(x, RMSE_L_ma2, label="L-MA", c="r", ls="-", lw=0.8, marker="8", markersize=3, markerfacecolor='none')
    ax2.plot(x, RMSE_S_ma2, label="S-MA", c="b", ls="-", lw=0.8, marker="s", markersize=3)
    ax2.set_xlabel("p\n(b)", labelpad=0.05)
    ax2.set_xticks(x)
    ax2.set_xticklabels(p)
    ax2.set_ylabel("RMSE", labelpad=3)
    ax2.legend(frameon=False, prop={'size': 8}, bbox_to_anchor=(0.49, 0.58), labelspacing=0.1)
    fig.savefig(r"rmse_1-2.svg")
    plt.show()
    return None

def fig2():
    with open('error_ma2.npy', 'rb') as f:
        error_ma = np.load(f)
    with open('error_rpca2.npy', 'rb') as f:
        error_rpca = np.load(f)
    dim = [10, 20, 40, 60, 80, 160, 320]
    rank = [1, 2, 3, 4, 5, 6, 7]
    MAPE_L_rpca1 = error_rpca[:7, :]
    MAPE_S_rpca1 = error_rpca[7:14, :]
    RMSE_L_rpca1 = error_rpca[14:21, :]
    RMSE_S_rpca1 = error_rpca[21:28, :]

    MAPE_L_ma1 = error_ma[:7, :]
    MAPE_S_ma1 = error_ma[7:14, :]
    RMSE_L_ma1 = error_ma[14:21, :]
    RMSE_S_ma1 = error_ma[21:28, :]

    fig = plt.figure(figsize=(5, 2.3))
    plt.subplots_adjust(left=0.1, right=0.92, top=0.9, bottom=0.18, wspace=0.3)
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)
    for i in range(4):
        if i == 0:
            MAPE_L_rpca1 = error_rpca[i * len(dim):(i + 1) * len(dim), 6:7].reshape(-1)
            MAPE_L_ma1 = error_ma[i * len(dim):(i + 1) * len(dim), 6:7].reshape(-1)
        elif i == 1:
            MAPE_S_rpca1 = error_rpca[i * len(dim):(i + 1) * len(dim), 6:7].reshape(-1)
            MAPE_S_ma1 = error_ma[i * len(dim):(i + 1) * len(dim), 6:7].reshape(-1)
        elif i == 2:
            RMSE_L_rpca1 = error_rpca[i * len(dim):(i + 1) * len(dim), 6:7].reshape(-1)
            RMSE_L_ma1 = error_ma[i * len(dim):(i + 1) * len(dim), 6:7].reshape(-1)
        else:
            RMSE_S_rpca1 = error_rpca[i * len(dim):(i + 1) * len(dim), 6:7].reshape(-1)
            RMSE_S_ma1 = error_ma[i * len(dim):(i + 1) * len(dim), 6:7].reshape(-1)
    for i in range(4):
        if i == 0:
            MAPE_L_rpca2 = error_rpca[5:6, :].reshape(-1)
            MAPE_L_ma2 = error_ma[5:6, :].reshape(-1)
        elif i == 1:
            MAPE_S_rpca2 = error_rpca[5 + len(dim) * i:6 + len(dim) * i, :].reshape(-1)
            MAPE_S_ma2 = error_ma[5 + len(dim) * i:6 + len(dim) * i, :].reshape(-1)
        elif i == 2:
            RMSE_L_rpca2 = error_rpca[5 + len(dim) * i:6 + len(dim) * i, :].reshape(-1)
            RMSE_L_ma2 = error_ma[5 + len(dim) * i:6 + len(dim) * i, :].reshape(-1)
        else:
            RMSE_S_rpca2 = error_rpca[5 + len(dim) * i:6 + len(dim) * i, :].reshape(-1)
            RMSE_S_ma2 = error_ma[5 + len(dim) * i:6 + len(dim) * i, :].reshape(-1)
    x = np.arange(7)
    xticklabel = [10, 20, 40, 60, 80, 160, 320]
    ##### fig 1
    ax1.plot(x, MAPE_L_rpca1, label="L-RPCA", c="r", ls="--", lw=0.8, marker="8", markersize=3, markerfacecolor='none')
    ax1.plot(x, MAPE_S_rpca1, label="S-RPCA", c="b", ls="--", lw=0.8, marker="s", markersize=3)
    ax1.plot(x, MAPE_L_ma1, label="L-MA", c="r", ls="-", lw=0.8, marker="8", markersize=3, markerfacecolor='none')
    ax1.plot(x, MAPE_S_ma1, label="S-MA", c="b", ls="-", lw=0.8, marker="s", markersize=3)
    ax1.set_xticks(np.arange(7))
    ax1.set_xticklabels(xticklabel)
    ax1.set_xlabel("n\n(c)", labelpad=0.05)
    ax1.set_ylabel("MAPE (%)", labelpad=0.1)
    ax1.legend(frameon=False, prop={'size': 8}, bbox_to_anchor=(0.45, 0.50), labelspacing=0.1)
    ax2.plot(x, MAPE_L_rpca2, label="L-RPCA", c="r", ls="--", lw=0.8, marker="8", markersize=3, markerfacecolor='none')
    ax2.plot(x, MAPE_S_rpca2, label="S-RPCA", c="b", ls="--", lw=0.8, marker="s", markersize=3)
    ax2.plot(x, MAPE_L_ma2, label="L-MA", c="r", ls="-", lw=0.8, marker="8", markersize=3, markerfacecolor='none')
    ax2.plot(x, MAPE_S_ma2, label="S-MA", c="b", ls="-", lw=0.8, marker="s", markersize=3)
    ax2.set_xticks(np.arange(7))
    ax2.set_xticklabels(rank)
    ax2.set_xlabel("r\n(d)", labelpad=0.05)
    ax2.set_ylabel("MAPE (%)", labelpad=0.1)
    ax2.legend(frameon=False, prop={'size': 8}, labelspacing=0.1)
    fig.savefig(r"mape_3-4.svg")
    #########################
    fig = plt.figure(figsize=(5, 2.3))
    plt.subplots_adjust(left=0.1, right=0.92, top=0.9, bottom=0.18, wspace=0.3)
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)
    x = np.arange(7)
    ax1.plot(x, RMSE_L_rpca1, label="L-RPCA", c="r", ls="--", lw=0.8, marker="8", markersize=3, markerfacecolor='none')
    ax1.plot(x, RMSE_S_rpca1, label="S-RPCA", c="b", ls="--", lw=0.8, marker="s", markersize=3)
    ax1.plot(x, RMSE_L_ma1, label="L-MA", c="r", ls="-", lw=0.8, marker="8", markersize=3, markerfacecolor='none')
    ax1.plot(x, RMSE_S_ma1, label="S-MA", c="b", ls="-", lw=0.8, marker="s", markersize=3)
    ax1.set_xticks(np.arange(7))
    ax1.set_xticklabels(xticklabel)
    ax1.set_xlabel("n\n(c)", labelpad=0.05)
    ax1.set_ylabel("RMSE", labelpad=0.1)
    ax1.legend(frameon=False, prop={'size': 8}, labelspacing=0.1)  # bbox_to_anchor=(1.04, 0.618)
    ax2.plot(x, RMSE_L_rpca2, label="L-RPCA", c="r", ls="--", lw=0.8, marker="8", markersize=3, markerfacecolor='none')
    ax2.plot(x, RMSE_S_rpca2, label="S-RPCA", c="b", ls="--", lw=0.8, marker="s", markersize=3)
    ax2.plot(x, RMSE_L_ma2, label="L-MA", c="r", ls="-", lw=0.8, marker="8", markersize=3, markerfacecolor='none')
    ax2.plot(x, RMSE_S_ma2, label="S-MA", c="b", ls="-", lw=0.8, marker="s", markersize=3)
    ax2.set_xticks(np.arange(7))
    ax2.set_xticklabels(rank)
    ax2.set_xlabel("r\n(d)", labelpad=0.05)
    ax2.set_ylabel("RMSE", labelpad=0.1)
    ax2.legend(frameon=False, prop={'size': 8}, bbox_to_anchor=(0.99, 0.85), labelspacing=0.1)  #
    plt.savefig(r"rmse_3-4.svg")
    plt.show()
    return None

if __name__ == '__main__':
    # fig1()
    fig2()