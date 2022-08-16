import numpy as np
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams['font.size'] = 8
plt.rcParams['text.usetex'] = False
greek_letterz = [chr(code) for code in range(945, 970)]
print(chr(947))
print(chr(948))

def fig1():
    with open('error_v1_01.npy', 'rb') as f:
        error = np.load(f)
    sigma = [5, 10, 20, 40, 60, 80, 100, 120]
    n_sigma = len(sigma)
    p = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
    ##############
    MAPE_L = error[:3]
    MAPE_S = error[3:6]
    RMSE_L = error[6:9]
    RMSE_S = error[9:12]
    
    MAPE_L_rpca = MAPE_L[0][:len(sigma), :]
    MAPE_S_rpca = MAPE_S[0][:len(sigma), :]
    RMSE_L_rpca = RMSE_L[0][:len(sigma), :]
    RMSE_S_rpca = RMSE_S[0][:len(sigma), :]

    MAPE_L_rpca_empir = MAPE_L[1][:len(sigma), :]
    MAPE_S_rpca_empir = MAPE_S[1][:len(sigma), :]
    RMSE_L_rpca_empir = RMSE_L[1][:len(sigma), :]
    RMSE_S_rpca_empir = RMSE_S[1][:len(sigma), :]

    MAPE_L_ma = MAPE_L[2][:len(sigma), :]
    MAPE_S_ma = MAPE_S[2][:len(sigma), :]
    RMSE_L_ma = RMSE_L[2][:len(sigma), :]
    RMSE_S_ma = RMSE_S[2][:len(sigma), :]

    ##############
    MAPE_L_rpca1_f1 = MAPE_L_rpca[:, 1:2].reshape(-1)
    MAPE_L_rpca2_f1 = MAPE_L_rpca_empir[:, 1:2].reshape(-1)
    MAPE_L_ma_f1 = MAPE_L_ma[:, 1:2].reshape(-1)

    MAPE_S_rpca1_f1 = MAPE_S_rpca[:, 1:2].reshape(-1)
    MAPE_S_rpca2_f1 = MAPE_S_rpca_empir[:, 1:2].reshape(-1)
    MAPE_S_ma_f1 = MAPE_S_ma[:, 1:2].reshape(-1)

    MAPE_L_rpca1_f2 = MAPE_L_rpca[6:7, :].reshape(-1)
    MAPE_L_rpca2_f2 = MAPE_L_rpca_empir[6:7, :].reshape(-1)
    MAPE_L_ma_f2 = MAPE_L_ma[6:7, :].reshape(-1)

    MAPE_S_rpca1_f2 = MAPE_S_rpca[6:7, :].reshape(-1)
    MAPE_S_rpca2_f2 = MAPE_S_rpca_empir[6:7, :].reshape(-1)
    MAPE_S_ma_f2 = MAPE_S_ma[6:7, :].reshape(-1)

    #############
    RMSE_L_rpca1_f1 = RMSE_L_rpca[:, 1:2].reshape(-1)
    RMSE_L_rpca2_f1 = RMSE_L_rpca_empir[:, 1:2].reshape(-1)
    RMSE_L_ma_f1 = RMSE_L_ma[:, 1:2].reshape(-1)

    RMSE_S_rpca1_f1 = RMSE_S_rpca[:, 1:2].reshape(-1)
    RMSE_S_rpca2_f1 = RMSE_S_rpca_empir[:, 1:2].reshape(-1)
    RMSE_S_ma_f1 = RMSE_S_ma[:, 1:2].reshape(-1)

    RMSE_L_rpca1_f2 = RMSE_L_rpca[6:7, :].reshape(-1)
    RMSE_L_rpca2_f2 = RMSE_L_rpca_empir[6:7, :].reshape(-1)
    RMSE_L_ma_f2 = RMSE_L_ma[6:7, :].reshape(-1)

    RMSE_S_rpca1_f2 = RMSE_S_rpca[6:7, :].reshape(-1)
    RMSE_S_rpca2_f2 = RMSE_S_rpca_empir[6:7, :].reshape(-1)
    RMSE_S_ma_f2 = RMSE_S_ma[6:7, :].reshape(-1)
    #############

    fig = plt.figure(figsize=(5, 2.3))
    plt.subplots_adjust(left=0.1, right=0.92, top=0.9, bottom=0.18, wspace=0.3)
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)

    x = np.arange(8)
    xticklabel = [5, 10, 20, 40, 60, 80, 100, 120]

    ##### fig 1a
    ax1.plot(x, MAPE_L_rpca1_f1, label="L-RPCA-I", c="r", ls="--", lw=0.8, marker="8", markersize=3, markerfacecolor='none')
    ax1.plot(x, MAPE_L_rpca2_f1, label="L-RPCA-II", c="g", ls="--", lw=0.8, marker="d", markersize=3, markerfacecolor='none')
    ax1.plot(x, MAPE_L_ma_f1, label="L-MA", c="b", ls="--", lw=0.8, marker="s", markersize=3, markerfacecolor='none')
    
    ax1.plot(x, MAPE_S_rpca1_f1, label="S-RPCA-I", c="r", ls="-", lw=0.8, marker="8", markersize=3)
    ax1.plot(x, MAPE_S_rpca2_f1, label="S-RPCA-II", c="g", ls="-", lw=0.8, marker="d", markersize=3)
    ax1.plot(x, MAPE_S_ma_f1, label="S-MA", c="b", ls="-", lw=0.8, marker="s", markersize=3)
    ax1.set_xticks(np.arange(8))
    ax1.set_xticklabels(xticklabel)
    ax1.set_xlabel(chr(948)+"\n(a)", labelpad=0.05)
    ax1.set_ylabel("MAPE (%)", labelpad=1)
    ax1.legend(frameon=False, prop={'size': 6.5}, labelspacing=0.1)
    
    ##### fig 1b
    x = np.arange(6)
    ax2.plot(x, MAPE_L_rpca1_f2, label="L-RPCA-I", c="r", ls="--", lw=0.8, marker="8", markersize=3, markerfacecolor='none')
    ax2.plot(x, MAPE_L_rpca2_f2, label="L-RPCA-II", c="g", ls="--", lw=0.8, marker="d", markersize=3, markerfacecolor='none')
    ax2.plot(x, MAPE_L_ma_f2, label="L-MA", c="b", ls="--", lw=0.8, marker="s", markersize=3, markerfacecolor='none')

    ax2.plot(x, MAPE_S_rpca1_f2, label="S-RPCA-I", c="r", ls="-", lw=0.8, marker="8", markersize=3)
    ax2.plot(x, MAPE_S_rpca2_f2, label="S-RPCA-II", c="g", ls="-", lw=0.8, marker="d", markersize=3)
    ax2.plot(x, MAPE_S_ma_f2, label="S-MA", c="b", ls="-", lw=0.8, marker="s", markersize=3)

    ax2.set_xticks(np.arange(6))
    ax2.set_xticklabels(p)
    ax2.set_xlabel("p\n(b)", labelpad=0.05)
    ax2.set_ylabel("MAPE (%)", labelpad=1)
    ax2.legend(frameon=False, prop={'size': 6.5},  bbox_to_anchor=(0.53, 0.55), labelspacing=0.1)
    fig.savefig(r"mape_1-2.svg")

    ##### fig 2a
    fig = plt.figure(figsize=(5, 2.3))
    plt.subplots_adjust(left=0.1, right=0.92, top=0.9, bottom=0.18, wspace=0.3)
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)
    x = np.arange(8)
    ax1.plot(x, RMSE_L_rpca1_f1, label="L-RPCA-I", c="r", ls="--", lw=0.8, marker="8", markersize=3, markerfacecolor='none')
    ax1.plot(x, RMSE_L_rpca2_f1, label="L-RPCA-II", c="g", ls="--", lw=0.8, marker="v", markersize=3, markerfacecolor='none')
    ax1.plot(x, RMSE_L_ma_f1, label="L-MA", c="b", ls="--", lw=0.8, marker="s", markersize=3, markerfacecolor='none')

    ax1.plot(x, RMSE_S_rpca1_f1, label="S-RPCA-I", c="r", ls="-", lw=0.8, marker="8", markersize=3)
    ax1.plot(x, RMSE_S_rpca2_f1, label="S-RPCA-II", c="g", ls="-", lw=0.8, marker="s", markersize=3)
    ax1.plot(x, RMSE_S_ma_f1, label="S-MA", c="b", ls="-", lw=0.8, marker="s", markersize=3)
    
    ax1.set_xticks(x)
    ax1.set_xticklabels(xticklabel)
    ax1.set_xlabel(chr(948)+"\n(a)", labelpad=0.05)
    ax1.set_ylabel("RMSE", labelpad=3)
    ax1.legend(frameon=False, prop={'size': 6.5}, labelspacing=0.1)

    ##### fig 2b
    x = np.arange(6)
    ax2.plot(x, RMSE_L_rpca1_f2, label="L-RPCA-I", c="r", ls="--", lw=0.8, marker="8", markersize=3, markerfacecolor='none')
    ax2.plot(x, RMSE_L_rpca2_f2, label="L-RPCA-II", c="g", ls="--", lw=0.8, marker="d", markersize=3, markerfacecolor='none')
    ax2.plot(x, RMSE_L_ma_f2, label="L-MA", c="b", ls="--", lw=0.8, marker="s", markersize=3, markerfacecolor='none')
    
    ax2.plot(x, RMSE_S_rpca1_f2, label="S-RPCA-I", c="r", ls="-", lw=0.8, marker="8", markersize=3)
    ax2.plot(x, RMSE_S_rpca2_f2, label="S-RPCA-II", c="g", ls="-", lw=0.8, marker="s", markersize=3)
    ax2.plot(x, RMSE_S_ma_f2, label="S-MA", c="b", ls="-", lw=0.8, marker="s", markersize=3)
    ax2.set_xlabel("p\n(b)", labelpad=0.05)
    ax2.set_xticks(x)
    ax2.set_xticklabels(p)
    ax2.set_ylabel("RMSE", labelpad=3)
    ax2.legend(frameon=False, prop={'size': 6.5}, labelspacing=0.1)
    fig.savefig(r"rmse_1-2.svg")
    plt.show()
    return None

def fig2():
    with open('error_v2_01.npy', 'rb') as f:
        error = np.load(f)
    dim = [100, 200, 300, 400, 500, 600]
    rank = [1, 2, 3, 4, 5, 6, 7]
    ##############
    MAPE_L = error[:3]
    MAPE_S = error[3:6]
    RMSE_L = error[6:9]
    RMSE_S = error[9:12]

    MAPE_L_rpca = MAPE_L[0][:len(dim), :]
    MAPE_S_rpca = MAPE_S[0][:len(dim), :]
    RMSE_L_rpca = RMSE_L[0][:len(dim), :]
    RMSE_S_rpca = RMSE_S[0][:len(dim), :]

    MAPE_L_rpca_empir = MAPE_L[1][:len(dim), :]
    MAPE_S_rpca_empir = MAPE_S[1][:len(dim), :]
    RMSE_L_rpca_empir = RMSE_L[1][:len(dim), :]
    RMSE_S_rpca_empir = RMSE_S[1][:len(dim), :]

    MAPE_L_ma = MAPE_L[2][:len(dim), :]
    MAPE_S_ma = MAPE_S[2][:len(dim), :]
    RMSE_L_ma = RMSE_L[2][:len(dim), :]
    RMSE_S_ma = RMSE_S[2][:len(dim), :]

    ##############
    MAPE_L_rpca1_f1 = MAPE_L_rpca[:, 6:7].reshape(-1)
    MAPE_L_rpca2_f1 = MAPE_L_rpca_empir[:, 6:7].reshape(-1)
    MAPE_L_ma_f1 = MAPE_L_ma[:, 6:7].reshape(-1)

    MAPE_S_rpca1_f1 = MAPE_S_rpca[:, 6:7].reshape(-1)
    MAPE_S_rpca2_f1 = MAPE_S_rpca_empir[:, 6:7].reshape(-1)
    MAPE_S_ma_f1 = MAPE_S_ma[:, 6:7].reshape(-1)

    MAPE_L_rpca1_f2 = MAPE_L_rpca[2:3, :].reshape(-1)
    MAPE_L_rpca2_f2 = MAPE_L_rpca_empir[2:3, :].reshape(-1)
    MAPE_L_ma_f2 = MAPE_L_ma[2:3, :].reshape(-1)

    MAPE_S_rpca1_f2 = MAPE_S_rpca[2:3, :].reshape(-1)
    MAPE_S_rpca2_f2 = MAPE_S_rpca_empir[2:3, :].reshape(-1)
    MAPE_S_ma_f2 = MAPE_S_ma[2:3, :].reshape(-1)

    #############
    RMSE_L_rpca1_f1 = RMSE_L_rpca[:, 6:7].reshape(-1)
    RMSE_L_rpca2_f1 = RMSE_L_rpca_empir[:, 6:7].reshape(-1)
    RMSE_L_ma_f1 = RMSE_L_ma[:,6:7].reshape(-1)

    RMSE_S_rpca1_f1 = RMSE_S_rpca[:, 6:7].reshape(-1)
    RMSE_S_rpca2_f1 = RMSE_S_rpca_empir[:, 6:7].reshape(-1)
    RMSE_S_ma_f1 = RMSE_S_ma[:, 6:7].reshape(-1)

    RMSE_L_rpca1_f2 = RMSE_L_rpca[2:3, :].reshape(-1)
    RMSE_L_rpca2_f2 = RMSE_L_rpca_empir[2:3, :].reshape(-1)
    RMSE_L_ma_f2 = RMSE_L_ma[2:3, :].reshape(-1)

    RMSE_S_rpca1_f2 = RMSE_S_rpca[2:3, :].reshape(-1)
    RMSE_S_rpca2_f2 = RMSE_S_rpca_empir[2:3, :].reshape(-1)
    RMSE_S_ma_f2 = RMSE_S_ma[2:3, :].reshape(-1)

    #############
    fig = plt.figure(figsize=(5, 2.3))
    plt.subplots_adjust(left=0.1, right=0.92, top=0.9, bottom=0.18, wspace=0.3)
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)
    x = np.arange(6)
    xticklabel = [100, 200, 300, 400, 500, 600]
    ##### fig 1
    ax1.plot(x, MAPE_L_rpca1_f1, label="L-RPCA-I", c="r", ls="--", lw=0.8, marker="8", markersize=3,
             markerfacecolor='none')
    ax1.plot(x, MAPE_L_rpca2_f1, label="L-RPCA-II", c="g", ls="--", lw=0.8, marker="d", markersize=3,
             markerfacecolor='none')
    ax1.plot(x, MAPE_L_ma_f1, label="L-MA", c="b", ls="--", lw=0.8, marker="s", markersize=3, markerfacecolor='none')

    ax1.plot(x, MAPE_S_rpca1_f1, label="S-RPCA-I", c="r", ls="-", lw=0.8, marker="8", markersize=3)
    ax1.plot(x, MAPE_S_rpca2_f1, label="S-RPCA-II", c="g", ls="-", lw=0.8, marker="d", markersize=3)
    ax1.plot(x, MAPE_S_ma_f1, label="S-MA", c="b", ls="-", lw=0.8, marker="s", markersize=3)
    ax1.set_xticks(np.arange(6))
    ax1.set_xticklabels(xticklabel)
    ax1.set_xlabel("n\n(c)", labelpad=0.05)
    ax1.set_ylabel("MAPE (%)", labelpad=0.1)
    ax1.legend(frameon=False, prop={'size': 6.5}, bbox_to_anchor=(0.45, 0.50), labelspacing=0.1) # bbox_to_anchor=(0.99, 0.85),

    x = np.arange(7)
    ax2.plot(x, MAPE_L_rpca1_f2, label="L-RPCA-I", c="r", ls="--", lw=0.8, marker="8", markersize=3,
             markerfacecolor='none')
    ax2.plot(x, MAPE_L_rpca2_f2, label="L-RPCA-II", c="g", ls="--", lw=0.8, marker="d", markersize=3,
             markerfacecolor='none')
    ax2.plot(x, MAPE_L_ma_f2, label="L-MA", c="b", ls="--", lw=0.8, marker="s", markersize=3, markerfacecolor='none')
    ax2.plot(x, MAPE_S_rpca1_f2, label="S-RPCA-I", c="r", ls="-", lw=0.8, marker="8", markersize=3)
    ax2.plot(x, MAPE_S_rpca2_f2, label="S-RPCA-II", c="g", ls="-", lw=0.8, marker="d", markersize=3)
    ax2.plot(x, MAPE_S_ma_f2, label="S-MA", c="b", ls="-", lw=0.8, marker="s", markersize=3)
    ax2.set_xticks(np.arange(7))
    ax2.set_xticklabels(rank)
    ax2.set_xlabel("r\n(d)", labelpad=0.05)
    ax2.set_ylabel("MAPE (%)", labelpad=0.1)
    ax2.legend(frameon=False, prop={'size': 6.5}, labelspacing=0.1) # bbox_to_anchor=(0.99, 0.85),
    fig.savefig(r"mape_3-4.svg")
    #########################
    fig = plt.figure(figsize=(5, 2.3))
    plt.subplots_adjust(left=0.1, right=0.92, top=0.9, bottom=0.18, wspace=0.3)
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)
    x = np.arange(6)
    ax1.plot(x, RMSE_L_rpca1_f1, label="L-RPCA-I", c="r", ls="--", lw=0.8, marker="8", markersize=3,
             markerfacecolor='none')
    ax1.plot(x, RMSE_L_rpca2_f1, label="L-RPCA-II", c="g", ls="--", lw=0.8, marker="v", markersize=3,
             markerfacecolor='none')
    ax1.plot(x, RMSE_L_ma_f1, label="L-MA", c="b", ls="--", lw=0.8, marker="s", markersize=3, markerfacecolor='none')

    ax1.plot(x, RMSE_S_rpca1_f1, label="S-RPCA-I", c="r", ls="-", lw=0.8, marker="8", markersize=3)
    ax1.plot(x, RMSE_S_rpca2_f1, label="S-RPCA-II", c="g", ls="-", lw=0.8, marker="s", markersize=3)
    ax1.plot(x, RMSE_S_ma_f1, label="S-MA", c="b", ls="-", lw=0.8, marker="s", markersize=3)
    ax1.set_xticks(np.arange(6))
    ax1.set_xticklabels(xticklabel)
    ax1.set_xlabel("n\n(c)", labelpad=0.05)
    ax1.set_ylabel("RMSE", labelpad=0.1)
    ax1.legend(frameon=False, prop={'size': 6.5}, labelspacing=0.1)  # bbox_to_anchor=(1.04, 0.618)

    x = np.arange(7)
    ax2.plot(x, RMSE_L_rpca1_f2, label="L-RPCA-I", c="r", ls="--", lw=0.8, marker="8", markersize=3,
             markerfacecolor='none')
    ax2.plot(x, RMSE_L_rpca2_f2, label="L-RPCA-II", c="g", ls="--", lw=0.8, marker="v", markersize=3,
             markerfacecolor='none')
    ax2.plot(x, RMSE_L_ma_f2, label="L-MA", c="b", ls="--", lw=0.8, marker="s", markersize=3, markerfacecolor='none')

    ax2.plot(x, RMSE_S_rpca1_f2, label="S-RPCA-I", c="r", ls="-", lw=0.8, marker="8", markersize=3)
    ax2.plot(x, RMSE_S_rpca2_f2, label="S-RPCA-II", c="g", ls="-", lw=0.8, marker="s", markersize=3)
    ax2.plot(x, RMSE_S_ma_f2, label="S-MA", c="b", ls="-", lw=0.8, marker="s", markersize=3)
    ax2.set_xticks(np.arange(7))
    ax2.set_xticklabels(rank)
    ax2.set_xlabel("r\n(d)", labelpad=0.05)
    ax2.set_ylabel("RMSE", labelpad=0.1)
    ax2.legend(frameon=False, prop={'size': 6.5}, labelspacing=0.1)  #bbox_to_anchor=(0.99, 0.85),
    plt.savefig(r"rmse_3-4.svg")
    plt.show()
    return None

if __name__ == '__main__':
    fig1()
    fig2()