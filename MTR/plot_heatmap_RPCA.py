import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)
import datetime
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams['font.size'] = 12


def load_data():
    date = ['1-8', '1-9', '1-10', '1-11', '1-12', '1-15', '1-16', '1-17', '1-18', '1-19', '1-22', '1-23', '1-24',
            '1-25', '1-26', '1-29', '1-30', '1-31', '2-1', '2-2', '2-5', '2-6', '2-7', '3-1', '3-2', '3-5', '3-6',
            '3-7', '3-8', '3-9', '3-12', '3-13', '3-14', '3-15', '3-16', '3-19', '3-20', '3-21', '3-22', '3-23',
            '3-26', '3-27', '3-28', '3-29', '3-30', '4-2', '4-3', '4-4', '4-9', '4-10', '4-11', '4-12', '4-13', '4-16',
            '4-17', '4-18', '4-19', '4-20', '4-23', '4-24', '4-25', '4-26', '4-27']

    m = pd.read_csv(r"flow_" + str(slot) + "_" + label + ".csv", index_col=0)
    l = pd.read_csv(r"regular_flow_"+str(slot)+"_"+label+".csv", index_col=0)
    s = pd.read_csv(r"outlier_flow_" + str(slot) + "_" + label + ".csv", index_col=0)
    ma = pd.read_csv(r"MA_"+str(slot)+"_"+label+".csv", index_col=0)

    l.columns = date
    s.columns = date
    m.columns = date
    ma.columns = date

    print("l shape", l.shape)
    print("number of days", len(date))

    incident_date1 = ['1-8', '1-9', '1-10', '1-11', '1-12', '1-15']

    l_ = l[incident_date1].transpose()
    s_ = s[incident_date1].transpose()
    m_ = m[incident_date1].transpose()
    ma_ = ma[incident_date1].transpose()
    hour_slot = int(60/slot)

    L_flow = l_.to_numpy().reshape(-1, 76//hour_slot).transpose()
    S_flow = s_.to_numpy().reshape(-1, 76//hour_slot).transpose()
    M_flow = m_.to_numpy().reshape(-1, 76//hour_slot).transpose()
    MA_flow = ma_.to_numpy().reshape(-1, 76//hour_slot).transpose()

    L_flow_v1 = pd.DataFrame()
    S_flow_v1 = pd.DataFrame()
    M_flow_v1 = pd.DataFrame()
    MA_flow_v1 = pd.DataFrame()

    for i in range(0, L_flow.shape[1], 13):
        L_day = pd.DataFrame(L_flow[:, i:i+13])
        S_day = pd.DataFrame(S_flow[:, i:i+13])
        M_day = pd.DataFrame(M_flow[:, i:i+13])
        MA_day = pd.DataFrame(MA_flow[:, i:i+13])
        L_flow_v1 = pd.concat([L_flow_v1, L_day], axis=0)
        S_flow_v1 = pd.concat([S_flow_v1, S_day], axis=0)
        M_flow_v1 = pd.concat([M_flow_v1, M_day], axis=0)
        MA_flow_v1 = pd.concat([MA_flow_v1, MA_day], axis=0)
    return M_flow_v1, L_flow_v1, S_flow_v1, MA_flow_v1

def plot_heatmap(M_flow, L_flow, S_flow, MA_flow):  # 91*38
    if label == "in":
        vmin_3 = -3300
        vmax_3 = 3600
        vmin_4 = -3300
        vmax_4 = 3600
    else:
        vmin_3 = -2600
        vmax_3 = 5800
        vmin_4 = -2600
        vmax_4 = 5800

    fig = plt.figure(figsize=(10, 14)) # top=0.05
    plt.subplots_adjust(left=0.13, right=0.98, top=0.99, bottom=0.07, wspace=0.1)
    ax1 = fig.add_subplot(1,4,1)
    ax2 = fig.add_subplot(1,4,2)
    ax3 = fig.add_subplot(1,4,3)
    ax4 = fig.add_subplot(1,4,4)
    if label == "out":
        color_list1 = ["darkcyan", "tab:blue", "mediumseagreen", "sandybrown", "red"]
    else:
        color_list1 = [ "darkcyan",  "mediumseagreen", "tab:blue", "sandybrown", "red"]
    color_list2 = ["tab:blue","darkcyan","mediumseagreen", "sandybrown", "red"]
    ylabel = ["06:00", "10:00", "14:00", "18:00", "22:00"]*6
    y_ticks = np.array([])
    for i in range(6):
        y_tick = np.arange(2+38*i, 38*(i+1), 8)
        y_ticks = np.concatenate((y_ticks, y_tick))
    minor_yticks = np.arange(0, 38*6, 1)

    sta_list = ["HUH", "MKK", "KOT", "TAW", "FOT", "RAC", "UNI", "TAP", "TWO", "FAN", "SHS", "LOW", "LMC"]
    sns.heatmap(M_flow, ax=ax1, annot=False, cmap=color_list2, yticklabels=2, linecolor="tab:blue", linewidths=0.0,
                cbar_kws=dict(use_gridspec=False, location="top", pad=0.02, ticks=[0, 3000, 6000, 9000]))
    ax1.set_xticks(np.arange(0.5, 13.5, 1))
    ax1.set_yticks(y_ticks)
    ax1.set_yticks(minor_yticks, minor=True)
    date = ["1-8", "1-9", "1-10", "1-11", "1-12", "1-15"]
    for i in range(6):
        if i == 0:
            y_pos = 19
        else:
            y_pos = 19 + 38*i
        ax1.text(-3.5, y_pos, date[i], rotation=0, verticalalignment='center', horizontalalignment='right')
    ax1.set_yticklabels(ylabel)
    ax1.set_xticklabels(sta_list)
    ax1.set_xlabel("station\n(a)")
    ax1.set_ylabel("Date", labelpad=32.5)
    ax1.tick_params(axis="x", direction="out", rotation=90)

    ###########################################
    sns.heatmap(L_flow, ax=ax2, annot=False, cmap=color_list2, yticklabels=2, linecolor="tab:blue", linewidths=0.0,
                cbar_kws=dict(use_gridspec=False, location="top", pad=0.02, ticks=[0, 3000, 6000, 9000]))
    ax2.set_xticks(np.arange(0.5, 13.5, 1))
    ax2.set_yticks(y_ticks)
    ax2.set_yticks(minor_yticks, minor=True)
    ax2.set_yticklabels([])
    ax2.set_xticklabels(sta_list)
    ax2.set_xlabel("station\n(b)")
    ax2.tick_params(axis="x", direction="out", rotation=90)

    ###########################################
    sns.heatmap(S_flow, ax=ax3, vmax=vmax_3, vmin=vmin_3, linecolor="tab:blue", linewidths=0.0, annot=False, cmap=color_list1,
                yticklabels=2, cbar_kws=dict(use_gridspec=False, location="top", pad=0.02, ticks=[-2000, 0, 2000, 4000]))
    ax3.set_xticks(np.arange(0.5, 13.5, 1))
    ax3.set_yticks(y_ticks)
    ax3.set_yticks(minor_yticks, minor=True)
    ax3.set_yticklabels([])
    ax3.set_xticklabels(sta_list)
    ax3.set_xlabel("station\n(c)")
    ax3.tick_params(axis="x", direction="out", rotation=90)

    ###########################################
    sns.heatmap(MA_flow, ax=ax4, vmax=vmax_4, vmin=vmin_4, linecolor="tab:blue", linewidths=0.0, annot=False, cmap=color_list1,
                yticklabels=2, cbar_kws=dict(use_gridspec=False, location="top", pad=0.02, ticks=[-2000, 0, 2000, 4000]),  rasterized=False)
    ax4.set_xticks(np.arange(0.5, 13.5, 1))
    ax4.set_yticks(y_ticks)
    ax4.set_yticks(minor_yticks, minor=True)
    ax4.set_yticklabels([])
    ax4.set_xticklabels(sta_list)
    ax4.set_xlabel("station\n(d)")
    # ax4.yaxis.tick_right()
    # ax4.set_yticklabels(["1-8", "1-9", "1-10", "1-11", "1-12", "1-15", "1-23"])
    ax4.tick_params(axis="x", direction="out", rotation=90)
    for i in range(38, S_flow.shape[0], 38):
        ax1.hlines(i, 0, 13, colors="white", lw=1, ls="-")
        ax2.hlines(i, 0, 13, colors="white", lw=1, ls="-")
        ax3.hlines(i, 0, 13, colors="white", lw=1, ls="-")
        ax4.hlines(i, 0, 13, colors="white", lw=1, ls="-")
    fig.savefig(r"heatmap_"+label+".svg", dpi=fig.dpi)
    plt.show()
    return None


if __name__ == '__main__':
    slot = 30
    label = "out"
    L_flow, S_flow, M_flow, MA_flow = load_data()
    plot_heatmap(L_flow, S_flow, M_flow, MA_flow)
