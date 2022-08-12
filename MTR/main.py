from Github.MTR.GMM import *
from Github.MTR.spcp import *
from Github.MTR.MA import *
from Github.MTR.spcp_empir import *

"""
Step 1: demand noise (sigma:standard deviation) estimation using moving average method
Step 2: PRCA via APG, parameter sigma obtained from GMM is used
Step 3: Compared with moving average method, and APG without using GMM
"""

# step 1
y1 = pd.read_csv(r"demand_noise_in.csv", index_col=0).to_numpy().reshape(-1,1)
y2 = pd.read_csv(r"demand_noise_out.csv", index_col=0).to_numpy().reshape(-1,1)
aic_list1, bic_list1 = build_gmm(y1) # optimal GMM components selection
aic_list2, bic_list2 = build_gmm(y2)
AIC_BIC_plot(aic_list1, bic_list1, aic_list2, bic_list2)
variance1, variance2 = gmm_plot(y1, y2) # GMM fitting results

# step 2
M_out = pd.read_csv(r"RPCA_demand_out.csv", index_col=0) #observed entry or exit demand matrix
APG = StablePCP(sigma=np.sqrt(variance2), max_rank=1)  #
APG.fit(M_out) #m*n, m:daily passenger demand of all stations, n:number of days
L_APG = APG.get_low_rank()
S_APG = APG.get_sparse()

# Step 3
L_MA, S_MA = MA(M_out)
sigma = [6, 206, None]  # GMM validation in APG by varying sigma. For exit demand noise, the estimated sigma value is 106.
for sig in sigma:
    if sig is not None:
        APG = StablePCP(sigma=sig, max_rank=1)
        APG.fit(M_out)
        S_APG = APG.get_sparse()
    else:
        APG = StablePCP_empir(max_rank=1)
        APG.fit(M_out)
        S_APG = APG.get_sparse()