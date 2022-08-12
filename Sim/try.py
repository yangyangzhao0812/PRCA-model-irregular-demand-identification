import numpy as np
import pandas as pd
M_out = pd.read_csv(r"demand_RPCA_out.csv") # observed entry or exit demand matrix
M_out.loc[:, :] *=0.001
print(M_out)


