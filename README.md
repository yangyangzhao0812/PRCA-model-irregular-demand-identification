# PRCA-model-irregular-demand-identification
Code for paper "Irregular passenger demand identification under disruptions: a robust principal component analysis-based approach"

Folder "Sim" and "MTR" represent the simulation and MTR experiments.
     * Syntheic data-see sim_rpca_ma.py.
     * MTR data-is not available due to the privacy issues. A sample data with the same size is used to execute the codes (see documents in MTR folder).

Sim fold
     For each parameter setting, the simulation is repeated for 20 times.
     * sim_rpca_ma.py: generation of synthetic, the implentation of RPCA and moving average methods.
     * plot_sim_fig.py: plots of estimation errors of RPCA and the moving average method.
    
MTR fold
     * main.py: all the functions used in the experiment.
     * GMM.py: the implentation of GMM to estimate the standard deviation of entry and exit demand noise.
     * spcp.py: the implentation of RPCA with the estimated parameter from GMM (target model).
     * spcp_empir.py: the implentation of RPCA without using GMM (baseline model).
     * MA.py: the implentation of moving average method (baseline model).
     * plot_heatmap_PRCA.py: the plot of decomposition results of the target model and moving average method.
     * plot_heatmap_vary_sigma.py: the plot of decomposition results of the target model and other PRCA models without using GMM.
