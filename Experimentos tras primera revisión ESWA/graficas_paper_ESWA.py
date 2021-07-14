# -*- coding: utf-8 -*-

# Hago las gráficas comparando los modelos Greedy y DQP

import matplotlib.pyplot as plt

# Número de niveles
dataset_sizes = [10,25,50,100,200]

# Coefs (num acciones modelo DQP/ num acciones random) para cada número de niveles
mean_action_coefs_each_num_lvs = [0.739, 0.549, 0.43, 0.411, 0.408]
std_action_coefs_each_num_lvs = [0.3	, 0.16, 0.083, 0.068, 0.053]

if __name__ == '__main__':
    plt.rcParams.update({'font.size': 12})
    plt.rcParams["font.family"] = "Helvetica"
    
    plt.errorbar(dataset_sizes, mean_action_coefs_each_num_lvs, std_action_coefs_each_num_lvs,
                 fmt="-.", marker='o', capsize=5)
    
    
    plt.grid(True)
    plt.ylabel("Action coefficient")
    plt.xlabel("Number of training levels")
    plt.ylim(0, 1.05)
    
    plt.savefig("grafica_action_coef_mean_and_std.png", dpi=500)
    
    plt.show()