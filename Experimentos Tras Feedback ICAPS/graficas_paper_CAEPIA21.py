# -*- coding: utf-8 -*-

# Hago las gráficas comparando los modelos Greedy y DQP

import matplotlib.pyplot as plt

# Número de niveles
dataset_sizes = [5, 25, 50, 75, 100]

# Coefs (num acciones modelo/ num acciones random) para cada número de niveles
# Modelo Greedy (gamma 0)
coefs_greedy = [0.77722304, 0.54350503, 0.4623118 , 0.44439325, 0.42983589]

# Modelo DQP (gamma 0.7)
coefs_DQP = [0.81360181, 0.57971692, 0.49835897, 0.44303916, 0.40542177]

if __name__ == '__main__':
    plt.rcParams.update({'font.size': 12})
    plt.gcf().subplots_adjust(left=0.15) # Al usar savefig, para que quepa la etiqueta del eje Y
    
    plt.plot(dataset_sizes, coefs_greedy, 's--', label="Greedy Model")
    plt.plot(dataset_sizes, coefs_DQP, '^-.', label="DQP Model")
    
    plt.grid(True)
    plt.title("Greedy Model vs DQP Model")
    plt.ylabel("Action Coef.")
    plt.xlabel("Dataset Size (Number of levels)")
    plt.legend(fontsize=13)
    plt.ylim(0, 1)
    
    plt.savefig("grafica_Greedy_vs_DQP.png")
    
    plt.show()