'''Assess the positional effect by mutating each base to the other three bases in sequential order. 
The baseline consists of 1000 randomly generated 23-nt sequences.
'''


__author__ = 'Chao Wu'


import numpy as np
import pandas as pd
from copy import deepcopy
import tensorflow as tf
from joblib import load
import matplotlib.pyplot as plt


MODEL_DIR = 'hybrid_seq_only'
TARGET_SCALER_FILE = 'target.scaler'
N_SAMPLES = 1000   


def main():
    
    model = tf.keras.saving.load_model(MODEL_DIR)   
    target_scaler = load(TARGET_SCALER_FILE)

    np.random.seed(0)
    X_seq = np.random.choice(list('ATCG'), (N_SAMPLES, 21))
    X_seq = np.concatenate((X_seq, np.full((N_SAMPLES, 2), 'G')), axis = 1)
    X_seq_lst = X_seq.tolist()
    
    vectorizer = tf.keras.layers.TextVectorization(
        split = None, 
        vocabulary = ['a', 't', 'c', 'g']   
    )
    Y_pred = model.predict(vectorizer(X_seq).numpy())
    Y_pred_all = np.concatenate((Y_pred, Y_pred, Y_pred))
    base_fitness = target_scaler.inverse_transform(Y_pred_all)
    
    fitness_diff = []
    for i_pos in range(21):   
        
        X_seq_mut_lsts = [
            deepcopy(X_seq_lst), 
            deepcopy(X_seq_lst), 
            deepcopy(X_seq_lst)
        ]
        for i_sample in range(N_SAMPLES):
            idx = 0
            for base in list('ATCG'):
                if X_seq_lst[i_sample][i_pos] != base:
                    X_seq_mut_lsts[idx][i_sample][i_pos] = base
                    idx += 1
                else:
                    continue
        X_seq_mut_all = np.concatenate(X_seq_mut_lsts)   
        
        Y_mut_pred_all = model.predict(vectorizer(X_seq_mut_all).numpy())
        mut_fitness = target_scaler.inverse_transform(Y_mut_pred_all)
        fitness_diff.append(base_fitness - mut_fitness)

    fitness_diff = pd.DataFrame(np.concatenate(fitness_diff, axis = 1))
    fitness_diff.to_excel('fitness_diff.xlsx', header = True, index = False)

    fig, ax = plt.subplots(figsize = (9, 4))

    boxplot = ax.boxplot(fitness_diff, patch_artist = True, showfliers = False)
    for i, box in enumerate(boxplot['boxes']):
        lower_whisker = boxplot['whiskers'][2*i].get_ydata()[1]
        upper_whisker = boxplot['whiskers'][2*i+1].get_ydata()[1]
        if lower_whisker < -1.0 and upper_whisker > 1.0:
            box.set_facecolor('green')
    
    ax.set_xlabel('Position', fontsize = 20)
    ax.set_ylabel('Fitness difference', fontsize = 20)

    xmin, xmax = ax.get_xlim()
    ax.hlines([1, -1], xmin = xmin, xmax = xmax, linestyles = '--', colors = 'gray')
    
    fig.savefig('positional_effect.jpg', bbox_inches = 'tight')    




if __name__ == '__main__':

    main()