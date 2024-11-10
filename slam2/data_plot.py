import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def find_initial_pair_plot(matches):

    data = np.zeros_like(matches, dtype=np.int16)
    # Преобразуем данные в таблицу
    for i in range(matches.shape[0]):
        for j in range(matches.shape[1]):
            if i != j:
                data[i,j] = len(matches[i,j])

    # Plot heatmap
    plt.figure(figsize=(20, 20))
    sns.heatmap(data, annot=False, cmap="jet", square=True, vmax=1000)
    # sns.heatmap(data, annot=False, cmap="jet", square=True)
    plt.title("Keypoint correlation along the frames on the video")
    plt.xlabel("Frames")
    plt.ylabel("Frames")
    plt.savefig("./data_plot/find_initial_pair.png")
    plt.close()
