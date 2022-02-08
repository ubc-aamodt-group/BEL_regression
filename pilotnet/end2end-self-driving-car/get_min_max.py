from matplotlib import pyplot as plt
import numpy as np

with open('driving_dataset/test.csv', 'r') as infile:
    s_min = 0
    s_max = 0
    s_all = []

    for line in infile:
        s = line.split(',')[1]
        if float(s) > s_max:
            s_max = float(s)
        if float(s) < s_min:
            s_min = float(s)
        if abs(float(s)) > 160:
            s = 160
        s_all.append(abs(float(s)))

    print(s_min)
    print(s_max)
    print(np.sum(np.array(s_all) >= 159))
    plt.hist(s_all, bins=np.logspace(np.log10(0.1), np.log10(100)))

    plt.savefig('fig.png')
    
