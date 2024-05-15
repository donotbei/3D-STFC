import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix,  recall_score
import itertools
import sub_dataset


'''
o_1 = pd.read_csv('/your/save_data/path/opt_Y1.csv', sep=' ', names=range(121)).astype(int)
o_2 = pd.read_csv('/your/save_data/path/opt_Y2.csv', sep=' ', names=range(145)).astype(int)
o_3 = pd.read_csv('/your/save_data/path/opt_Y3.csv', sep=' ', names=range(121)).astype(int)
o_1.to_csv('/your/save_data/path/opt_Y1.csv', index=False)
o_2.to_csv('/your/save_data/path/opt_Y2.csv', index=False)
o_3.to_csv('/your/save_data/path/opt_Y3.csv', index=False)
'''

t_Y1 = np.array(pd.read_csv('/your/save_data/path/opt_Y1.csv'))
t_Y2 = np.array(pd.read_csv('/your/save_data/path/opt_Y2.csv'))
t_Y3 = np.array(pd.read_csv('/your/save_data/path/opt_Y3.csv'))


'''
def row(p):
    ele_1, ele_2, ele_3 = [], [], []
    ele_1_1, ele_2_2, ele_3_3 = [], [], []
    for i in range(117):
        if t_Y1[p, i] == t_Y1[p, i + 1] == t_Y1[p, i + 2] ==t_Y1[p,i+3]==t_Y1[p,i+4]== 1:
            ele_1.append((i, i + 1, i + 2, i+3, i+4))  # ,i+3,i+4
        else:
            continue
        for j in range(len(ele_1)):
            ele_1_1 = list(set(ele_1_1) | set(ele_1[j]))

    for i in range(141):
        if t_Y2[p, i] == t_Y2[p, i + 1] == t_Y2[p, i + 2]==t_Y2[p,i+3]==t_Y2[p,i+4] == 1:
            ele_2.append((i, i + 1, i + 2, i+3, i+4))  # ,j+3,j+4
        else:
            continue
        for j in range(len(ele_2)):
            ele_2_2 = list(set(ele_2_2) | set(ele_2[j]))

    for i in range(117):
        if t_Y3[p, i] == t_Y3[p, i + 1] == t_Y3[p, i + 2] == t_Y3[p,i+3]== t_Y3[p,i+4] == 1:
            ele_3.append((i, i + 1, i + 2, i+3, i+4))  # ,k+3,k+4
        else:
            continue
        for k in range(len(ele_3)):
            ele_3_3 = list(set(ele_3_3) | set(ele_3[k]))

    return ele_1_1, ele_2_2, ele_3_3


def comb(p):
    N=[]
    for i in row(p)[0]:
        for j in row(p)[1]:
            for k in row(p)[2]:
                N.append([i,j,k])
    return N


plot_data = []

for i in range(0,171):
    plot_data.append(comb(i))


t_pre = np.zeros(171)
for i in range(171):
    if len(plot_data[i])==0:
        t_pre[i]=0
    else:
        t_pre[i]=1

print(confusion_matrix(test.y2, t_pre))
print(recall_score(test.y2, t_pre))
'''


def Decision_3d(Y1, Y2, Y3, length_2):
    decision_matrix = np.zeros([length_2, 121, 145, 121])
    matrix_1 = np.zeros([121, 145, 121])
    matrix_2 = np.zeros([121, 145, 121])
    matrix_3 = np.zeros([121, 145, 121])
    for n in range(length_2):
        for i in range(121):
            if Y1[n, i] == 1:
                matrix_1[i, :, :] = 1
            else:
                matrix_1[i, :, :] = 0
        for i in range(145):
            if Y2[n, i] == 1:
                matrix_2[:, i, :] = 1
            else:
                matrix_2[:, i, :] = 0
        for i in range(121):
            if Y3[n, i] == 1:
                matrix_3[:, :, i] = 1
            else:
                matrix_3[:, :, i] = 0

        decision_matrix[n] = matrix_1 * matrix_2 * matrix_3

    return decision_matrix

def Judge(X, k=3):
    count = 0
    for i in range(122 - k):
        for j in range(146 - k):
            for l in range(122 - k):
                if X[i, j, l] == 1:
                    for a, b, c in itertools.product(range(k), range(k), range(k)):
                        if X[i+a, j+b, l+c] == 1:
                            count += 1
                    if count == k*k*k:
                        return 1
                    else:
                        count = 0
                        continue
    return 0


y_2 = sub_dataset.y_2
length_2 = sub_dataset.length_2
decision_matrix_3d = Decision_3d(t_Y1, t_Y2, t_Y3, length_2)
t_pre = np.zeros(length_2)
for i in range(length_2):
    t_pre[i] = Judge(decision_matrix_3d[i], 8)

acc = (confusion_matrix(y_2, t_pre)[0,0]+confusion_matrix(y_2, t_pre)[1,1])/length_2
print(confusion_matrix(y_2, t_pre))
print(acc)
