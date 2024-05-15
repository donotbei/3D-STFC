import pandas as pd
import numpy as np
import os
from nilearn import image
#import tensorly as tl
#from tensorly.decomposition import tucker
#from sklearn.decomposition import NMF
#import matplotlib.pyplot as plt


#import x
#get file name
file_name = os.listdir(r'your/data/path/train')
#generate file path
path = []
img_arr1 = np.zeros((767,121,145,121))
for i in range(0,767):
    path.append(os.path.join(r'/your/data/path/train',file_name[i]))
    smooth_anat_img = image.smooth_img(path[i],fwhm=3)   #read image path
    img_arr1[i] = smooth_anat_img.get_fdata()   #get image data


#import y:
#read csv file
y_data = pd.read_csv(r'/your/data/path/train_767.CSV')
#get match name for data
file_name_match = []
for i in range(0,767):
    file_name_match.append(file_name[i][7:12])
#get match name for y
y_data_match = []
for j in range(0,767):
    y_data_match.append(str(y_data["ScanDir ID"][j]))
    if len(y_data_match[j])>5:    #split the string about name
        y_data_match[j] = y_data_match[j][2:7]
    else:
        y_data_match[j] = y_data_match[j]
#replace
y_data['ID'] = y_data_match
rep_y_data = y_data.drop(columns=['ScanDir ID'])
#match
train_y = []
ID_y = []
for j in range(0,767):
    for i in range(0,767):
        if file_name_match[j] == rep_y_data['ID'][i]:
            train_y.append(rep_y_data['DX'][i])
            ID_y.append(rep_y_data['ID'][i])
        else:
            continue
#look for the repetition str
#class Solution(object):
#    def findDuplicate(self, nums):
#        dic = dict()
#        for n in nums:
#            dic[n] = dic.get(n, 0) + 1
#            if dic[n] >= 2:
#               return n
#solution = Solution()
#solution.findDuplicate(ID_y) 
#get label
y1 = np.zeros(767)
for i in range(0,767):
    if float(train_y[i])>=1:
        y1[i]=1
    else:
        y1[i]=0

# reduce rank
#tk_factors = []
#tk_core = np.zeros((767,10,12,10))
##img_arr1_hat = np.zeros((767,121,145,121))
#for i in range(len(img_arr1)):
#    ten_img_arr = tl.tensor(img_arr1[i,:,:,:],dtype='float64')
#    core,each_factor = tucker(ten_img_arr, rank=[10,12,10], init='svd', tol=10e-6)
#    tk_factors.append(each_factor)
#    tk_core[i,:,:,:] = core
#    img_arr1_hat[i,:,:,:] = tl.tucker_to_tensor(core,each_factor)

# full cut
def data_p(p,q):
    if p==1:
        pic = np.zeros((767,145,121))
        for j in range(0, 767):
            pic[j, :] = img_arr1[j, q, :, :]
    elif p==2:
        pic = np.zeros((767,121,121))
        for j in range(0, 767):
            pic[j, :] = img_arr1[j, :, q, :]
    elif p==3:
        pic = np.zeros((767,121,145))
        for j in range(0, 767):
            pic[j, :] = img_arr1[j, :, :, q]
    else:
        print('error')
    return pic
