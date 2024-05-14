import pandas as pd
import numpy as np
import os
from nilearn import image
#import x
#get file name
file_name = os.listdir(r'/Users/dd/PycharmProjects/ADHD/test')
file_name.remove('.DS_Store')
#generate file path
path = []
img_arr2 = np.zeros((171, 121, 145, 121))
for i in range(0, 171):
    path.append(os.path.join(r'/Users/dd/PycharmProjects/ADHD/test', file_name[i]))
    smooth_anat_img = image.smooth_img(path[i], fwhm=3)   #read image path
    img_arr2[i] = smooth_anat_img.get_fdata()   #get image data


#import y:
#read tsv file
y_data = pd.read_csv(r'/Users/dd/PycharmProjects/ADHD/test.file/allSubs_testSet_phenotypic_dx.csv')
#get match name for data
file_name_match = []
for i in range(0, 171):
    file_name_match.append(file_name[i][7:12])
#get match name for y
y_data_match = []
for j in range(0, 171):
    y_data_match.append(str(y_data["ID"][j]))
    if len(y_data_match[j]) > 5:    #split the string about name
        y_data_match[j] = y_data_match[j][2:7]
    else:
        y_data_match[j] = y_data_match[j]
#replace
y_data['ID'] = y_data_match
#rep_y_data = y_data.drop(columns=['ScanDir ID'])
#match
test_y = []
ID_y = []
for j in range(0,171):
    for i in range(0,171):
        if file_name_match[j] == y_data['ID'][i]:
            test_y.append(y_data['DX'][i])
            ID_y.append(y_data['ID'][i])
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

y2 = np.zeros(171)
for i in range(0,171):
    if float(test_y[i]) >= 1:
        y2[i] = 1
    else:
        y2[i] = 0


# full cut
def t_data_p(p, q):
    if p == 1:
        t_pic = np.zeros((171, 145, 121))
        for i in range(0,171):
            t_pic[i, :] = img_arr2[i, q, :, :]
    elif p==2:
        t_pic = np.zeros((171,121,121))
        for i in range(0,171):
            t_pic[i,:] = img_arr2[i,:,q,:]
    elif p==3:
        t_pic = np.zeros((171,121,145))
        for i in range(0,171):
            t_pic[i,:] = img_arr2[i,:,:,q]
    else:
        print('error')
    return t_pic
