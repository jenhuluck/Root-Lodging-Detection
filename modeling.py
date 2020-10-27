# -*- coding: utf-8 -*-
"""
Created on Mon Sep 28 14:34:59 2020

@author: Jen Hu
"""
#1. Augmentation of pictures
#2. Try other strategies with no line fitting
import numpy as np
from skimage.io import imread
import matplotlib.pyplot as plt
from PIL import Image
import math
from sklearn.linear_model import LinearRegression
#import driver as dr
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import h5py
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import svm

#Divide the picture to number of segments with the same width and count number of points in each segment.
#Return a count array
def count_array(image,seg_num):
    col = len(image[0])
    row = len(image)
   # print(row)
    #print(col)
    seg_width = row/seg_num
    #print(seg_width)
    res = [0 for i in range(seg_num)] 
    count = 0
    for i in range(row):
        for j in range(col):
            if image[i][j] == 255:  
                #print(i,j)
                n = int(i/seg_width)
                #print(n)
                res[n] += 1
                count += 1
    #density = [i/count for i in res]                      
    return res;

# Merge two arrays and return a density array
def augment(count_arr1, count_arr2):  
    #print(count_arr1)
    #print(count_arr2)
    merge = count_arr1 + count_arr2
    #print(merge)
    density = [i/len(merge) for i in merge]
    #print(density)
    return density
    
 
#If row = 4, directly save density data and labels into dataset
#If row = 2, combine two pictures to make a new sample
#In this research, only consider the pictures of 4 rows.
def prepare_data(seg_4num):
    labels_path = 'labels.csv'
    data = pd.read_csv(labels_path, usecols = ['No','RowNum','Lodged'], dtype={'No':'str','RowNum':'Int32', 'Lodged':'Int32'})
    df_2rows = data.where(data['RowNum'] == 2).dropna()
    df_4rows = data.where(data['RowNum'] == 4).dropna()
    X = []
    Y = []
    folder = 'processed_images/'
   
    for index, row in df_4rows.iterrows():
        pic_path = folder + row['No']+'.png'
        img = imread(pic_path)
        count = count_array(img,seg_4num)
        X.append([i/sum(count) for i in count]) # add density array to X
        Y.append(row['Lodged'])
    
    seg_2num = int(seg_4num/2)
    for index_i, row1 in df_2rows.iterrows():
        for index_j, row2 in df_2rows.iterrows():
            if index_j >= index_i:
                #print(index_i,index_j)
                pic_path1 = folder + row1['No']+'.png'
                pic_path2 = folder + row2['No']+'.png'
                img1 = imread(pic_path1)
                img2 = imread(pic_path2)
                #print(pic_path1)
                #print(pic_path2)
                count1 = count_array(img1,seg_2num)
                #print(count1)
                count2 = count_array(img2,seg_2num)
                merge = augment(count1, count2)
                X.append(merge)
                
                label1 = row1['Lodged']
                label2 = row2['Lodged']
                label_for_sample = label1 or label2 # if any of the photo is lodged, label it as lodged.
                Y.append(label_for_sample)  
    return {'input':X, 'labels':Y}   

         
def gbx_model(X,y):
    #test for the learning rate
    #0.75 showed up with a higher accuracy
    #Accuracy score (training): 0.979
    #Accuracy score (validation): 0.953
    '''
    for lr in [0.05, 0.1, 0.25, 0.5, 0.75, 1]:
        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 1)
        gb = GradientBoostingClassifier(n_estimators=150, learning_rate = lr, max_features=2, max_depth = 2, random_state = 1)
        gb.fit(x_train, y_train)
        print("Accuracy score (training): {0:.3f}".format(gb.score(x_train, y_train)))
        print("Accuracy score (validation): {0:.3f}".format(gb.score(x_test, y_test)))
        print()
    '''   
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 1)
    gb = GradientBoostingClassifier(n_estimators=150, learning_rate = 0.75, max_features=2, max_depth = 2, random_state = 0)
    gb.fit(x_train, y_train)
    predictions = gb.predict(x_test)
    
    
    print("Accuracy of gradient boosting tree:",metrics.accuracy_score(y_test, predictions))
    '''
    print("Confusion Matrix of gradient boosting tree:")
    print(confusion_matrix(y_test, predictions))
    print()
    
    print("Classification Report of gradient boosting tree:")
    print(classification_report(y_test, predictions))
    
    y_scores_gb = gb.decision_function(x_test)
    fpr_gb, tpr_gb, _ = roc_curve(y_test, y_scores_gb)
    roc_auc_gb = auc(fpr_gb, tpr_gb)  
    print("Area under ROC curve = {:0.2f}".format(roc_auc_gb))
    '''
    print('----------------------------------------------')
    
def decision_tree(X,y):
    dt = DecisionTreeClassifier()    
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 1)
    dt.fit(x_train,y_train)
    predictions = dt.predict(x_test)
    
    print("Accuracy of decision tree:",metrics.accuracy_score(y_test, predictions))
    '''
    print("Confusion Matrix of decision tree:")
    print(confusion_matrix(y_test, predictions))
    print()
    
    print("Classification Report of decision tree")
    print(classification_report(y_test, predictions))
    '''
    print('----------------------------------------------')
    
    
def random_forest(X,y):
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 1)
    rf = RandomForestClassifier(n_estimators=150)
    rf.fit(x_train, y_train)
    predictions = rf.predict(x_test)
    
    print("Accuracy of random forest:",metrics.accuracy_score(y_test, predictions))
    '''
    print("Confusion Matrix of random forest:")
    print(confusion_matrix(y_test, predictions))
    print()
    
    print("Classification Report of random forest")
    print(classification_report(y_test, predictions))
    '''
    print('----------------------------------------------')
    
    
def logistic_regression(X,y):
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 1)
    logreg = LogisticRegression()
    logreg.fit(x_train, y_train)
    predictions = logreg.predict(x_test)
    
    print("Accuracy of logistic regression:",metrics.accuracy_score(y_test, predictions))
    '''
    print("Confusion Matrix of logistic regression:")
    print(confusion_matrix(y_test, predictions))
    print()
    
    print("Classification Report of logistic regression:")
    print(classification_report(y_test, predictions))
    '''
    print('----------------------------------------------')
    
def knn(X,y):
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 1)
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(x_train, y_train)
    predictions = knn.predict(x_test)
    
    print("Accuracy of k nearest neighbors:",metrics.accuracy_score(y_test, predictions))
    '''
    print("Confusion Matrix of k nearest neighbors:")
    print(confusion_matrix(y_test, predictions))
    print()
    
    print("Classification Report of k nearest neighbors:")
    print(classification_report(y_test, predictions))
    '''
    print('----------------------------------------------')

def svm_model(X,y):
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 1)
    svm_model = svm.SVC(kernel = 'poly') # can be'linear', 'poly', 'rbf'
    # Polynomial kernel showed up the highest accuracy
    svm_model.fit(x_train, y_train)
    predictions = svm_model.predict(x_test)
    print("Accuracy of svm:",metrics.accuracy_score(y_test, predictions))
    '''
    print("Confusion Matrix of svm:")
    print(confusion_matrix(y_test, predictions))
    print()
    
    print("Classification Report of svm:")
    print(classification_report(y_test, predictions))
    '''
    print('----------------------------------------------')
    
    
def naitive_bayes(X,y):
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 1)   
    nb = GaussianNB()  
    nb.fit(x_train, y_train)
    predictions = nb.predict(x_test)
    
    print("Accuracy of naitive bayes:",metrics.accuracy_score(y_test, predictions))
    '''
    print("Confusion Matrix of naitive bayes:")
    print(confusion_matrix(y_test, predictions))
    print()
    
    print("Classification Report of naitive bayes:")
    print(classification_report(y_test, predictions))
    '''
    print('----------------------------------------------')
    
    
def read_h5(path):
    f = h5py.File(path, 'r')
    X = f.get('input')
    Y = f.get('labels')
    #self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state = 1)
    X = np.array(X)
    Y = np.array(Y)
    #print(X.shape)
    #print(Y.shape)
    return [X,Y]

def save_data(data,file_path):
    f = h5py.File(file_path,'w')
    for key in data:
        print(key)
        f.create_dataset(key,data = data[key])       
    f.close()
    print('Done.') 
    
if __name__ == '__main__':   
    save_data_flag = False #Save data to h5 formate. If it has already been generated, turn it to false.
    seg_4num = [10,20,30,40,50,60,70,80,90,100] # Test the number of segments for 4 rows pictures
    for num in seg_4num:
        file_path = 'density_data_'+str(num)+'.h5'
        if save_data_flag:
            data = prepare_data(num)           
            save_data(data, file_path)
        
        dataset = read_h5(file_path)
        
        #classifiers
        print("Segment number is: "+str(num))
        
        logistic_regression(dataset[0],dataset[1])
        naitive_bayes(dataset[0],dataset[1])
        knn(dataset[0],dataset[1])
        decision_tree(dataset[0],dataset[1])
        
        svm_model(dataset[0],dataset[1]) # When seg_4num == 10, the program died. 
        
        gbx_model(dataset[0],dataset[1]) 
        
        print('*****************************')
        
    
    
    
    
    
    