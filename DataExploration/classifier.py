#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 10 10:06:58 2017

@author: dgriley
"""
#%%
import pandas as pd
import numpy as np
import seaborn as sns
from collections import Counter, defaultdict, OrderedDict
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier as rfc
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import RMSprop
from keras.regularizers import l2

def get_feat(column):
    feat_labels = column.drop_duplicates()
    one_hot = np.eye(len(feat_labels))
    one_hot_labels = {g:o for g,o in zip(feat_labels,one_hot)}
    return np.array([one_hot_labels[l] for l in column],ndmin = 2)
#import keras.sequential

#matplotlib inline

users_path = '../../Data/train_users_2.csv'
users = pd.read_csv(users_path)
sessions = pd.read_csv('../../Data/sessions.csv')

#Drop train users that are not in sessions
train_users = users[users['id'].drop_duplicates().isin(sessions['user_id'].drop_duplicates())]
#train_users = train_users[train_users['country_destination'] != 'NDF']
train_users = train_users.reset_index()
del train_users['index']
train_users.head()

#%%
tr_bias_feat = np.ones((len(train_users),1))
val_bias_feat = np.ones((len(val_users),1))

#%%
tr_month_feat = pd.to_numeric(train_users['date_account_created'].str[5:7])
tr_month_feat = np.array(tr_month_feat - tr_month_feat.mean(),ndmin = 2).transpose()

val_month_feat = pd.to_numeric(val_users['date_account_created'].str[5:7])
val_month_feat = np.array(val_month_feat - val_month_feat.mean(),ndmin = 2).transpose()
#%%
tr_gender_feat = get_feat(train_users['gender'])
val_gender_feat = get_feat(val_users['gender'])

#%%
tr_affiliate_feat = get_feat(train_users['affiliate_provider'])
val_affiliate_feat = get_feat(val_users['affiliate_provider'])

#%%
tr_device_feat = get_feat(train_users['first_device_type'])
val_device_feat = get_feat(val_users['first_device_type'])
#%%
actions = sessions['action'].dropna().drop_duplicates().reset_index(drop =True)
#del actions['index']

actionBin = np.zeros((len(train_users),len(actions)))
for actionNum in range(len(actions)):
    user_action = sessions[sessions['action'] == actions[actionNum]]
    actionBin[:,actionNum] = train_users['id'].isin(user_action['user_id'])    







#%%



tr_feature = np.hstack((tr_bias_feat,tr_month_feat,tr_gender_feat,tr_affiliate_feat,tr_device_feat,actionBin))
tr_target = train_users['country_destination'] == 'NDF'
tr_tot = np.hstack((tr_feature,np.array(tr_target,ndmin=2).transpose()))


#%%
val_acc = sum(np.abs(np.bitwise_xor(g,val_target)))/len(val_target)

#%%
def get_val_tr(data,section):
    if section == 0:
        ind1 = 0
        ind2 = int(np.floor(len(data)/10))
        tr = data[ind2:,:]
    else:
        ind1 = int(np.floor(section*len(data)/10))+1
        ind2 = int(np.floor(ind1 + len(data)/10))
        tr = np.concatenate((data[:ind1,:],data[ind2:,:]),axis = 0)
    return data[ind1:ind2,:],tr

def symmetrize_data(data):
    negs = data[data[:,-1] == 0,:]
    tr_data = np.concatenate((data,negs),axis = 0)
    return tr_data
    
    


#%%
def get_model():
    model = Sequential()
    model.add(Dense(400, input_dim=391, activation='relu',W_regularizer=l2(0.005)))
    model.add(Dense(400, activation = 'relu',W_regularizer=l2(0.005)))
    model.add(Dense(400,activation = 'relu',W_regularizer=l2(0.005)))
    model.add(Dense(1,activation = 'sigmoid'))
    model.compile(loss = 'binary_crossentropy',optimizer = 'adam',metrics = ['accuracy','fmeasure'])
    return model

#%%
roc_auc = []
truefalse = []
for i in range(10):
    val, tr = get_val_tr(tr_tot,i)
    tr = symmetrize_data(tr)
    model = get_model()
    history = model.fit(tr[:,:391],tr[:,391],batch_size = 2000,nb_epoch = 20,verbose = 1)
    y_score = model.predict_proba(val[:,:391])
    roc_auc.append(metrics.roc_auc_score(val[:,391],y_score))
    print('roc_auc = ' + str(roc_auc) + '\n')
    truefalse.append(metrics.roc_curve(val[:,391],y_score))
    



#%%
plt.rc('xtick', labelsize=10) 
plt.rc('ytick', labelsize=10) 

plt.hold
plt.grid()
for i in range(len(truefalse)):
    plt.plot(truefalse[i][0],truefalse[i][1],label='ROC={0:.5f}'.format(roc_auc[i]))

plt.plot([0,1],[0,1],linestyle='--',color='r')
    
plt.ylabel('tpr',fontsize=20)
plt.xlabel('fpr',fontsize=20)
plt.title('ROC Curves for K = 10 CV 3-layer NN Classifier',fontsize=15)
plt.legend(loc=4)
plt.show()


