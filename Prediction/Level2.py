import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from datetime import datetime, date
from collections import defaultdict

train_users_path = '../../Data/final_train_users.csv'
train_users = pd.read_csv(train_users_path)
train_users.head()

sessions_path = '../../Data/sessions.csv'
sessions = pd.read_csv(sessions_path)
sessions.head()

booked_train_users = train_users[train_users['country_destination'] != 'NDF']

from sklearn.preprocessing import OneHotEncoder
enc = OneHotEncoder()

def makeFeature(column):
    temp = (booked_train_users[column]).as_matrix()

    D = defaultdict(int)
    count = 1
    for i in np.unique(temp):
        D[i] = count
        count += 1

    newX = np.zeros((temp.shape))
    for i in range(temp.shape[0]):
        lang = temp[i]
        newX[i] = D[lang]
    newX = newX[:,np.newaxis]
    
    newX = enc.fit_transform(newX)
    newX = newX.toarray()
    
    return newX

columns = ['signup_method', 'signup_flow', 'affiliate_channel', 'affiliate_provider', 'first_affiliate_tracked', 'signup_app', 'first_device_type', 'first_browser']

X1 = makeFeature('gender')
for c in columns:
    tempX = makeFeature(c)
    X1 = np.concatenate((X1,tempX),axis=1)

def action_bool(action):
    user_action = sessions[sessions['action'] == action]
    performed = booked_train_users['id'].isin(user_action['user_id'])
    colname = 'b_' + action
    booked_train_users[colname] = performed

actions = sessions['action'].dropna().drop_duplicates()

for action in actions:
    action_bool(action)
    
booked_train_users.head()

col_name = ['b_' + action for action in actions]

actX = booked_train_users[col_name].as_matrix().astype(int)

X = (booked_train_users['language'] == 'en').as_matrix().astype(int)
X = X[:,np.newaxis]
labels = (booked_train_users['country_destination'] == 'US').as_matrix().astype(int)

Y = 2000 # dummy leap year to allow input X-02-29 (leap day)
seasons = [(0, (date(Y,  1,  1),  date(Y,  3, 20))),  #'winter'
           (1, (date(Y,  3, 21),  date(Y,  6, 20))),  #'spring'
           (2, (date(Y,  6, 21),  date(Y,  9, 22))),  #'summer'
           (3, (date(Y,  9, 23),  date(Y, 12, 20))),  #'autumn'
           (0, (date(Y, 12, 21),  date(Y, 12, 31)))]  #'winter'
def get_season(now):
    date_format = "%Y-%m-%d"
    d1 = now
    now = datetime.strptime(d1, date_format)
    
    if isinstance(now, datetime):
        now = now.date()
    now = now.replace(year=Y)
    return next(season for season, (start, end) in seasons
                if start <= now <= end)



enc = OneHotEncoder()
a = np.array(booked_train_users['date_account_created'].apply(get_season))
a = a[:,np.newaxis]
dateFeat = enc.fit_transform(a)
dateFeat = dateFeat.toarray()

def getTrainData(labels):
	bookX = np.concatenate((actX,X,dateFeat,X1),axis=1)
	
	return bookX,labels