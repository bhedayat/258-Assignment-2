import numpy as np
import pandas as pd
from sklearn.utils import shuffle



class Level1:

	def __init__(self,train_users,sessions):
		#Drop train users that are not in sessions
		self.final_train_users = train_users[train_users['id'].drop_duplicates().isin(sessions['user_id'].drop_duplicates())]
		self.final_train_users = self.final_train_users.reset_index()
		self.actions = sessions['action'].dropna().drop_duplicates()
		self.sessions = sessions


	def action_bool(self,action):
		user_action = self.sessions[self.sessions['action'] == action]
		performed = self.final_train_users['id'].isin(user_action['user_id'])
		colname = 'b_' + action
		self.final_train_users[colname] = performed
		
	def extract(self,train_users,sessions):		
		#all actions
		for action in self.actions:
			self.action_bool(action)
		feats = map(lambda x: 'b_' + x,self.actions)
		X = self.final_train_users[feats + ['country_destination']]
		Xsh = shuffle(X,random_state = 42).reset_index()
		Xsh = Xsh.as_matrix()[:,1:]	
		ntrain = len(Xsh)

		Xtrain = Xsh[:ntrain,:-1]
		ytrain = np.ravel(Xsh[:ntrain,-1:] != 'NDF')

		return Xtrain, ytrain
