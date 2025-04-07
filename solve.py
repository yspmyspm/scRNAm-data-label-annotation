import numpy as np
import multiprocessing as mp
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import cohen_kappa_score

from utils import read_data
from pre_process import pre_process
from train import multi_process_training, train_model_with_random_search,train_single_model

METHOD = 3
max_workers_train_data = 2
max_workers_full_data  = 2

def sample_from_distribution(param_distributions, n_samples=100):
	parameter_list = []
	for i in range(0,n_samples):
		id1 = np.random.randint(0, len(param_distributions['n_estimators']))
		id2 = np.random.randint(0, len(param_distributions['max_depth']))
		id3 = np.random.randint(0, len(param_distributions['learning_rate']))
		id4 = np.random.randint(0, len(param_distributions['subsample']))
		id5 = np.random.randint(0, len(param_distributions['colsample_bytree']))
		parameter_list.append({
			'n_estimators': param_distributions['n_estimators'][id1],
			'max_depth': param_distributions['max_depth'][id2],
			'learning_rate': param_distributions['learning_rate'][id3],
			'subsample':  param_distributions['subsample'][id4],
			'colsample_bytree': param_distributions['colsample_bytree'][id5]
		})
	return parameter_list

if __name__ == "__main__":
	mp.set_start_method('spawn',force=True)
	train_X, test_X, y, y_encode = read_data("train_adata.h5ad", "test_adata_noLabel.h5ad")
	train_X, test_X = pre_process(train_X, test_X, y_encode,METHOD)
	print("process data done")

	TRAIN_MODEL = True
	if TRAIN_MODEL:
		param_distributions = {
			'n_estimators': [400,500],
			'max_depth': [12,15,18,20,25,30],
			'learning_rate': [0.1,0.05,0.15,0.2],
			'subsample': [0.4,0.5,0.6,0.7,0.8],
			'colsample_bytree': [0.3,0.4,0.5,0.6,0.7,0.8]
		}
		parameter_list = sample_from_distribution(param_distributions, n_samples=40)
		X_train, X_val, y_train, y_val = train_test_split(train_X, y_encode, test_size=0.2, random_state=20041014,stratify=y_encode)
		multi_process_training(X_train, y_train, parameter_list, max_workers=max_workers_train_data,METHOD=METHOD)
		score_list = []
		for model_para in parameter_list:
			model_name = f"xgb_model_{model_para['n_estimators']}_{model_para['max_depth']}_{model_para['learning_rate']}_{model_para['subsample']}_{model_para['colsample_bytree']}_method_{METHOD}_{X_train.shape[0]}"
			model = xgb.XGBClassifier(
				objective='multi:softmax',
				num_class=len(np.unique(y)),
				tree_method='hist',
				device='cuda:0',     
				random_state=20041014,
				n_jobs=-1,
				verbosity=2
			)
			model.load_model(model_name+".json")
			y_pred = model.predict(X_val)
			kappa_score = cohen_kappa_score(y_val, y_pred)
			print(f"Model {model_name} Validation Kappa Score: {kappa_score}")
			score_list.append((kappa_score,model_para))
		
		# 只看第一关键字
		score_list.sort(key=lambda x: x[0], reverse=True)
		for i in range(len(score_list)):
			paramter = score_list[i][1]
			model_name = f"xgb_model_{paramter['n_estimators']}_{paramter['max_depth']}_{paramter['learning_rate']}_{paramter['subsample']}_{paramter['colsample_bytree']}_method_{METHOD}.json"
			print(f"Model {score_list[i][1]} Validation Kappa Score: {score_list[i][0]}")
		
		# 选 validation set 中最好的 10 个训练
		best_model_list = []
		for i in range(min(len(score_list),5)):
			best_model_list.append(score_list[i][1])
		
		multi_process_training(train_X, y_encode, best_model_list, max_workers=max_workers_full_data,METHOD=METHOD)

	else:
		# construct model list
		import os
		list_dir = os.listdir()
		model_list = []
		for name in list_dir:
			if name.endswith('971.json') and name.startswith('xgb_model'):
				model_list.append(name[:-5])
		
		# do testing with model_list
		from utils import calc_threshold, test
		for model_name in model_list: 
			print(model_name)
			model = xgb.XGBClassifier(
				objective='multi:softmax',
				num_class=len(np.unique(y)),
				tree_method='hist',
				device='cuda:0',     
				random_state=20041014,
				n_jobs=-1,
				verbosity=2
			)
			model.load_model(f"{model_name}.json")
			threshold = calc_threshold(model, train_X, y_encode)
			test(model,f"submission_{model_name}.csv",threshold,test_X,y)
		