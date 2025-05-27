import numpy as np
import multiprocessing as mp
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import cohen_kappa_score

from utils import read_data,sample_from_distribution
from pre_process import pre_process
from train import multi_process_training_shared_data,train_single_model
from train import multi_process_training


METHOD = 6
max_workers_train_data =  40
max_workers_full_data  = 10

if __name__ == "__main__":
	mp.set_start_method('spawn',force=True)
	train_X, test_X, y, y_encode = read_data("train_adata.h5ad", "test_adata_noLabel.h5ad")
	train_X, test_X = pre_process(train_X, test_X, y_encode,METHOD)
	print("process data done")

	param_distributions = {
		'n_estimators': [400,500,600,700,800,900,1000,2000,3000,5000],
		'max_depth': [10,12,15,18,20,25,30,40,50],
		'learning_rate': [0.1,0.05,0.15,0.2,0.01],
		'subsample': [0.4,0.5,0.6,0.7,0.8,0.3,0.9],
		'colsample_bytree': [0.3,0.4,0.5,0.6,0.7,0.8,0.9]		
	}
	X_train, X_val, y_train, y_val = train_test_split(train_X, y_encode, test_size=0.2, random_state=20041014,stratify=y_encode)
	
	parameter_list = sample_from_distribution(param_distributions, n_samples=200)
	multi_process_training(X_train, y_train, parameter_list, max_workers=max_workers_train_data,METHOD=METHOD)
	# parameter_list = []
	# import os
	# file_list = os.listdir()
	# for file in file_list:
	# 	if not(file.endswith(f"{METHOD}_776.json") and file.startswith("xgb_model")):
	# 		continue
	# 	if os.path.exists(file.replace("776","971")):
	# 		continue
	# 	n_estimators = int(file.split("_")[2])
	# 	max_depth = int(file.split("_")[3])
	# 	learning_rate = float(file.split("_")[4])
	# 	subsample = float(file.split("_")[5])
	# 	colsample_bytree = float(file.split("_")[6])
	# 	parameter = {
	# 		'n_estimators': n_estimators,
	# 		'max_depth': max_depth,
	# 		'learning_rate': learning_rate,
	# 		'subsample': subsample,
	# 		'colsample_bytree': colsample_bytree
	# 	}
	# 	parameter_list.append(parameter)
	
	

	score_list = []
	for model_para in parameter_list:
		try:
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
		except Exception as e:
			os.remove(model_name+".json")
	
	# 只看第一关键字
	score_list.sort(key=lambda x: x[0], reverse=True)
	for i in range(len(score_list)):
		parameter = score_list[i][1]
		model_name = f"xgb_model_{parameter['n_estimators']}_{parameter['max_depth']}_{parameter['learning_rate']}_{parameter['subsample']}_{parameter['colsample_bytree']}_method_{METHOD}.json"
		print(f"Model {score_list[i][1]} Validation Kappa Score: {score_list[i][0]}")
	
	# 选 validation set 中最好的 n 个训练
	n = 10
	best_model_list = []
	for i in range(min(len(score_list),n)):
		best_model_list.append(score_list[i][1])

	multi_process_training(train_X, y_encode, best_model_list, max_workers=max_workers_full_data,METHOD=METHOD)
		
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
	