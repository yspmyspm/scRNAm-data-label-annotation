import cupy as cp
from cupy.cuda import memory
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import cohen_kappa_score,make_scorer
from sklearn.utils import compute_sample_weight
from concurrent.futures import ProcessPoolExecutor,ThreadPoolExecutor
import multiprocessing as mp
import numpy as np


def train_single_model(X, y,parameter,METHOD):
	
	n_estimators = parameter['n_estimators']
	max_depth = parameter['max_depth']
	learning_rate = parameter['learning_rate']
	subsample = parameter['subsample']
	colsample_bytree = parameter['colsample_bytree']
	validate = parameter['validate'] if 'validate' in parameter else False


	model = xgb.XGBClassifier(
		n_estimators=n_estimators,
		max_depth=max_depth,
		learning_rate=learning_rate,
		subsample=subsample,
		colsample_bytree=colsample_bytree,
		objective='multi:softmax',
		num_class=len(np.unique(y)),
		tree_method='hist',
		device = 'cuda', 
		random_state=20041014,
		n_jobs=-1,
		verbosity=2
	)
	if validate:
		X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=20041014)
		model.fit(X_train, y_train, verbose=0,sample_weight = compute_sample_weight(class_weight='balanced', y=y_train))
		y_pred = model.predict(X_val)
		kappa_score = cohen_kappa_score(y_val, y_pred)
		print("Validation Kappa Score:", kappa_score)
	
	model.fit(X,y,verbose=1,sample_weight = compute_sample_weight(class_weight='balanced', y=y))
	model.save_model(f'xgb_model_{n_estimators}_{max_depth}_{learning_rate}_{subsample}_{colsample_bytree}_method_{METHOD}_{X.shape[0]}.json')
	return 

def train_model_with_random_search(X, y,parameter_distributions,METHOD):
	# 超参数搜索：
	# 1. n_estimators: 树的数量
	# 2. max_depth: 树的最大深度
	# 3. learning_rate: 学习率
	# 4. subsample: 每棵树使用的样本比例
	# 5. colsample_bytree: 每棵树使用的特征比例
	model = xgb.XGBClassifier(
		objective='multi:softmax',
		num_class=len(np.unique(y)),
		tree_method='hist',
		device='cuda',
		random_state=20041014,
		n_jobs=-1,    
		verbosity=2
	)
	kappa_scorer = make_scorer(cohen_kappa_score)
	from sklearn.model_selection import RandomizedSearchCV, train_test_split
	random_search = RandomizedSearchCV(
		estimator=model,
		param_distributions=parameter_distributions,
		n_iter=100,
		scoring=kappa_scorer,
		cv = 4,    
		random_state=20041014,
		n_jobs=1,
		verbose=2
	)
	random_search.fit(X, y,sample_weight = compute_sample_weight(class_weight='balanced', y=y),verbose = 1)
	return random_search.best_params_, random_search.best_score_,random_search.best_estimator_

def train_shared_data_model(handle_bytes,shape, parameter,METHOD):
	import cupy as cp
	def open_ipc_handle(handle_bytes):
		import cupy as cp
		ptr = cp.cuda.runtime.ipcOpenMemHandle(handle_bytes)
		memptr = cp.cuda.MemoryPointer(cp.cuda.memory.UnownedMemory(ptr, 0, None), 0)
		return memptr

	train_X_memptr = open_ipc_handle(handle_bytes['X'])
	y_memptr = open_ipc_handle(handle_bytes['y'])
	weights_memptr = open_ipc_handle(handle_bytes['weights'])

	X = cp.ndarray(shape['X'], dtype=cp.float32, memptr=train_X_memptr)
	y = cp.ndarray(shape['y'], dtype=cp.int8, memptr=y_memptr)
	weights = cp.ndarray(shape['weights'], dtype=cp.float32, memptr=weights_memptr)

	dtrain = xgb.DMatrix(X, label=y, weight=weights)
	
	n_estimators, max_depth, learning_rate, subsample, colsample_bytree = parameter["n_estimators"], parameter["max_depth"], parameter["learning_rate"], parameter["subsample"], parameter["colsample_bytree"]
	num_classes = parameter["num_classes"]
	params = {
		"objective": "multi:softmax",
		"num_class": num_classes,
		"tree_method": "hist", 
		"device": "cuda",
		"max_depth": max_depth,
		"learning_rate": learning_rate,
		"subsample": subsample,
		"colsample_bytree": colsample_bytree,
		"random_state": 20041014,
		"nthread":-1,
	}
	model = xgb.train(
		params=params,
		dtrain=dtrain,
		num_boost_round=n_estimators,
		evals=[(dtrain, "train")],
		verbose_eval=100,
		early_stopping_rounds=20,
	)

	model.save_model(f'xgb_model_{n_estimators}_{max_depth}_{learning_rate}_{subsample}_{colsample_bytree}_method_{METHOD}_{X.shape[0]}.json')
	
	return

def multi_process_training_shared_data(X,y,parameters,max_workers = 3,METHOD = 0):
	import json
	print(json.dumps(parameters,indent=4))

	num_classes = len(np.unique(y))
	class_weights = compute_sample_weight(class_weight='balanced', y=y)
	weights = class_weights[y]
	
	X_gpu = cp.ascontiguousarray(cp.asarray(X))
	y_gpu = cp.ascontiguousarray(cp.asarray(y))
	weights_gpu = cp.ascontiguousarray(cp.asarray(weights.astype(np.float32)))
	shape = {
		'X': X.shape,
		'y': y.shape,
		'weights': weights.shape
	}

	def get_ipc_handle(arr):
		arr = cp.ascontiguousarray(arr)
		ipc_handle = cp.cuda.runtime.ipcGetMemHandle(arr.data.ptr)
		handle_bytes = bytes(ipc_handle)
		return handle_bytes

	handle_bytes = {
		'X': get_ipc_handle(X_gpu),
		'y': get_ipc_handle(y_gpu),
		'weights': get_ipc_handle(weights_gpu)
	}
	
	with ProcessPoolExecutor(max_workers=max_workers) as executor:
		futures = []
		for parameter in parameters:
			parameter['num_classes'] = num_classes
			futures.append(executor.submit(
				train_shared_data_model,
				handle_bytes,shape,parameter,
				METHOD
			))
		
		for future in futures:
			future.result()
	
	del X_gpu, y_gpu, weights_gpu
	del handle_bytes,shape
	return 

def train_model(X,y,weights,parameter,METHOD):
	
	import random
	import time
	if random.randint(0,3) >= 2:
		time.sleep(15)
	

	dtrain = xgb.DMatrix(X, label=y, weight=weights)
	
	n_estimators, max_depth, learning_rate, subsample, colsample_bytree = parameter["n_estimators"], parameter["max_depth"], parameter["learning_rate"], parameter["subsample"], parameter["colsample_bytree"]
	num_classes = parameter["num_classes"]
	params = {
		"objective": "multi:softmax",
		"num_class": num_classes,
		"tree_method": "hist", 
		"device": "cuda",
		"max_depth": max_depth,
		"learning_rate": learning_rate,
		"subsample": subsample,
		"colsample_bytree": colsample_bytree,
		"random_state": 20041014,
		"n_jobs":-1,
	}
	model = xgb.train(
		params=params,
		dtrain=dtrain,
		num_boost_round=n_estimators,
		evals=[(dtrain, "train")], 
		verbose_eval=10,
		early_stopping_rounds=20,
	)
	
	model.save_model(f'xgb_model_{n_estimators}_{max_depth}_{learning_rate}_{subsample}_{colsample_bytree}_method_{METHOD}_{X.shape[0]}.json')
	return

def multi_process_training(X,y,parameters,max_workers = 3,METHOD = 0):
	import json
	print(json.dumps(parameters,indent=4))

	num_classes = len(np.unique(y))
	class_weights = compute_sample_weight(class_weight='balanced', y=y)
	weights = class_weights[y]

	
	with ProcessPoolExecutor(max_workers=max_workers) as executor:
		futures = []
		for parameter in parameters:
			parameter['num_classes'] = num_classes
			futures.append(executor.submit(
				train_model,
				X,y,weights,parameter,
				METHOD
			))
		
		for future in futures:
			future.result()
	
	del X_gpu, y_gpu, weights_gpu
	del handle_bytes,shape
	return 
