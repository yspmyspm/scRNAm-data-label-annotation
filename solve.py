import numpy as np
import warnings
from anndata import ImplicitModificationWarning
warnings.filterwarnings("ignore", category=ImplicitModificationWarning)
np.random.seed(20041014)

import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import cohen_kappa_score, make_scorer
from sklearn.utils import compute_sample_weight
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp

from utils import std_visualization
from utils import fill_method_2,fill_method_3,fill_method_4
from utils import read_data

import cupy as cp
from cupy.cuda import memory

import torch
import torch

import torch

def random_shuffle_and_average_pooling(train_X, test_X, pool_size):
	"""
	对 train_X 和 test_X 的列进行相同的随机排列，然后分别应用平均池化。
	
	参数:
		train_X (torch.Tensor): 训练集输入矩阵，形状为 (n_samples_train, n_features)
		test_X (torch.Tensor): 测试集输入矩阵，形状为 (n_samples_test, n_features)
		pool_size (int): 平均池化的窗口大小
	
	返回:
		tuple: (train_result, test_result)
			- train_result: 训练集平均池化后的结果，形状为 (n_samples_train, n_pooled_features)
			- test_result: 测试集平均池化后的结果，形状为 (n_samples_test, n_pooled_features)
	"""
	# 确保 train_X 和 test_X 的特征维度相同
	assert train_X.shape[1] == test_X.shape[1], "train_X 和 test_X 的列数必须相同！"
	
	n_samples_train, n_features = train_X.shape
	n_samples_test = test_X.shape[0]

	# Step 1: 生成共享的随机排列索引
	shuffled_indices = torch.argsort(torch.rand(1, n_features), dim=1).expand(n_samples_train + n_samples_test, -1)
	shuffled_indices = shuffled_indices.to(train_X.device)
	# Step 2: 使用共享索引对 train_X 和 test_X 进行随机打乱
	shuffled_train_X = torch.gather(train_X, dim=1, index=shuffled_indices[:n_samples_train])
	shuffled_test_X = torch.gather(test_X, dim=1, index=shuffled_indices[n_samples_train:])

	# Step 3: 应用平均池化
	def apply_average_pooling(X, pool_size):
		# 将特征维度调整为可以被池化窗口整除
		n_features = X.shape[1]
		n_pooled_features = n_features // pool_size
		pooled_X = X[:, :n_pooled_features * pool_size].reshape(-1, n_pooled_features, pool_size)
		# 计算每个窗口的平均值
		return pooled_X.mean(dim=2)

	train_pooled = apply_average_pooling(shuffled_train_X, pool_size)
	test_pooled = apply_average_pooling(shuffled_test_X, pool_size)

	return train_pooled, test_pooled

def random_projection(data: torch.Tensor, k: int, device: torch.device = None) -> torch.Tensor:
	if device is None:
		device = data.device
	n, d = data.shape
	R = torch.randn(d, k, device=device)
	R = R / torch.norm(R, dim=0, keepdim=True)
	reduced_data = torch.matmul(data, R)
	return reduced_data

def pca_transform(train_X, test_X, n_components):
	combined_data = torch.cat((train_X, test_X), dim=0)
	mean = combined_data.mean(dim=0, keepdim=True)
	centered_data = combined_data - mean
	U, S, Vt = torch.linalg.svd(centered_data, full_matrices=False)
	top_eigenvectors = Vt[:n_components, :].t()  # 取前n_components个主成分
	reduced_data = centered_data @ top_eigenvectors
	reduced_train = reduced_data[:train_X.size(0)]
	reduced_test = reduced_data[train_X.size(0):]
	return reduced_train, reduced_test

METHOD = 4

max_workers_train_data = 40
max_workers_full_data  = 30

def pre_process(train_X, test_X, train_y):
	import torch
	train_X = torch.tensor(train_X, dtype=torch.float32).cuda()
	test_X = torch.tensor(test_X, dtype=torch.float32).cuda()
	train_y = torch.tensor(train_y, dtype=torch.int64).cuda()


	# 删掉 train_X 中全是 NaN 的列
	mask = torch.all(torch.isnan(train_X), dim=0)
	train_X = train_X[:, ~mask]
	test_X = test_X[:, ~mask]
	if METHOD == 3:
		train_X,test_X = fill_method_3(train_X,test_X,train_y)
		train_X = random_projection(train_X, 10000)
		test_X = random_projection(test_X, 10000)
	elif METHOD ==2:
		train_X,test_X = fill_method_2(train_X,test_X,train_y)
	elif METHOD == 4:
		train_X,test_X = fill_method_3(train_X,test_X,train_y)
		train_X,test_X = pca_transform(train_X,test_X, 10000)
	elif METHOD == 5:
		train_X,test_X = fill_method_3(train_X,test_X,train_y)
		train_X,test_X = random_shuffle_and_average_pooling(train_X,test_X, 10)
	

	train_X = train_X.cpu().numpy()
	test_X = test_X.cpu().numpy()
	train_y = train_y.cpu().numpy()
	torch.cuda.empty_cache()
	return train_X, test_X

def train_single_model(X, y, n_estimators, max_depth, learning_rate, subsample, colsample_bytree,validate = False):
	print("Training model with parameters:")
	print("n_estimators:", n_estimators)
	print("max_depth:", max_depth)
	print("learning_rate:", learning_rate)
	print("subsample:", subsample)
	print("colsample_bytree:", colsample_bytree)
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
		n_jobs=3,
		verbosity=2
	)
	if validate:
		X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=20041014)
		model.fit(X_train, y_train, verbose=0,sample_weight = compute_sample_weight(class_weight='balanced', y=y_train))
		y_pred = model.predict(X_val)
		kappa_score = cohen_kappa_score(y_val, y_pred)
		print("Validation Kappa Score:", kappa_score)
	model.fit(X,y,verbose=0,sample_weight = compute_sample_weight(class_weight='balanced', y=y))
	model.save_model(f'xgb_model_{n_estimators}_{max_depth}_{learning_rate}_{subsample}_{colsample_bytree}_method_{METHOD}.json')
	return 

def train_model_with_random_search(X, y,parameter_distributions):
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
	random_search.fit(X, y,sample_weight = compute_sample_weight(class_weight='balanced', y=y))
	return random_search.best_params_, random_search.best_score_,random_search.best_estimator_

def train_shared_data_model(handle_bytes,shape, parameter):
	import cupy as cp
	def open_ipc_handle(handle_bytes):
		import cupy as cp
		ptr = cp.cuda.runtime.ipcOpenMemHandle(handle_bytes)
		memptr = cp.cuda.MemoryPointer(cp.cuda.memory.UnownedMemory(ptr, 0, None), 0)
		return memptr

	train_X_memptr = open_ipc_handle(handle_bytes['train_X'])
	y_memptr = open_ipc_handle(handle_bytes['y_encode'])
	weights_memptr = open_ipc_handle(handle_bytes['weights'])

	train_X = cp.ndarray(shape['train_X'], dtype=cp.float32, memptr=train_X_memptr)
	y_encode = cp.ndarray(shape['y_encode'], dtype=cp.int8, memptr=y_memptr)
	weights = cp.ndarray(shape['weights'], dtype=cp.float32, memptr=weights_memptr)

	dtrain = xgb.DMatrix(train_X, label=y_encode, weight=weights)
	
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
		"nthread":3,
	}
	model = xgb.train(
		params=params,
		dtrain=dtrain,
		num_boost_round=n_estimators,
		evals=[(dtrain, "train")],
		verbose_eval=100,
		early_stopping_rounds=20,
	)

	model.save_model(f'xgb_model_{n_estimators}_{max_depth}_{learning_rate}_{subsample}_{colsample_bytree}_method_{METHOD}_{train_X.shape[0]}.json')
	
	return

def multi_process_training(train_X,y_encode,parameters,max_workers = 3):
	import json
	print(json.dumps(parameters,indent=4))

	num_classes = len(np.unique(y_encode))
	class_weights = compute_sample_weight(class_weight='balanced', y=y_encode)
	weights = class_weights[y_encode]
	
	train_X_gpu = cp.ascontiguousarray(cp.asarray(train_X))
	y_encode_gpu = cp.ascontiguousarray(cp.asarray(y_encode))
	weights_gpu = cp.ascontiguousarray(cp.asarray(weights.astype(np.float32)))
	shape = {
		'train_X': train_X.shape,
		'y_encode': y_encode.shape,
		'weights': weights.shape
	}

	def get_ipc_handle(arr):
		arr = cp.ascontiguousarray(arr)
		ipc_handle = cp.cuda.runtime.ipcGetMemHandle(arr.data.ptr)
		handle_bytes = bytes(ipc_handle)
		return handle_bytes

	handle_bytes = {
		'train_X': get_ipc_handle(train_X_gpu),
		'y_encode': get_ipc_handle(y_encode_gpu),
		'weights': get_ipc_handle(weights_gpu)
	}
	
	with ProcessPoolExecutor(max_workers=max_workers) as executor:
		futures = []
		for parameter in parameters:
			parameter['num_classes'] = num_classes
			futures.append(executor.submit(
				train_shared_data_model,
				handle_bytes,shape,parameter
			))
		
		for future in futures:
			future.result()
	
	del train_X_gpu, y_encode_gpu, weights_gpu
	del handle_bytes,shape
	return 

if __name__ == "__main__":
	mp.set_start_method('spawn',force=True)
	train_X, test_X, y, y_encode = read_data("train_adata.h5ad", "test_adata_noLabel.h5ad")
	train_X, test_X = pre_process(train_X, test_X, y_encode)
	print("process data done")
	TRAIN_MODEL = True
	if TRAIN_MODEL:
		import os
		param_distributions = {
			'n_estimators': [1500,2000,3000,4000],
			'max_depth': [12,15,18,20,25,30],
			'learning_rate': [0.1,0.05,0.15,0.2],
			'subsample': [0.4,0.5,0.6,0.7,0.8],
			'colsample_bytree': [0.3,0.4,0.5,0.6,0.7,0.8]
		}
		parameter_list = []
		for i in range(0,100):
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
		
		X_train, X_val, y_train, y_val = train_test_split(train_X, y_encode, test_size=0.2, random_state=20041014,stratify=y_encode)
		multi_process_training(X_train, y_train, parameter_list, max_workers=max_workers_train_data)
		score_list = []
		for model_para in parameter_list:
			model_name = f"xgb_model_{model_para['n_estimators']}_{model_para['max_depth']}_{model_para['learning_rate']}_{model_para['subsample']}_{model_para['colsample_bytree']}_method_{METHOD}_{X_train.shape[0]}.json"
			model = xgb.XGBClassifier(
				objective='multi:softmax',
				num_class=len(np.unique(y)),
				tree_method='hist',
				device='cuda:0',     
				random_state=20041014,
				n_jobs=-1,
				verbosity=2
			)
			model.load_model(model_name)
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
		for i in range(min(len(score_list),10)):
			best_model_list.append(score_list[i][1])
		
		multi_process_training(train_X, y_encode, best_model_list, max_workers=max_workers_full_data)
	else:
		# construct model list
		import os
		list_dir = os.listdir()
		model_list = []
		for name in list_dir:
			if name.endswith('.json') and name.startswith('xgb_model'):
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
		