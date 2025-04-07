from anndata import ImplicitModificationWarning
import warnings
warnings.filterwarnings("ignore", category=ImplicitModificationWarning)
import torch
import numpy as np

def read_data(train_path,test_path):
    import scanpy as sc
    import pandas as pd
    train_ad = sc.read_h5ad(train_path)
    test_ad = sc.read_h5ad(test_path)
    y = train_ad.obs['cell_type']
    y_encode = pd.Categorical(y).codes
    return train_ad.X,test_ad.X,train_ad.obs['cell_type'],y_encode


def std_visualization(train_X):
	import numpy as np
	std = np.nanstd(train_X, axis=0)
	import matplotlib.pyplot as plt
	plt.figure(figsize=(12, 6))
	plt.bar(range(len(std)), std)
	plt.xlabel('Feature Index')
	plt.ylabel('Standard Deviation')
	plt.title('Standard Deviation of Each Feature in train_X')
	plt.savefig('std_visualization.png')
	return 



def fetch_info(name):
	n_estimators,max_depth,learning_rate,subsample,colsample_bytree = name.split(".json")[0].split("_")[2:7]
	n_estimators = int(n_estimators)
	max_depth = int(max_depth)
	learning_rate = float(learning_rate)
	subsample = float(subsample)
	colsample_bytree = float(colsample_bytree)
	return n_estimators,max_depth,learning_rate,subsample,colsample_bytree

def test(model,save_path,threshold,test_X,y):
	import pandas as pd

	test_X = torch.tensor(test_X,device='cuda:0')
	ans = model.predict(test_X)
	proba = model.predict_proba(test_X)
	max_proba = np.max(proba, axis=1)

	ans = pd.Categorical(y).categories[ans]
	ans = pd.Series(ans)
	ans[max_proba < threshold] = 'Unknown'

	pd.DataFrame({'ID': [i for i in range(0, 283)], 'TARGET': ans}).to_csv(save_path, index=False)        
	return 

def calc_threshold(model, train_X, y_encode):
	y_proba = model.predict_proba(train_X)
	y_pred = model.predict(train_X)
	correct_proba = y_proba[y_pred == y_encode]
	max_proba = np.max(correct_proba, axis=1)
	threshold = np.percentile(max_proba, 0.1)
	print(threshold)
	return threshold

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