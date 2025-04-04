import torch
import numpy as np

# 读数据
def read_data(train_path,test_path):
    import scanpy as sc
    import pandas as pd
    train_ad = sc.read_h5ad(train_path)
    test_ad = sc.read_h5ad(test_path)
    y = train_ad.obs['cell_type']
    y_encode = pd.Categorical(y).codes
    return train_ad.X,test_ad.X,train_ad.obs['cell_type'],y_encode


# 原数据每列标准差分布
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


# 从模型名称中提取参数
def fetch_info(name):
	n_estimators,max_depth,learning_rate,subsample,colsample_bytree = name.split(".json")[0].split("_")[2:7]
	n_estimators = int(n_estimators)
	max_depth = int(max_depth)
	learning_rate = float(learning_rate)
	subsample = float(subsample)
	colsample_bytree = float(colsample_bytree)
	return n_estimators,max_depth,learning_rate,subsample,colsample_bytree

# 在 test 上测试
def test(model,save_path,threshold,test_X,y):
	import pandas as pd
	test_X = test_X.cuda()
	ans = model.predict(test_X)
	proba = model.predict_proba(test_X)
	max_proba = np.max(proba, axis=1)

	ans = pd.Categorical(y).categories[ans]
	ans = pd.Series(ans)
	ans[max_proba < threshold] = 'Unknown'

	pd.DataFrame({'ID': [i for i in range(0, 283)], 'TARGET': ans}).to_csv(save_path, index=False)        
	return 

# 计算填充 unknown 的阈值
def calc_threshold(model, train_X, y_encode):
	y_proba = model.predict_proba(train_X)
	y_pred = model.predict(train_X)
	correct_proba = y_proba[y_pred == y_encode]
	max_proba = np.max(correct_proba, axis=1)
	# threshold = np.percentile(max_proba, 0.1)
	threshold = 0.2
	# print(threshold)
	return threshold