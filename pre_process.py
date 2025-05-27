from projection import random_projection, pca_transform, random_shuffle_and_average_pooling
import torch
import numpy as np

def fill_nan_with_independent_random(data, mean, std,flag = True):
	# method_1
	nan_mask = torch.isnan(data)
	random_values = torch.randn_like(data)
	if not flag:
		random_values = random_values * std + mean
	else:
		random_values = mean
	filled_data = torch.where(nan_mask, random_values, data)
	return filled_data


data_Mean = 0.19
data_Std = 0.06710541457415352


def fill_method_2(train_X,test_X,train_y):
	data = torch.concat([train_X,test_X],dim=0)
	median_values = torch.nanmedian(data, dim=0).values
	median_tensor = median_values.unsqueeze(0).expand_as(data)
	filled_data = torch.where(torch.isnan(data), median_tensor, data)
	train_X = filled_data[:train_X.shape[0]].clone()
	test_X = filled_data[train_X.shape[0]:].clone()
	del median_tensor,median_values,filled_data
	return train_X, test_X


def fill_method_3(train_X,test_X,train_y):
	data = torch.concat([train_X,test_X],dim=0)
	median_values = torch.nanmedian(data, dim=0).values
	median_tensor = median_values.unsqueeze(0).expand_as(data)
	std = torch.normal(data_Mean, data_Std, size=data.shape,device=data.device)
	std = torch.abs(std)
	fill_with_noise = torch.normal(mean=median_tensor, std=std)
	fill_with_noise = torch.clamp(fill_with_noise, min=0, max=1)
	filled_data = torch.where(torch.isnan(data), fill_with_noise, data)
	train_X = filled_data[:train_X.shape[0]].clone()
	test_X = filled_data[train_X.shape[0]:].clone()
	del median_tensor,median_values,filled_data,fill_with_noise,std
	return train_X, test_X

def fill_method_4(train_X,test_X,train_y):

	for class_label in torch.unique(train_y):
		class_mask = (train_y == class_label)
		class_X = train_X[class_mask]
		
		medians = torch.nanmedian(class_X, dim=0).values
		medians = medians.unsqueeze(0)
		
		# 计算每列的标准差 std ~ N(data_Mean, data_Std)，并确保 std > 0
		stds = torch.normal(data_Mean, data_Std, size=(1,class_X.shape[1]), device=train_X.device)
		stds = torch.abs(stds) 
		nan_mask = torch.isnan(class_X)

		random_fill = torch.normal(
			mean=medians.expand_as(class_X),
			std=stds.expand_as(class_X),
		).to(class_X.device)
		class_X[nan_mask] = random_fill[nan_mask]
		train_X[class_mask] = class_X

	
	combined_X = torch.cat([train_X, test_X], dim=0)
	nan_mask = torch.isnan(combined_X)
	
	medians = torch.nanmedian(combined_X, dim=0).values
	medians = medians.unsqueeze(0)

	# 生成标准差
	stds = torch.normal(data_Mean, data_Std, size=(1,combined_X.shape[1]), device=train_X.device)
	stds = torch.abs(stds)
	
	random_fill = torch.normal(
		mean=medians.expand_as(combined_X),
		std = stds.expand_as(combined_X),
	).to(combined_X.device)
	
	combined_X[nan_mask] = random_fill[nan_mask]
	torch.clamp(combined_X, min=0, max=1, out=combined_X)  # 限制范围在 [0, 1]
	train_X = combined_X[:train_X.shape[0]]
	test_X = combined_X[train_X.shape[0]:]
	return train_X, test_X


def pre_process(train_X, test_X, train_y,METHOD):
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
	elif METHOD ==2:
		train_X,test_X = fill_method_2(train_X,test_X,train_y)
	elif METHOD == 4:
		train_X,test_X = fill_method_4(train_X,test_X,train_y)
		train_X,test_X = pca_transform(train_X,test_X, 10000)
	elif METHOD == 5:
		train_X,test_X = fill_method_3(train_X,test_X,train_y)
		train_X,test_X = random_shuffle_and_average_pooling(train_X,test_X, 10)
	elif METHOD == 6:
		train_X,test_X = fill_method_4(train_X,test_X,train_y)
		train_X,test_X = pca_transform(train_X,test_X, 10000)

	train_X = train_X.cpu().numpy()
	test_X = test_X.cpu().numpy()
	train_y = train_y.cpu().numpy()
	torch.cuda.empty_cache()
	return train_X, test_X
