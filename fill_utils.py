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

		# 计算每列的标准差 std ~ N(data_Mean, data_Std)，并确保 std > 0
		stds = torch.tensor(np.random.normal(data_Mean, data_Std,size = class_X.shape[1]),dtype=torch.float32, device=train_X.device)
		stds = torch.abs(stds) 

		nan_mask = torch.isnan(class_X)
		medians = medians.unsqueeze(0)
		stds = stds.unsqueeze(0)        
		

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
	stds = np.random.normal(data_Mean, data_Std, size=combined_X.shape[1])
	stds = torch.tensor(stds, dtype=torch.float32).cuda()
	stds = torch.abs(stds)
	stds = stds.unsqueeze(0) 
	
	random_fill = torch.normal(
		mean=medians.expand_as(combined_X),
		std = stds.expand_as(combined_X),
	).to(combined_X.device)
	combined_X[nan_mask] = random_fill[nan_mask]
	torch.clamp(combined_X, min=0, max=1, out=combined_X)  # 限制范围在 [0, 1]
	train_X = combined_X[:train_X.shape[0]]
	test_X = combined_X[train_X.shape[0]:]
	return train_X, test_X
