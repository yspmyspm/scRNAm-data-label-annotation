import torch
def random_shuffle_and_average_pooling(train_X, test_X, pool_size):
	n_samples_train, n_features = train_X.shape
	n_samples_test = test_X.shape[0]

	shuffled_indices = torch.argsort(torch.rand(1, n_features), dim=1).expand(n_samples_train + n_samples_test, -1)
	shuffled_indices = shuffled_indices.to(train_X.device)
	shuffled_train_X = torch.gather(train_X, dim=1, index=shuffled_indices[:n_samples_train])
	shuffled_test_X = torch.gather(test_X, dim=1, index=shuffled_indices[n_samples_train:])

	def apply_average_pooling(X, pool_size):
		n_features = X.shape[1]
		n_pooled_features = n_features // pool_size
		pooled_X = X[:, :n_pooled_features * pool_size].reshape(-1, n_pooled_features, pool_size)
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


