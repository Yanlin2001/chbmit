import numpy as np

def equnder_sample(X, y, under_ratio):
    # 找到所有唯一的标签
    unique_labels = np.unique(y)

    # 存储采样后的数据和标签
    sampled_X = []
    sampled_y = []

    for label in unique_labels:
        # 筛选出当前标签的样本
        samples = X[y == label]
        num_samples = len(samples)

        # 计算要保留的样本数量
        num_to_sample = int(num_samples * under_ratio)

        # 确保采样不超过原始样本数量
        num_to_sample = min(num_to_sample, num_samples)

        # 计算采样间隔
        interval = num_samples // num_to_sample

        # 进行等间距采样
        sampled_indices = np.arange(0, num_samples, interval)[:num_to_sample]
        sampled_X.append(samples[sampled_indices])
        sampled_y.append(np.full(num_to_sample, label))

    # 将所有标签的样本合并
    sampled_X = np.vstack(sampled_X)
    sampled_y = np.hstack(sampled_y)

    return sampled_X, sampled_y

def random_under_sample(X, y, under_ratio):
    # 找到所有唯一的标签
    unique_labels = np.unique(y)

    # 存储采样后的数据和标签
    sampled_X = []
    sampled_y = []

    for label in unique_labels:
        # 筛选出当前标签的样本
        samples = X[y == label]
        num_samples = len(samples)

        # 计算要保留的样本数量
        num_to_sample = int(num_samples * under_ratio)

        # 确保采样不超过原始样本数量
        num_to_sample = min(num_to_sample, num_samples)

        # 随机选择样本
        sampled_indices = np.random.choice(num_samples, num_to_sample, replace=False)
        sampled_X.append(samples[sampled_indices])
        sampled_y.append(np.full(num_to_sample, label))

    # 将所有标签的样本合并
    sampled_X = np.vstack(sampled_X)
    sampled_y = np.hstack(sampled_y)

    return sampled_X, sampled_y
