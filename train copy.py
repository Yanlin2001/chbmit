import numpy as np 
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score,f1_score
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from src.data.loaddata import load_data

subject_id = 1
base_path = "data"
all_X,all_y = load_data(subject_id,base_path)

val_subject_id = 5
val_base_path = "data"
#val_all_X,val_all_y = load_data(val_subject_id,val_base_path)

# 合并 all_X 和 all_y
X = np.vstack(all_X)
y = np.concatenate(all_y)
#val_X = np.vstack(val_all_X)
#val_y = np.concatenate(val_all_y)
val_X = X
val_y = y


from imblearn.under_sampling import RandomUnderSampler

# 初始化 RandomUnderSampler 实例
rus = RandomUnderSampler(random_state=42)  # 可以设置随机种子以确保结果的可重复性

# 应用 RandomUnderSampler 欠采样
X_resampled, y_resampled = rus.fit_resample(X, y)
# 分割处理后的数据集
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.3, random_state=0)

# 训练决策树分类器
clf = DecisionTreeClassifier(random_state=0)
clf.fit(X_train, y_train)

# 对测试集进行预测
y_pred = clf.predict(X_test)
val_y_pred = clf.predict(val_X)
# 评估模型
accuracy = accuracy_score(y_test, y_pred)
val_accuracy = accuracy_score(val_y,val_y_pred)
print("Accuracy:", accuracy)
print("Val Accuracy:", val_accuracy)

# 计算 F1 分数
f1 = f1_score(y_test, y_pred)
val_f1 = f1_score(val_y,val_y_pred)
print(f"F1 分数: {f1}")
print(f"Val F1 分数: {val_f1}")