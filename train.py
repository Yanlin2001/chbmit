import numpy as np 
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score,f1_score
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from src.data.loaddata import load_data
import os

# 多个受试者的数据
subject_ids = [1, 2,5]
# 打印当前目录
print(os.getcwd())
base_path = "data" # 数据存放路径
# 初始化 all_X 和 all_y 为空列表，用于存储所有主题的 X 和 y 数据
all_X = []
all_y = []

for subject_id in subject_ids:
    # 加载数据
    X, y = load_data(subject_id, base_path)
    # 将 X 和 y 添加到 all_X 和 all_y 中
    all_X.append(X)
    all_y.append(y)

'''subject_id = 1
base_path = "data"
all_X,all_y = load_data(subject_id,base_path)'''

val_subject_id = 3
val_base_path = "data"
val_all_X,val_all_y = load_data(val_subject_id,val_base_path)
val_X = np.vstack(val_all_X)
val_y = np.concatenate(val_all_y)


# 合并 all_X 和 all_y
X = np.vstack(all_X)
X = X.reshape(-1, X.shape[-1])
# 把 X 划分为两部分train_X和test_X（4:1）
train_X = X[:int(len(X)*4/5)]
test_X = X[int(len(X)*4/5):]
y = np.concatenate(all_y)
y = y.reshape(-1)
# 把 y 划分为两部分train_y和test_y（4:1）
train_y = y[:int(len(y)*4/5)]
test_y = y[int(len(y)*4/5):]


# 初始化 SMOTE 实例
smote = SMOTE()

# 应用 SMOTE 过采样

X_test_resampled, y_test_resampled = smote.fit_resample(test_X, test_y)
# 将原本train_X和train_y与X_test_resampled, y_test_resampled合并为X_train_resampled, y_train_resampled
X_train_resampled = np.vstack((train_X, X_test_resampled))
y_train_resampled = np.concatenate((train_y, y_test_resampled))
# 分割处理后的数据集
X_train, X_test, y_train, y_test = train_test_split(X_train_resampled, y_train_resampled, test_size=0.3, random_state=0)
print('训练集样本数:', len(X_train))
print('测试集样本数:', len(X_test))
# 训练决策树分类器
#clf = DecisionTreeClassifier(random_state=0)
#clf.fit(X_train, y_train)
# max-min normalization
'''from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
val_X = scaler.transform(val_X)
X_test_resampled = scaler.transform(X_test_resampled)
test_X = scaler.transform(test_X)'''

# 训练随机森林分类器
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(random_state=0)
clf.fit(X_train, y_train)


# 对测试集进行预测
y_pred = clf.predict(X_test)
# 评估模型
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# 计算 F1 分数
f1 = f1_score(y_test, y_pred)
print(f"F1 分数: {f1}")

# 计算灵敏度和特异性
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()
sensitivity = tp / (tp + fn)
specificity = tn / (tn + fp)
print(f"灵敏度: {sensitivity}")
print(f"特异性: {specificity}")


# 对resample前的验证集进行预测
val_y_pred = clf.predict(val_X)
val_acc_before_resample = accuracy_score(val_y, val_y_pred)
print("Val Accuracy Before Resample:", val_acc_before_resample)
val_f1_before_resample = f1_score(val_y, val_y_pred)
print(f"Val F1 Before Resample: {val_f1_before_resample}")
# 计算灵敏度和特异性
cm2 = confusion_matrix(val_y, val_y_pred)
tn2, fp2, fn2, tp2 = cm2.ravel()
sensitivity2 = tp2 / (tp2 + fn2)
specificity2 = tn2 / (tn2 + fp2)
print(f"Val 灵敏度: {sensitivity2}")
print(f"Val 特异性: {specificity2}")


# 对resample前的test_X进行预测
test_y_pred = clf.predict(test_X)
test_acc_before_resample = accuracy_score(test_y, test_y_pred)
print("Test Accuracy Before Resample:", test_acc_before_resample)
test_f1_before_resample = f1_score(test_y, test_y_pred)
print(f"Test F1 Before Resample: {test_f1_before_resample}")
# 计算灵敏度和特异性
cm4 = confusion_matrix(test_y, test_y_pred)
tn4, fp4, fn4, tp4 = cm4.ravel()
sensitivity4 = tp4 / (tp4 + fn4)
specificity4 = tn4 / (tn4 + fp4)
print(f"Test 灵敏度: {sensitivity4}")
print(f"Test 特异性: {specificity4}")

# 对resample后的test_X进行预测
test_y_pred_resampled = clf.predict(X_test_resampled)
test_acc_after_resample = accuracy_score(y_test_resampled, test_y_pred_resampled)
print("Test Accuracy After Resample:", test_acc_after_resample)
test_f1_after_resample = f1_score(y_test_resampled, test_y_pred_resampled)
print(f"Test F1 After Resample: {test_f1_after_resample}")
# 计算灵敏度和特异性
cm5 = confusion_matrix(y_test_resampled, test_y_pred_resampled)
tn5, fp5, fn5, tp5 = cm5.ravel()
sensitivity5 = tp5 / (tp5 + fn5)
specificity5 = tn5 / (tn5 + fp5)
print(f"Test 灵敏度: {sensitivity5}")
print(f"Test 特异性: {specificity5}")