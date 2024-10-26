import numpy as np 
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score,f1_score
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from src.data.loaddata import load_data
import os
import matplotlib.pyplot as plt
# 多个受试者的数据
subject_ids = [1, 5]
# 打印当前目录
print(os.getcwd())
base_path = "data" # 数据存放路径
# 初始化 all_X 和 all_y 为空列表，用于存储所有主题的 X 和 y 数据
all_X = []
all_y = []


import pickle
for subject_id in subject_ids:
    data_file = f"{base_path}/data_subject_{subject_id}.pkl"
    
    # 检查数据是否已存在
    if os.path.exists(data_file):
        print(f"Loading data from {data_file} for subject {subject_id}...")
        with open(data_file, 'rb') as f:
            X, y = pickle.load(f)
    else:
        print(f"Data file not found, loading and saving data for subject {subject_id}...")
        X, y = load_data(subject_id, base_path)

        # 保存数据
        with open(data_file, 'wb') as f:
            pickle.dump((X, y), f)
        print(f"Data saved to {data_file}.")

    all_X.append(X)
    all_y.append(y)

'''subject_id = 1
base_path = "data"
all_X,all_y = load_data(subject_id,base_path)'''

val_subject_id = 3
val_base_path = "data"
val_data_file = f"{val_base_path}/data_subject_{val_subject_id}.pkl"
if os.path.exists(val_data_file):
    print(f"Loading data from {val_data_file} for subject {val_subject_id}...")
    with open(val_data_file, 'rb') as f:
        val_all_X, val_all_y = pickle.load(f)
else:
    print(f"Data file not found, loading and saving data for subject {val_subject_id}...")
    val_all_X, val_all_y = load_data(val_subject_id, val_base_path)

    # 保存数据
    with open(val_data_file, 'wb') as f:
        pickle.dump((val_all_X, val_all_y), f)
    print(f"Data saved to {val_data_file}.")
val_X = np.vstack(val_all_X)
val_y = np.concatenate(val_all_y)


# 合并 all_X 和 all_y
X = np.vstack(all_X)
X = X.reshape(-1, X.shape[-1])
# 把 X 划分为两部分train_X和test_X（4:1）
# X_2d = X.reshape(X.shape[0], 2, -1)

train_X = X[:int(len(X)*3/5)]
test_X = X[int(len(X)*3/5):]
y = np.concatenate(all_y)
y = y.reshape(-1)

# 把 y 划分为两部分train_y和test_y（4:1）
train_y = y[:int(len(y)*3/5)]
test_y = y[int(len(y)*3/5):]
# 随机欠采样
from imblearn.under_sampling import RandomUnderSampler
rus = RandomUnderSampler(random_state=0)
#train_X_resampled, train_y_resampled = rus.fit_resample(train_X, train_y)
#test_X_resampled, test_y_resampled = rus.fit_resample(test_X, test_y)
val_X_resampled, val_y_resampled = rus.fit_resample(val_X, val_y)
# SMOTE 过采样
# 初始化 SMOTE 实例
smote = SMOTE()
# @NOTE: 训练集过采样还是有用的
print('过采样前的训练集样本数:', len(train_X))
train_X_resampled, train_y_resampled = smote.fit_resample(train_X, train_y)
print('过采样后的训练集样本数:', len(train_X_resampled))
# train_X_resampled, train_y_resampled = train_X, train_y
test_X_resampled, test_y_resampled = rus.fit_resample(test_X, test_y)
# 分割处理后的数据集
train_X_resampled = np.vstack((train_X_resampled, test_X_resampled))
train_y_resampled = np.concatenate((train_y_resampled, test_y_resampled))
train_2d = train_X_resampled.reshape(train_X_resampled.shape[0], 2, -1)
X_train, X_test, y_train, y_test = train_test_split(train_X_resampled, train_y_resampled, test_size=0.3, random_state=0)
print('训练集样本数:', len(X_train))
print('测试集样本数:', len(X_test))
# 训练决策树分类器
#clf = DecisionTreeClassifier(random_state=0)
#clf.fit(X_train, y_train)
# max-min normalization

# 训练随机森林分类器
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(random_state=0)
clf.fit(X_train, y_train)

# @NOTE:-------------------------------------------------------------------------
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense

X = train_2d
# max-min normalization
X = (X - np.min(X)) / (np.max(X) - np.min(X))
y = train_y_resampled

# 增加通道维度
X = X[:, :, :, np.newaxis]

from sklearn.metrics import f1_score, confusion_matrix, accuracy_score

# 创建卷积神经网络模型
model = Sequential()
model.add(Conv2D(filters=4, kernel_size=(1, 2), activation='relu', input_shape=(2, 96, 1)))
model.add(Conv2D(filters=16, kernel_size=(1, 2), activation='relu'))
model.add(Conv2D(filters=4, kernel_size=(1, 2), activation='relu'))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='sigmoid'))  # 输出层，使用 sigmoid 激活函数

# 编译模型
#model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
import keras
model.compile(optimizer='adam', loss=keras.losses.BinaryFocalCrossentropy(gamma=5.0, from_logits=False), metrics=['accuracy'])
# 训练模型
model.fit(X, y, epochs=15, batch_size=32)

# 对 val_X 进行预测
val_X2d = val_X.reshape(val_X.shape[0], 2, -1, 1)
val_X2d = (val_X2d - np.min(val_X2d)) / (np.max(val_X2d) - np.min(val_X2d))  # 归一化
val_y_pred = model.predict(val_X2d)
val_y_pred = np.where(val_y_pred > 0.5, 1, 0)  # 二值化预测

# 计算准确率
accuracy = accuracy_score(val_y, val_y_pred)
print("Accuracy:", accuracy)

# 计算 F1 分数
f1 = f1_score(val_y, val_y_pred)
print("F1 Score:", f1)

# 计算混淆矩阵
tn, fp, fn, tp = confusion_matrix(val_y, val_y_pred).ravel()
plt.figure()
plt.matshow([[tp, fp], [fn, tn]], cmap='Blues')
plt.colorbar()
plt.xlabel('Predicted')
plt.ylabel('True')
plt.xticks([0, 1], ['Seizure', 'Non-seizure'])
plt.yticks([0, 1], ['Seizure', 'Non-seizure'])
plt.title('Confusion Matrix')
plt.show()

# 计算灵敏度（真正率）
sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
print("Sensitivity (True Positive Rate):", sensitivity)

# 计算特异性 (真负率)
specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
print("Specificity:", specificity)



# @NOTE:-------------------------------------------------------------------------

'''# 十折交叉验证
from sklearn.model_selection import cross_val_score
scores = cross_val_score(clf, X_train, y_train, cv=10)
print("十折交叉验证准确率:", scores.mean())'''

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

val_y_pred_resampled = clf.predict(val_X_resampled)
val_acc_after_resample = accuracy_score(val_y_resampled, val_y_pred_resampled)
print("Val Accuracy After Resample:", val_acc_after_resample)
val_f1_after_resample = f1_score(val_y_resampled, val_y_pred_resampled)
print(f"Val F1 After Resample: {val_f1_after_resample}")
# 计算灵敏度和特异性
cm3 = confusion_matrix(val_y_resampled, val_y_pred_resampled)
tn3, fp3, fn3, tp3 = cm3.ravel()
sensitivity3 = tp3 / (tp3 + fn3)
specificity3 = tn3 / (tn3 + fp3)
print(f"Val 灵敏度: {sensitivity3}")
print(f"Val 特异性: {specificity3}")

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