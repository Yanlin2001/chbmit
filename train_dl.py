import numpy as np 
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score,f1_score
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from src.data.loaddata import load_data
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
class FocalLoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # 计算软最大值
        BCE_loss = nn.CrossEntropyLoss(reduction='none')(inputs, targets)
        pt = torch.exp(-BCE_loss)  # 计算预测概率
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
        
        if self.reduction == 'mean':
            return F_loss.mean()
        elif self.reduction == 'sum':
            return F_loss.sum()
        else:
            return F_loss


# 多个受试者的数据
subject_ids = [1, 2, 5]
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
val_X = np.vstack(val_all_X)
val_y = np.concatenate(val_all_y)

# 初始化 SMOTE 实例
smote = SMOTE()

# 应用 SMOTE 过采样
# X_train_resampled, y_train_resampled = smote.fit_resample(train_X, train_y)
X_test_resampled, y_test_resampled = smote.fit_resample(test_X, test_y)
val_X_resampled, val_y_resampled = smote.fit_resample(val_X, val_y)
# 将原本train_X和train_y与X_test_resampled, y_test_resampled合并为X_train_resampled, y_train_resampled
X_train_resampled = np.vstack((train_X, X_test_resampled))
y_train_resampled = np.concatenate((train_y, y_test_resampled))
# 分割处理后的数据集

X_train, X_test, y_train, y_test = train_test_split(train_X, train_y, test_size=0.3, random_state=0)
print('训练集样本数:', len(X_train))
print('测试集样本数:', len(X_test))
# 训练决策树分类器
#clf = DecisionTreeClassifier(random_state=0)
#clf.fit(X_train, y_train)

train_loader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long)),
    batch_size=64,
    shuffle=True
)

val_loader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(torch.tensor(val_X_resampled, dtype=torch.float32), torch.tensor(val_y_resampled, dtype=torch.long)),
    batch_size=64,
    shuffle=False
)

# 定义 MLP 模型
class MLP(nn.Module):
    def __init__(self, input_dim=18, hidden_dim=64, output_dim=10):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# 假设类别数量为10
model = MLP(input_dim=18, hidden_dim=64, output_dim=10)

# 定义 Focal Loss 和优化器
criterion = FocalLoss(alpha=1.0, gamma=2.0)  # 调整 alpha 和 gamma 根据需要
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练过程示例
def train(model, train_loader, optimizer, criterion, epochs=10):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)  # 使用 Focal Loss
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader)}")

# 训练模型
train(model, train_loader, optimizer, criterion, epochs=10)
y_pred = model(torch.tensor(X_test, dtype=torch.float32)).argmax(dim=1).numpy()

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

# 对resample后的验证集进行预测
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