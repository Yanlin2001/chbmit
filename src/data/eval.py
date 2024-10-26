import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

def evaluate_model(y, y_pred):
    # 计算混淆矩阵
    tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()

    # 计算 Precision, Recall, Sensitivity, Specificity
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    sensitivity = recall  # Sensitivity 和 Recall 是相同的
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    # 绘制混淆矩阵
    plt.figure()
    plt.matshow([[tp, fn], [fp, tn]], cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.xticks([0, 1], [
        'Seizure\n(Precision: {:.2f})'.format(precision), 
        'Non-seizure\n(Specificity: {:.2f})'.format(specificity)
    ])
    plt.yticks([0, 1], [
        'Seizure\n(Recall/Sensitivity: {:.2f})'.format(recall), 
        'Non-seizure\n(FN: {})'.format(fn)
    ])
    plt.title('Confusion Matrix')

    # 添加标签和计数
    plt.text(0, 0, f'(TP)\n{tp}', ha='center', va='center', color='black')  # True Positive
    plt.text(1, 0, f'(FN)\n{fn}', ha='center', va='center', color='black')  # False Negative
    plt.text(0, 1, f'(FP)\n{fp}', ha='center', va='center', color='black')  # False Positive
    plt.text(1, 1, f'(TN)\n{tn}', ha='center', va='center', color='white')  # True Negative

    plt.show()

    # 打印f1分数
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    print(f"F1 分数: {f1}")

# 使用示例
# y = np.array([0, 1, 0, 1, 1])  # 真实标签
# y_pred = np.array([0, 0, 1, 1, 1])  # 预测标签
# evaluate_model(y, y_pred)
