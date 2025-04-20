import os
import numpy as np
import pandas as pd
import cv2
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# 加载数据函数
def load_data(image_folder, label_file):
    labels_df = pd.read_excel(label_file, engine='openpyxl')
    labels_df['imagename'] = labels_df['imagename'].astype(str)
    images = []
    labels = []
    for index, row in labels_df.iterrows():
        image_path = os.path.join(image_folder, f"{row['imagename']}.jpg")
        image = cv2.imread(image_path)
        if image is not None:
            image = cv2.resize(image, (224, 224))
            images.append(image)
            labels.append([row['A'], row['B'], row['C']])
        else:
            print(f"图片 {image_path} 无法找到或打开。")
    return np.array(images), np.array(labels)

# 数据文件夹路径
image_folder = r'E:\Ky\documents\xl\3y\Total6'
label_file = r'E:\Ky\\documents\xl\3y\labels6.xlsx'

# 加载数据
images, labels = load_data(image_folder, label_file)

# 数据预处理与划分
images = images / 255.0
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.1, random_state=42)

# 构建CNN模型
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(3)
])

# 编译模型
model.compile(optimizer='adam', loss='mse')

# 训练模型
history = model.fit(X_train, y_train, epochs=20, validation_data=(X_test, y_test), batch_size=20)

# 保存模型
save_path = r'E:\Ky\documents\xl\3y\3y.keras'
model.save(save_path)
print(f"模型已保存到：{save_path}")

# 模型评估
y_pred = model.predict(X_test)

# 初始化存储结果的字典
results = {}
for i, antibiotic in enumerate(['A', 'B', 'C']):
    rmse = np.sqrt(mean_squared_error(y_test[:, i], y_pred[:, i]))
    r2 = r2_score(y_test[:, i], y_pred[:, i])
    mae = mean_absolute_error(y_test[:, i], y_pred[:, i])
    results[antibiotic] = {'RMSE': rmse, 'R²': r2, 'MAE': mae}

# 打印评估结果
for antibiotic, metrics in results.items():
    print(f"抗生素 {antibiotic}:")
    print(f"  RMSE: {metrics['RMSE']}")
    print(f"  R²: {metrics['R²']}")
    print(f"  MAE: {metrics['MAE']}\n")

# 绘制散点图
y_train_pred = model.predict(X_train)
plt.figure(figsize=(14, 6))
plt.subplot(1, 2, 1)
plt.scatter(y_train[:, 0], y_train_pred[:, 0], label='抗生素 A', alpha=0.5)
plt.scatter(y_train[:, 1], y_train_pred[:, 1], label='抗生素 B', alpha=0.5)
plt.scatter(y_train[:, 2], y_train_pred[:, 2], label='抗生素 C', alpha=0.5)
plt.xlabel('真实值')
plt.ylabel('预测值')
plt.title('训练集: 真实值 vs 预测值')
plt.legend()
plt.subplot(1, 2, 2)
plt.scatter(y_test[:, 0], y_pred[:, 0], label='抗生素 A', alpha=0.5)
plt.scatter(y_test[:, 1], y_pred[:, 1], label='抗生素 B', alpha=0.5)
plt.scatter(y_test[:, 2], y_pred[:, 2], label='抗生素 C', alpha=0.5)
plt.xlabel('真实值')
plt.ylabel('预测值')
plt.title('测试集: 真实值 vs 预测值')
plt.legend()
plt.tight_layout()
plt.show()

# 创建DataFrame存储训练集真实值和预测值
train_results_df = pd.DataFrame({
    'Actual_A': y_train[:, 0],
    'Predicted_A': y_train_pred[:, 0],
    'Actual_B': y_train[:, 1],
    'Predicted_B': y_train_pred[:, 1],
    'Actual_C': y_train[:, 2],
    'Predicted_C': y_train_pred[:, 2]
})

# 创建DataFrame存储测试集真实值和预测值
test_results_df = pd.DataFrame({
    'Actual_A': y_test[:, 0],
    'Predicted_A': y_pred[:, 0],
    'Actual_B': y_test[:, 1],
    'Predicted_B': y_pred[:, 1],
    'Actual_C': y_test[:, 2],
    'Predicted_C': y_pred[:, 2]
})

# 定义保存路径
output_file = r'E:\Ky\documents\xl\3y\results\cs\results3.xlsx'

# 使用 ExcelWriter 将训练集和测试集结果保存到同一个 Excel 文件的不同工作表中
with pd.ExcelWriter(output_file) as writer:
    train_results_df.to_excel(writer, sheet_name='训练集', index=False)
    test_results_df.to_excel(writer, sheet_name='测试集', index=False)

print(f"训练集和测试集结果已保存到：{output_file}")