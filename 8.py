import os
import numpy as np
import pandas as pd
import cv2
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense


# 数据生成器函数：分批加载数据，避免一次性将全部数据加载到内存
def data_generator(df, image_folder, batch_size=16, target_size=(224, 224), shuffle=True):
    """
    参数说明：
    df: pandas DataFrame，包含 imagename 及 A-H 等标注列
    image_folder: 图片存储路径
    batch_size: 每批次加载的数据量大小
    target_size: 图片缩放大小 (width, height)
    shuffle: 是否在每个epoch开始前对数据进行打乱

    生成器会不断循环输出 (images, labels) 元组，直到训练结束。
    """
    # 将索引提取出来，以便后续打乱顺序
    indices = df.index.tolist()

    while True:
        if shuffle:
            np.random.shuffle(indices)

        for start in range(0, len(indices), batch_size):
            end = start + batch_size
            batch_indices = indices[start:end]
            batch_images = []
            batch_labels = []

            # 遍历当前批次的索引，从数据集中提取对应的记录
            for idx in batch_indices:
                row = df.loc[idx]
                image_name = str(row['imagename']) + ".jpg"
                image_path = os.path.join(image_folder, image_name)

                # 读取并预处理图片
                image = cv2.imread(image_path)
                if image is not None:
                    # 调整图片大小
                    image = cv2.resize(image, target_size)
                    # 归一化到0-1
                    image = image.astype(np.float32) / 255.0
                    batch_images.append(image)

                    # 标签列，这里假设存在 A-H 8列标签
                    labels = [row['A'], row['B'], row['C'], row['D'],
                              row['E'], row['F'], row['G'], row['H']]
                    batch_labels.append(labels)
                else:
                    # 如果图片无法读取，跳过该样本
                    print(f"无法读取图片：{image_path}，将跳过此样本。")

            # 转换为 numpy 数组
            batch_images = np.array(batch_images, dtype=np.float32)
            batch_labels = np.array(batch_labels, dtype=np.float32)

            yield batch_images, batch_labels


# ---------------- 以下为主流程代码 ----------------

# 数据文件夹路径（用户需根据实际情况修改）
image_folder = r'E:\Ky\documents\xl\8y\Total'
label_file = r'E:\Ky\documents\xl\8y\labels2.xlsx'

# 读取标签文件
labels_df = pd.read_excel(label_file, engine='openpyxl')
labels_df['imagename'] = labels_df['imagename'].astype(str)

# 将数据集分为训练集和测试集
train_df, test_df = train_test_split(labels_df, test_size=0.1, random_state=24)

# 定义参数
batch_size = 16
target_size = (224, 224)  # 图片缩放大小
epochs = 20

# 创建训练和测试的数据生成器
train_gen = data_generator(train_df, image_folder, batch_size=batch_size, target_size=target_size, shuffle=True)
test_gen = data_generator(test_df, image_folder, batch_size=batch_size, target_size=target_size, shuffle=False)

# 计算 steps_per_epoch 和 validation_steps
steps_per_epoch = len(train_df) // batch_size
validation_steps = len(test_df) // batch_size

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
    Dense(8)  # 输出为8个指标对应的预测值
])

# 编译模型
model.compile(optimizer='adam', loss='mse')

# 使用生成器进行训练
history = model.fit(
    train_gen,
    steps_per_epoch=steps_per_epoch,
    epochs=epochs,
    validation_data=test_gen,
    validation_steps=validation_steps
)

# 自定义模型保存路径
save_path = r'E:\Ky\documents\xl\8y\8y.keras'
model.save(save_path)
print(f"模型已保存到：{save_path}")

# 为了评估模型，需要从测试集生成器中取出所有测试数据（这里假设测试数据总量不大，可一次性取出进行评估）。
# 如果数据仍然较大，可考虑分批评估或随机抽取一部分测试数据。

# 获取完整的测试数据（可能需要多次迭代生成器）
test_images = []
test_labels = []

test_df_indices = test_df.index.tolist()
for start in range(0, len(test_df_indices), batch_size):
    end = start + batch_size
    batch_indices = test_df_indices[start:end]
    temp_images = []
    temp_labels = []
    for idx in batch_indices:
        row = test_df.loc[idx]
        image_path = os.path.join(image_folder, str(row['imagename']) + ".jpg")
        image = cv2.imread(image_path)
        if image is not None:
            image = cv2.resize(image, target_size)
            image = image.astype(np.float32) / 255.0
            temp_images.append(image)
            temp_labels.append([row['A'], row['B'], row['C'], row['D'],
                                row['E'], row['F'], row['G'], row['H']])

    if len(temp_images) > 0:
        test_images.extend(temp_images)
        test_labels.extend(temp_labels)

test_images = np.array(test_images, dtype=np.float32)
test_labels = np.array(test_labels, dtype=np.float32)

# 对测试集进行预测
y_pred = model.predict(test_images)

# 计算每种抗生素浓度指标
antibiotics = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
results = {}
for i, antibiotic in enumerate(antibiotics):
    rmse = np.sqrt(mean_squared_error(test_labels[:, i], y_pred[:, i]))
    r2 = r2_score(test_labels[:, i], y_pred[:, i])
    mae = mean_absolute_error(test_labels[:, i], y_pred[:, i])
    results[antibiotic] = {'RMSE': rmse, 'R²': r2, 'MAE': mae}

# 打印评估结果
for antibiotic, metrics in results.items():
    print(f"抗生素 {antibiotic}:")
    print(f"  RMSE: {metrics['RMSE']}")
    print(f"  R²: {metrics['R²']}")
    print(f"  MAE: {metrics['MAE']}\n")

# 绘制散点图：真实值 vs 预测值
plt.figure(figsize=(14, 6))
for i, antibiotic in enumerate(antibiotics):
    plt.scatter(test_labels[:, i], y_pred[:, i], label=f'抗生素 {antibiotic}', alpha=0.5)

plt.xlabel('真实值')
plt.ylabel('预测值')
plt.title('测试集: 真实值 vs 预测值')
plt.legend()
plt.tight_layout()
plt.show()

# 将测试集的预测结果和评估指标保存到Excel文件中
output_file = r'E:\Ky\documents\xl\8y\results.xlsx'
results_df = pd.DataFrame({
    'imagename': test_df['imagename'],
    'Actual_A': test_labels[:, 0],
    'Predicted_A': y_pred[:, 0],
    'Actual_B': test_labels[:, 1],
    'Predicted_B': y_pred[:, 1],
    'Actual_C': test_labels[:, 2],
    'Predicted_C': y_pred[:, 2],
    'Actual_D': test_labels[:, 3],
    'Predicted_D': y_pred[:, 3],
    'Actual_E': test_labels[:, 4],
    'Predicted_E': y_pred[:, 4],
    'Actual_F': test_labels[:, 5],
    'Predicted_F': y_pred[:, 5],
    'Actual_G': test_labels[:, 6],
    'Predicted_G': y_pred[:, 6],
    'Actual_H': test_labels[:, 7],
    'Predicted_H': y_pred[:, 7]
})

results_df.to_excel(output_file, index=False)
print(f"测试集的预测结果和评估指标已保存到：{output_file}")