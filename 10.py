# -*- coding: utf-8 -*-
"""
Created on Wed Jan 15 23:12:33 2025

@author: 86187
"""

import os
import numpy as np
import pandas as pd
import cv2
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam


# 数据生成器函数
def data_generator(df, image_folder, batch_size=16, target_size=(224, 224), shuffle=True):
    indices = df.index.tolist()
    while True:
        if shuffle:
            np.random.shuffle(indices)
        for start in range(0, len(indices), batch_size):
            end = start + batch_size
            batch_indices = indices[start:end]
            batch_images = []
            batch_labels = []
            for idx in batch_indices:
                row = df.loc[idx]
                image_name = str(row['imagename']) + ".jpg"
                image_path = os.path.join(image_folder, image_name)
                image = cv2.imread(image_path)
                if image is None:
                    print(f"无法读取图片：{image_path}，将跳过此样本。")
                    continue
                if np.isnan(image).any() or np.isinf(image).any():
                    print(f"无效图像：{image_path}")
                    continue
                image = cv2.resize(image, target_size)
                image = image.astype(np.float32) / 255.0
                batch_images.append(image)
                labels = [row['A'], row['B'], row['C'], row['D'],
                          row['E'], row['F'], row['G'], row['H'],
                          row['I'], row['J']]
                if np.isnan(labels).any() or np.isinf(labels).any():
                    print(f"无效标签：{labels}，样本编号：{row['imagename']}")
                    continue
                batch_labels.append(labels)
            batch_images = np.array(batch_images, dtype=np.float32)
            batch_labels = np.array(batch_labels, dtype=np.float32)
            yield batch_images, batch_labels


# 检查数据函数
def check_data(df, image_folder):
    """
    检查数据集中是否存在无效图像或标签，并输出异常标签的编号。
    """
    invalid_images = 0
    invalid_labels = 0
    invalid_label_samples = []  # 存储异常标签的编号

    for idx, row in df.iterrows():
        image_name = str(row['imagename']) + ".jpg"
        image_path = os.path.join(image_folder, image_name)
        image = cv2.imread(image_path)

        # 检查图像
        if image is None:
            print(f"无法读取图片：{image_path}")
            invalid_images += 1
            continue
        if np.isnan(image).any() or np.isinf(image).any():
            print(f"无效图像：{image_path}")
            invalid_images += 1
            continue

        # 检查标签
        labels = [row['A'], row['B'], row['C'], row['D'],
                  row['E'], row['F'], row['G'], row['H'],
                  row['I'], row['J']]
        if np.isnan(labels).any() or np.isinf(labels).any():
            print(f"无效标签：{labels}，样本编号：{row['imagename']}")
            invalid_labels += 1
            invalid_label_samples.append(row['imagename'])  # 记录异常标签的编号
            continue

    print(f"检查完成：{len(df)} 个样本中，{invalid_images} 个无效图像，{invalid_labels} 个无效标签。")
    if invalid_label_samples:
        print("异常标签的样本编号：", invalid_label_samples)


# 主流程代码
if __name__ == "__main__":
    # 数据文件夹路径
    image_folder = r'E:\Ky\documents\xl\10y\Total2'
    label_file = r'E:\Ky\documents\xl\10y\labels2.xlsx'

    # 读取标签文件
    labels_df = pd.read_excel(label_file, engine='openpyxl')
    labels_df['imagename'] = labels_df['imagename'].astype(str)

    # 检查数据
    print("检查训练数据...")
    check_data(labels_df, image_folder)

    # 将数据集分为训练集和测试集
    train_df, test_df = train_test_split(labels_df, test_size=0.1, random_state=24)

    # 定义参数
    batch_size = 16
    target_size = (224, 224)
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
        Dense(10)  # 输出为10个指标对应的预测值
    ])

    # 编译模型，使用默认学习率 0.001
    optimizer = Adam(learning_rate=0.001)  # 默认学习率为 0.001
    model.compile(optimizer=optimizer, loss='mse')

    # 使用生成器进行训练
    history = model.fit(
        train_gen,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        validation_data=test_gen,
        validation_steps=validation_steps
    )

    # 保存模型
    save_path = r'E:\Ky\documents\xl\10y\mx\10yzdy.keras'
    model.save(save_path)
    print(f"模型已保存到：{save_path}")

    # 获取完整的训练数据（可能需要多次迭代生成器）
    train_images = []
    train_labels = []

    train_df_indices = train_df.index.tolist()
    for start in range(0, len(train_df_indices), batch_size):
        end = start + batch_size
        batch_indices = train_df_indices[start:end]
        temp_images = []
        temp_labels = []
        for idx in batch_indices:
            row = train_df.loc[idx]
            image_path = os.path.join(image_folder, str(row['imagename']) + ".jpg")
            image = cv2.imread(image_path)
            if image is not None:
                image = cv2.resize(image, target_size)
                image = image.astype(np.float32) / 255.0
                temp_images.append(image)
                temp_labels.append([row['A'], row['B'], row['C'], row['D'],
                                    row['E'], row['F'], row['G'], row['H'],
                                    row['I'], row['J']])

        if len(temp_images) > 0:
            train_images.extend(temp_images)
            train_labels.extend(temp_labels)

    train_images = np.array(train_images, dtype=np.float32)
    train_labels = np.array(train_labels, dtype=np.float32)

    # 对训练集进行预测
    y_train_pred = model.predict(train_images)

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
                                    row['E'], row['F'], row['G'], row['H'],
                                    row['I'], row['J']])

        if len(temp_images) > 0:
            test_images.extend(temp_images)
            test_labels.extend(temp_labels)

    test_images = np.array(test_images, dtype=np.float32)
    test_labels = np.array(test_labels, dtype=np.float32)

    # 对测试集进行预测
    y_pred = model.predict(test_images)

    # 计算每种抗生素浓度指标
    antibiotics = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
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

    # 绘制训练集的散点图：真实值 vs 预测值
    plt.figure(figsize=(14, 6))
    for i, antibiotic in enumerate(antibiotics):
        plt.scatter(train_labels[:, i], y_train_pred[:, i], label=f'抗生素 {antibiotic}', alpha=0.5)

    plt.xlabel('真实值')
    plt.ylabel('预测值')
    plt.title('训练集: 真实值 vs 预测值')
    plt.legend()
    plt.tight_layout()
    plt.show()

    # 绘制测试集的散点图：真实值 vs 预测值
    plt.figure(figsize=(14, 6))
    for i, antibiotic in enumerate(antibiotics):
        plt.scatter(test_labels[:, i], y_pred[:, i], label=f'抗生素 {antibiotic}', alpha=0.5)

    plt.xlabel('真实值')
    plt.ylabel('预测值')
    plt.title('测试集: 真实值 vs 预测值')
    plt.legend()
    plt.tight_layout()
    plt.show()

    # 将训练集和测试集的预测结果保存到同一个Excel文件的不同工作表中
    output_file = r'E:\Ky\documents\xl\10y\results\results.xlsx'
    with pd.ExcelWriter(output_file) as writer:
        train_results_df = pd.DataFrame({
            'imagename': train_df['imagename'],
            'Actual_A': train_labels[:, 0],
            'Predicted_A': y_train_pred[:, 0],
            'Actual_B': train_labels[:, 1],
            'Predicted_B': y_train_pred[:, 1],
            'Actual_C': train_labels[:, 2],
            'Predicted_C': y_train_pred[:, 2],
            'Actual_D': train_labels[:, 3],
            'Predicted_D': y_train_pred[:, 3],
            'Actual_E': train_labels[:, 4],
            'Predicted_E': y_train_pred[:, 4],
            'Actual_F': train_labels[:, 5],
            'Predicted_F': y_train_pred[:, 5],
            'Actual_G': train_labels[:, 6],
            'Predicted_G': y_train_pred[:, 6],
            'Actual_H': train_labels[:, 7],
            'Predicted_H': y_train_pred[:, 7],
            'Actual_I': train_labels[:, 8],
            'Predicted_I': y_train_pred[:, 8],
            'Actual_J': train_labels[:, 9],
            'Predicted_J': y_train_pred[:, 9]
        })
        train_results_df.to_excel(writer, sheet_name='训练集', index=False)

        test_results_df = pd.DataFrame({
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
            'Predicted_H': y_pred[:, 7],
            'Actual_I': test_labels[:, 8],
            'Predicted_I': y_pred[:, 8],
            'Actual_J': test_labels[:, 9],
            'Predicted_J': y_pred[:, 9]
        })
        test_results_df.to_excel(writer, sheet_name='测试集', index=False)

    print(f"训练集和测试集的预测结果已保存到：{output_file}")