import os
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import RandomizedSearchCV
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
from tensorflow.keras import backend as K
import itertools

# ---------------------- 配置参数区 ----------------------
# 输入输出路径
IMAGE_FOLDER = r"E:\Ky\documents\xl\数据集路径"  # 图像文件夹
LABEL_FILE = r"E:\Ky\documents\xl\标签文件.xlsx"  # 标签Excel文件
SAVE_ROOT = r"E:\Ky\documents\xl\结果保存路径"  # 结果保存根目录

# 训练基本参数
BATCH_SIZE = 16
EPOCHS = 20
TARGET_SIZE = (224, 224)
TEST_SIZE = 0.1  # 测试集比例
RANDOM_STATE = 42

# 交叉验证参数
KFOLD_SPLITS = 5  # K折交叉验证的折数

# 超参数搜索空间
PARAM_GRID = {
    'learning_rate': [0.001, 0.0005, 0.0001],
    'conv_filters': [[32, 64, 128], [16, 32, 64], [64, 128, 256]],
    'dense_units': [64, 128, 256],
    'dropout_rate': [0.0, 0.2, 0.4]
}

# 超参数搜索设置
SEARCH_ITERATIONS = 10  # 随机搜索的迭代次数

# 可视化参数
VISUALIZATION_SAMPLES = 5  # 用于可视化的样本数量
LAYER_VISUALIZATION = [0, 2, 4]  # 要可视化的卷积层索引


# -------------------------------------------------------


def data_generator(df, image_folder, target_size=(224, 224), batch_size=16, shuffle=True):
    """通用数据生成器，支持动态读取不同数量的指标"""
    indices = df.index.tolist()
    # 获取指标列（排除imagename）
    metric_cols = [col for col in df.columns if col != 'imagename']

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
                image_name = f"{row['imagename']}.jpg"
                image_path = os.path.join(image_folder, image_name)
                image = cv2.imread(image_path)

                if image is None:
                    print(f"跳过无效图像：{image_path}")
                    continue
                # 图像预处理
                image = cv2.resize(image, target_size)
                image = image.astype(np.float32) / 255.0
                batch_images.append(image)

                # 动态获取标签（根据指标列）
                labels = [row[col] for col in metric_cols]
                if np.isnan(labels).any() or np.isinf(labels).any():
                    print(f"跳过无效标签：{labels}（样本：{row['imagename']}）")
                    continue
                batch_labels.append(labels)

            if batch_images:  # 确保批次非空
                yield np.array(batch_images), np.array(batch_labels)


def check_data_validity(df, image_folder):
    """检查数据有效性并返回有效样本"""
    metric_cols = [col for col in df.columns if col != 'imagename']
    valid_indices = []

    for idx, row in df.iterrows():
        image_path = os.path.join(image_folder, f"{row['imagename']}.jpg")
        if not os.path.exists(image_path) or cv2.imread(image_path) is None:
            print(f"无效图像：{image_path}")
            continue

        labels = [row[col] for col in metric_cols]
        if np.isnan(labels).any() or np.isinf(labels).any():
            print(f"无效标签：{labels}（样本：{row['imagename']}）")
            continue

        valid_indices.append(idx)

    print(f"有效样本数：{len(valid_indices)}/{len(df)}")
    return df.loc[valid_indices]


def build_model(input_shape, num_metrics, learning_rate=0.001,
                conv_filters=[32, 64, 128], dense_units=128, dropout_rate=0.2):
    """构建模型的函数，用于超参数搜索"""
    model = Sequential()

    # 添加卷积层
    for i, filters in enumerate(conv_filters):
        if i == 0:
            model.add(Conv2D(filters, (3, 3), activation='relu', input_shape=input_shape, name=f'conv_{i}'))
        else:
            model.add(Conv2D(filters, (3, 3), activation='relu', name=f'conv_{i}'))
        model.add(MaxPooling2D((2, 2), name=f'pool_{i}'))

    model.add(Flatten(name='flatten'))
    model.add(Dense(dense_units, activation='relu', name='dense_1'))
    model.add(Dropout(dropout_rate, name='dropout'))
    model.add(Dense(num_metrics, name='output'))  # 动态设置输出维度

    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mse')
    return model


def evaluate_model(model, test_gen, validation_steps, metric_cols):
    """评估模型并返回指标"""
    # 获取测试集预测结果
    test_preds = []
    test_labels = []

    for _ in range(validation_steps):
        x, y = next(test_gen)
        preds = model.predict(x)
        test_preds.extend(preds)
        test_labels.extend(y)

    test_preds = np.array(test_preds)
    test_labels = np.array(test_labels)

    # 计算评估指标
    results = {}
    for i, metric in enumerate(metric_cols):
        rmse = np.sqrt(mean_squared_error(test_labels[:, i], test_preds[:, i]))
        r2 = r2_score(test_labels[:, i], test_preds[:, i])
        mae = mean_absolute_error(test_labels[:, i], test_preds[:, i])
        results[metric] = {"RMSE": rmse, "R²": r2, "MAE": mae}

    return results, test_labels, test_preds


def visualize_intermediate_layers(model, image, layer_indices, save_path=None):
    """可视化中间卷积层的激活"""
    # 获取指定层的输出
    layer_outputs = [model.layers[i].output for i in layer_indices]
    activation_model = Model(inputs=model.input, outputs=layer_outputs)
    activations = activation_model.predict(np.expand_dims(image, axis=0))

    # 创建可视化图像
    images_per_row = 16
    plt.figure(figsize=(15, 10))

    for layer_idx, layer_activation in zip(layer_indices, activations):
        n_features = layer_activation.shape[-1]  # 特征图数量
        size = layer_activation.shape[1]  # 特征图大小

        n_cols = n_features // images_per_row
        display_grid = np.zeros((size * n_cols, images_per_row * size))

        # 填充特征图到网格
        for col in range(n_cols):
            for row in range(images_per_row):
                channel_image = layer_activation[0, :, :, col * images_per_row + row]
                # 标准化以更好地可视化
                channel_image -= channel_image.mean()
                channel_image /= (channel_image.std() + 1e-5)
                channel_image *= 64
                channel_image += 128
                channel_image = np.clip(channel_image, 0, 255).astype('uint8')
                display_grid[col * size: (col + 1) * size,
                row * size: (row + 1) * size] = channel_image

        # 显示网格
        plt.subplot(len(layer_indices), 1, layer_indices.index(layer_idx) + 1)
        plt.title(f'Layer {layer_idx}: {model.layers[layer_idx].name}')
        plt.grid(False)
        plt.imshow(display_grid, aspect='auto', cmap='viridis')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"中间层可视化已保存至：{save_path}")
    plt.close()


def grad_cam(model, img_array, metric_index, layer_name=None):
    """
    实现Grad-CAM方法生成特征重要性热力图
    https://arxiv.org/pdf/1610.02391.pdf
    """
    # 如果未指定层名，使用最后一个卷积层
    if layer_name is None:
        for layer in reversed(model.layers):
            if isinstance(layer, Conv2D):
                layer_name = layer.name
                break

    # 创建一个模型，输入为原始模型输入，输出为目标层输出和最终输出
    grad_model = Model(
        inputs=[model.inputs],
        outputs=[model.get_layer(layer_name).output, model.output]
    )

    # 计算梯度
    with K.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, metric_index]  # 针对特定指标的损失

    # 目标层输出关于损失的梯度
    grads = tape.gradient(loss, conv_outputs)

    # 全局平均池化梯度以获得通道重要性权重
    pooled_grads = K.mean(grads, axis=(0, 1, 2))

    # 将权重应用于卷积输出通道
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., np.newaxis]
    heatmap = K.squeeze(heatmap, axis=-1)

    # ReLU激活，只保留正影响
    heatmap = K.maximum(heatmap, 0)

    # 归一化
    heatmap /= K.max(heatmap)
    return heatmap.numpy()


def visualize_feature_importance(model, img, img_array, metric_index, metric_name, save_path=None):
    """可视化特征重要性（Grad-CAM热力图）"""
    # 生成热力图
    heatmap = grad_cam(model, img_array, metric_index)

    # 将热力图调整为与原始图像相同的大小
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))

    # 将热力图转换为RGB
    heatmap = np.uint8(255 * heatmap)

    # 应用颜色映射
    heatmap = cm.jet(heatmap)[..., :3]

    # 将热力图与原始图像叠加
    superimposed_img = heatmap * 0.4 + img
    superimposed_img = np.uint8(superimposed_img / np.max(superimposed_img) * 255)

    # 显示结果
    plt.figure(figsize=(12, 4))

    plt.subplot(131)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title('原始图像')
    plt.axis('off')

    plt.subplot(132)
    plt.imshow(heatmap)
    plt.title(f'Grad-CAM热力图\n({metric_name})')
    plt.axis('off')

    plt.subplot(133)
    plt.imshow(cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB))
    plt.title(f'叠加结果\n({metric_name})')
    plt.axis('off')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"特征重要性可视化已保存至：{save_path}")
    plt.close()


def main():
    # 创建保存目录及可视化子目录
    os.makedirs(SAVE_ROOT, exist_ok=True)
    vis_dir = os.path.join(SAVE_ROOT, "可视化结果")
    os.makedirs(vis_dir, exist_ok=True)
    layer_vis_dir = os.path.join(vis_dir, "中间层激活")
    os.makedirs(layer_vis_dir, exist_ok=True)
    feature_vis_dir = os.path.join(vis_dir, "特征重要性")
    os.makedirs(feature_vis_dir, exist_ok=True)

    # 1. 读取标签并自动识别指标
    labels_df = pd.read_excel(label_file=LABEL_FILE, engine='openpyxl')
    labels_df['imagename'] = labels_df['imagename'].astype(str)
    metric_cols = [col for col in labels_df.columns if col != 'imagename']
    num_metrics = len(metric_cols)
    print(f"自动识别预测指标：{metric_cols}（共{num_metrics}个）")

    # 2. 数据校验与划分
    valid_df = check_data_validity(labels_df, IMAGE_FOLDER)
    train_val_df, test_df = train_test_split(
        valid_df,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE
    )

    # 准备测试集生成器
    test_gen = data_generator(
        test_df,
        IMAGE_FOLDER,
        TARGET_SIZE,
        BATCH_SIZE,
        shuffle=False
    )
    validation_steps = max(1, len(test_df) // BATCH_SIZE)

    # 3. 超参数搜索
    print("\n开始超参数搜索...")
    input_shape = (*TARGET_SIZE, 3)

    # 创建KerasRegressor包装器
    model = KerasRegressor(
        build_fn=build_model,
        input_shape=input_shape,
        num_metrics=num_metrics,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=1
    )

    # 随机搜索超参数
    random_search = RandomizedSearchCV(
        estimator=model,
        param_distributions=PARAM_GRID,
        n_iter=SEARCH_ITERATIONS,
        cv=KFOLD_SPLITS,
        scoring='neg_mean_squared_error',
        random_state=RANDOM_STATE,
        verbose=1
    )

    # 准备用于超参数搜索的数据
    def prepare_data(df):
        images, labels = [], []
        metric_cols = [col for col in df.columns if col != 'imagename']
        for _, row in df.iterrows():
            img_path = os.path.join(IMAGE_FOLDER, f"{row['imagename']}.jpg")
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.resize(img, TARGET_SIZE) / 255.0
                images.append(img)
                labels.append([row[col] for col in metric_cols])
        return np.array(images), np.array(labels)

    X_search, y_search = prepare_data(train_val_df)

    # 执行超参数搜索
    random_search.fit(X_search, y_search)

    print(f"最佳参数: {random_search.best_params_}")
    print(f"最佳交叉验证分数: {-random_search.best_score_:.4f}")

    # 保存超参数搜索结果
    search_results = pd.DataFrame(random_search.cv_results_)
    search_results.to_excel(os.path.join(SAVE_ROOT, "超参数搜索结果.xlsx"), index=False)

    # 4. 使用最佳参数的模型进行K折交叉验证
    print("\n开始K折交叉验证...")
    kf = KFold(n_splits=KFOLD_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    cv_results = {metric: [] for metric in metric_cols}

    # 训练最终模型并进行交叉验证
    fold = 1
    for train_idx, val_idx in kf.split(train_val_df):
        print(f"\n----- 第 {fold} 折 -----")

        # 划分训练集和验证集
        fold_train_df = train_val_df.iloc[train_idx]
        fold_val_df = train_val_df.iloc[val_idx]

        # 创建生成器
        train_gen = data_generator(
            fold_train_df,
            IMAGE_FOLDER,
            TARGET_SIZE,
            BATCH_SIZE,
            shuffle=True
        )
        val_gen = data_generator(
            fold_val_df,
            IMAGE_FOLDER,
            TARGET_SIZE,
            BATCH_SIZE,
            shuffle=False
        )

        steps_per_epoch = max(1, len(fold_train_df) // BATCH_SIZE)
        val_steps = max(1, len(fold_val_df) // BATCH_SIZE)

        # 构建模型
        best_model = build_model(
            input_shape=input_shape,
            num_metrics=num_metrics, **random_search.best_params_
        )

        # 训练模型
        best_model.fit(
            train_gen,
            steps_per_epoch=steps_per_epoch,
            epochs=EPOCHS,
            validation_data=val_gen,
            validation_steps=val_steps,
            verbose=1
        )

        # 评估模型
        fold_results, _, _ = evaluate_model(best_model, val_gen, val_steps, metric_cols)

        # 保存每折的结果
        for metric, metrics in fold_results.items():
            cv_results[metric].append(metrics)

        fold += 1

    # 计算交叉验证的平均结果
    print("\n交叉验证平均结果：")
    cv_avg_results = {}
    for metric, results in cv_results.items():
        avg_rmse = np.mean([r['RMSE'] for r in results])
        avg_r2 = np.mean([r['R²'] for r in results])
        avg_mae = np.mean([r['MAE'] for r in results])
        cv_avg_results[metric] = {"平均RMSE": avg_rmse, "平均R²": avg_r2, "平均MAE": avg_mae}
        print(f"{metric}:")
        print(f"  平均RMSE: {avg_rmse:.4f}")
        print(f"  平均R²: {avg_r2:.4f}")
        print(f"  平均MAE: {avg_mae:.4f}\n")

    # 保存交叉验证结果
    cv_results_df = pd.DataFrame()
    for metric, results in cv_avg_results.items():
        for key, value in results.items():
            cv_results_df.at[metric, key] = value
    cv_results_df.to_excel(os.path.join(SAVE_ROOT, "交叉验证结果.xlsx"))

    # 5. 使用最佳参数在整个训练集上训练最终模型
    print("\n训练最终模型...")
    final_model = build_model(
        input_shape=input_shape,
        num_metrics=num_metrics,
        **random_search.best_params_
    )

    # 准备完整训练集生成器
    final_train_gen = data_generator(
        train_val_df,
        IMAGE_FOLDER,
        TARGET_SIZE,
        BATCH_SIZE,
        shuffle=True
    )
    final_steps_per_epoch = max(1, len(train_val_df) // BATCH_SIZE)

    # 训练最终模型
    final_model.fit(
        final_train_gen,
        steps_per_epoch=final_steps_per_epoch,
        epochs=EPOCHS,
        validation_data=test_gen,
        validation_steps=validation_steps
    )

    # 6. 保存最终模型
    model_save_path = os.path.join(SAVE_ROOT, f"{num_metrics}指标最佳模型.keras")
    final_model.save(model_save_path)
    print(f"最终模型已保存至：{model_save_path}")

    # 7. 在测试集上评估最终模型
    print("\n测试集评估结果：")
    test_results, test_labels, test_preds = evaluate_model(
        final_model,
        test_gen,
        validation_steps,
        metric_cols
    )

    for metric, metrics in test_results.items():
        print(f"{metric}:")
        print(f"  RMSE: {metrics['RMSE']:.4f}")
        print(f"  R²: {metrics['R²']:.4f}")
        print(f"  MAE: {metrics['MAE']:.4f}\n")

    # 8. 生成完整预测结果（训练集+测试集）
    def get_full_predictions(df):
        images, labels, img_paths = [], [], []
        metric_cols = [col for col in df.columns if col != 'imagename']
        for _, row in df.iterrows():
            img_path = os.path.join(IMAGE_FOLDER, f"{row['imagename']}.jpg")
            img = cv2.imread(img_path)
            if img is not None:
                img_normalized = cv2.resize(img, TARGET_SIZE) / 255.0
                images.append(img_normalized)
                labels.append([row[col] for col in metric_cols])
                img_paths.append(img_path)
        images = np.array(images)
        labels = np.array(labels)
        preds = final_model.predict(images)
        return images, labels, preds, img_paths

    train_imgs, train_labels, train_preds, _ = get_full_predictions(train_val_df)
    test_imgs, test_labels, test_preds, test_img_paths = get_full_predictions(test_df)

    # 9. 可视化中间层激活
    print("\n生成中间层激活可视化...")
    # 选择一些样本图像进行可视化
    sample_indices = np.random.choice(len(test_imgs), min(VISUALIZATION_SAMPLES, len(test_imgs)), replace=False)

    for i, idx in enumerate(sample_indices):
        img = test_imgs[idx]
        img_name = os.path.basename(test_img_paths[idx]).split('.')[0]

        # 可视化中间层
        visualize_intermediate_layers(
            final_model,
            img,
            LAYER_VISUALIZATION,
            save_path=os.path.join(layer_vis_dir, f"样本_{img_name}_中间层激活.png")
        )

    # 10. 可视化特征重要性（Grad-CAM）
    print("\n生成特征重要性可视化...")
    for i, idx in enumerate(sample_indices):
        img_array = np.expand_dims(test_imgs[idx], axis=0)
        img_original = cv2.imread(test_img_paths[idx])
        img_original = cv2.resize(img_original, TARGET_SIZE)
        img_name = os.path.basename(test_img_paths[idx]).split('.')[0]

        # 为每个指标生成特征重要性图
        for metric_idx, metric_name in enumerate(metric_cols):
            visualize_feature_importance(
                final_model,
                img_original,
                img_array,
                metric_idx,
                metric_name,
                save_path=os.path.join(feature_vis_dir, f"样本_{img_name}_{metric_name}_特征重要性.png")
            )

    # 11. 可视化真实值vs预测值
    plt.figure(figsize=(14, 6))
    # 训练集散点图
    plt.subplot(1, 2, 1)
    for i, metric in enumerate(metric_cols):
        plt.scatter(train_labels[:, i], train_preds[:, i], label=metric, alpha=0.5)
    plt.xlabel("真实值")
    plt.ylabel("预测值")
    plt.title("训练集：真实值 vs 预测值")
    plt.legend()

    # 测试集散点图
    plt.subplot(1, 2, 2)
    for i, metric in enumerate(metric_cols):
        plt.scatter(test_labels[:, i], test_preds[:, i], label=metric, alpha=0.5)
    plt.xlabel("真实值")
    plt.ylabel("预测值")
    plt.title("测试集：真实值 vs 预测值")
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_ROOT, "预测对比图.png"))
    plt.show()

    # 12. 保存结果到Excel
    def create_results_df(df, true_labels, preds):
        data = {"imagename": df['imagename'].values}
        for i, metric in enumerate(metric_cols):
            data[f"Actual_{metric}"] = true_labels[:, i]
            data[f"Predicted_{metric}"] = preds[:, i]
        return pd.DataFrame(data)

    train_df_results = create_results_df(train_val_df, train_labels, train_preds)
    test_df_results = create_results_df(test_df, test_labels, test_preds)

    excel_path = os.path.join(SAVE_ROOT, f"{num_metrics}指标预测结果.xlsx")
    with pd.ExcelWriter(excel_path) as writer:
        train_df_results.to_excel(writer, sheet_name="训练集", index=False)
        test_df_results.to_excel(writer, sheet_name="测试集", index=False)
    print(f"预测结果已保存至：{excel_path}")
    print(f"可视化结果已保存至：{vis_dir}")


if __name__ == "__main__":
    main()
