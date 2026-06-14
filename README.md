# TensorFlow花卉模型训练与LiteRT移动端部署实验报告

# 实验五 实现智能图像分类APP实验报告

**实验项目**：实现智能图像分类APP
**实验环境**：MacBook Pro（Apple M2 芯片）、Android Studio Hedgehog
**实验日期**：2026 年 6 月

---

## 一、实验目的

1. 了解机器学习与深度学习的基础概念，理解图像分类任务的技术原理。

2. 掌握 TensorFlow 与 LiteRT（TensorFlow Lite）的核心定位与应用场景，理解端侧机器学习的技术价值。

3. 了解 TensorFlow Lite Model Maker 方案的原理与现状，明确其废弃原因与适用限制。

4. 按照参考教程，基于 TensorFlow/Keras 完成花卉分类模型的构建、训练、评估全流程。

5. 使用 TensorFlow Lite Converter 将训练好的模型转换为 LiteRT 格式，掌握模型量化的方法与效果。

6. 结合实验四的 Android 花卉识别应用，验证自定义训练模型的推理效果与可用性。

7. 将完整的训练过程 Jupyter Notebook 上传至 GitHub，实现实验成果的共享与归档。

---

## 二、实验环境

### 2\.1 模型训练端环境

|项目|具体配置|
|---|---|
|硬件设备|MacBook Pro（Apple M2 芯片，8GB 统一内存）|
|操作系统|macOS Sonoma 14\.x|
|环境管理工具|Anaconda（Python 3\.11）|
|深度学习框架|TensorFlow 2\.15\.0（tensorflow\-macos \+ tensorflow\-metal GPU 加速）|
|开发工具|Jupyter Notebook 6\.5\.x|
|依赖库|NumPy 1\.24\.x、Matplotlib 3\.7\.x|
|软件源|清华大学 Anaconda 镜像源、清华 PyPI 镜像源|

### 2\.2 移动端验证环境

|项目|具体配置||
|---|---|---|
|开发工具|Android Studio Hedgehog|2023\.1\.1|
|构建系统|Gradle 8\.9 \+ Android Gradle Plugin 8\.7\.0||
|开发语言|Kotlin 1\.9\.x||
|测试设备|Pixel 7 模拟器（Android 14，API 37）||
|相机框架|CameraX 1\.3\.x||
|推理框架|TensorFlow Lite Support 0\.4\.4||

### 2\.3 实验数据集

本实验使用 TensorFlow 官方开源花卉数据集，总计 3670 张 RGB 彩色图片，分为 5 个类别：雏菊（daisy）、蒲公英（dandelion）、玫瑰（roses）、向日葵（sunflowers）、郁金香（tulips）。数据集图片尺寸不一，包含不同光照、角度、背景的样本，适合验证图像分类模型的泛化能力。

---

## 三、实验原理与技术说明

### 3\.1 TensorFlow 与 LiteRT 概述

- **TensorFlow**：谷歌开源的端到端机器学习框架，核心由 “张量（Tensor）” 与 “计算流（Flow）” 组成，以数据流图的形式完成张量的计算与传递，可快速开发、训练各类机器学习模型，支持 CPU、GPU 等多种硬件加速。

- **LiteRT（TensorFlow Lite）**：面向移动设备、微控制器等终端设备的轻量级机器学习推理框架。它通过模型转换、量化压缩、算子优化等技术，将训练好的深度学习模型转换为轻量化格式，在终端设备上实现低延迟、低功耗、离线可用的 AI 推理。

### 3\.2 TensorFlow Lite Model Maker 方案（已废弃）

TensorFlow Lite Model Maker 是谷歌推出的简化版模型训练工具，基于迁移学习封装了完整的训练、评估、导出流程，用户仅需少量代码即可完成自定义数据集的模型训练。
但该方案已停止更新，存在严重的依赖兼容性问题：其依赖的底层库与 Python 3\.10\+、TensorFlow 2\.15 \+ 版本高度不兼容，在新版系统中安装与运行极易出现依赖冲突，因此本实验不再采用该方案，改用更稳定的 TensorFlow/Keras 原生训练方案。

### 3\.3 TensorFlow/Keras \+ TFLite Converter 方案

本实验采用官方推荐的标准流程，分为两大阶段：

1. **桌面端模型训练**：基于 TensorFlow/Keras 完成「数据准备 → 模型构建 → 模型编译 → 模型训练 → 模型评估 → 模型保存」全流程。

2. **移动端模型转换**：使用 TensorFlow Lite Converter 将 Keras 模型转换为`.tflite`格式，并通过量化技术压缩模型体积，适配 LiteRT 端侧推理。

### 3\.4 迁移学习与 MobileNetV2

本实验使用迁移学习技术，基于在 ImageNet 数据集上预训练的 MobileNetV2 网络进行二次开发：

- 冻结底层特征提取网络，复用预训练模型学习到的通用视觉特征；

- 仅重新训练顶部的全连接分类层，大幅减少训练数据量与训练时长，同时保证较高的识别精度。
MobileNetV2 采用深度可分离卷积与倒残差结构，参数量与计算量远低于常规 CNN，非常适合移动端部署。

### 3\.5 模型量化技术

实验采用**动态范围量化**方案：将模型权重从 32 位浮点数转换为 8 位整数，推理时动态反量化为浮点计算。该方案无需校准数据集，实现简单，可将模型体积缩小约 75%，同时精度损失极小，是移动端部署的首选方案。

---

## 四、实验步骤与结果

### 4\.1 训练环境搭建（M2 芯片 Mac 专属）

#### 4\.1\.1 Conda 镜像源与 SSL 配置

由于境外网络访问受限与系统 SSL 证书异常，首先配置国内镜像源并关闭 SSL 验证，确保依赖包正常下载。
打开终端依次执行命令：

```bash
# 清除原有通道配置
conda config --remove-key channels

# 关闭SSL证书验证，解决X509证书错误
conda config --set ssl_verify false

# 添加清华大学Anaconda镜像源
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge/
conda config --set show_channel_urls yes
```

#### 4\.1\.2 创建实验环境并安装依赖

M2 芯片需安装原生适配的 TensorFlow 版本，通过 conda 创建独立环境避免冲突：

```bash
# 创建Python 3.11实验环境
conda create -n tf-flower python=3.11 -y

# 激活环境
conda activate tf-flower

# 安装M2芯片GPU加速版TensorFlow
pip install tensorflow-macos tensorflow-metal --trusted-host pypi.tuna.tsinghua.edu.cn

# 安装其他实验依赖
pip install matplotlib numpy jupyter --trusted-host pypi.tuna.tsinghua.edu.cn
```


#### 4\.1\.3 启动 Jupyter Notebook

终端执行命令启动服务：

```bash
jupyter notebook
```

浏览器自动打开 Jupyter 主页，点击右上角`New` → `Python 3 (ipykernel)`创建实验笔记本。

<img width="906" height="88" alt="cd82f963f51ab6392990524d9ef83896" src="https://github.com/user-attachments/assets/0c92e844-cffc-4dc5-8155-fd4e80b66775" />


### 4\.2 数据集加载与预处理

#### 4\.2\.1 导入库与参数配置

第一个单元格导入依赖并设置全局参数：

```python
import tarfile
from pathlib import Path
import numpy as np
import tensorflow as tf

# 官方花卉数据集下载地址
FLOWER_URL = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
DATA_DIR = None          # 设为None自动下载官方数据集
EXPORT_DIR = "exported_flower_model"  # 模型导出目录
EPOCHS = 5               # 训练轮次
BATCH_SIZE = 32          # 批次大小
IMAGE_SIZE = 224         # 模型输入图片尺寸
LEARNING_RATE = 1e-3     # 学习率
QUANTIZATION = "dynamic" # 量化方式：动态量化
SEED = 123               # 随机种子，保证实验可复现

print("TensorFlow 版本:", tf.__version__)
```
<img width="1292" height="376" alt="894fd750d31014c633ee18c6ec54e86e" src="https://github.com/user-attachments/assets/739f4663-aba5-4875-9505-b70939ff1e69" />


#### 4\.2\.2 数据集加载函数实现

编写函数完成数据集的自动下载、解压、划分与优化：

```python
def load_flower_datasets(data_dir, image_size, batch_size, seed):
    # 自动下载并解压数据集
    if data_dir is None:
        archive_path = tf.keras.utils.get_file(
            "flower_photos.tgz", FLOWER_URL, extract=False,
        )
        archive_path = Path(archive_path)
        # 检查缓存，避免重复解压
        candidates = [
            archive_path.parent / "flower_photos",
            archive_path.parent / "flower_photos_extracted" / "flower_photos",
        ]
        data_dir = next((p for p in candidates if p.exists()), None)
        if data_dir is None:
            with tarfile.open(archive_path, "r:gz") as tar:
                tar.extractall(archive_path.parent / "flower_photos_extracted")
            data_dir = archive_path.parent / "flower_photos_extracted" / "flower_photos"
    else:
        data_dir = Path(data_dir)

    # 按文件夹名生成标签，划分训练集与验证集
    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir, validation_split=0.2, subset="training", seed=seed,
        image_size=(image_size, image_size), batch_size=batch_size,
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir, validation_split=0.2, subset="validation", seed=seed,
        image_size=(image_size, image_size), batch_size=batch_size,
    )
    class_names = train_ds.class_names

    # 将验证集拆分为验证集与测试集
    val_batches = int(tf.data.experimental.cardinality(val_ds).numpy())
    test_ds = val_ds.take(val_batches // 2)
    val_ds = val_ds.skip(val_batches // 2)

    # 开启缓存、预取优化，提升训练速度
    autotune = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000, seed=seed).prefetch(autotune)
    val_ds = val_ds.cache().prefetch(autotune)
    test_ds = test_ds.cache().prefetch(autotune)
    return train_ds, val_ds, test_ds, class_names
```

#### 4\.2\.3 数据集验证

调用函数加载数据集并输出类别信息：

```python
train_ds, val_ds, test_ds, class_names = load_flower_datasets(
    DATA_DIR, IMAGE_SIZE, BATCH_SIZE, SEED
)
print("类别数量:", len(class_names))
print("类别名称:", class_names)
```

运行结果：输出 5 个类别，顺序为`daisy, dandelion, roses, sunflowers, tulips`，与数据集文件夹结构一致。

<img width="1640" height="434" alt="26ddceaabf7f43887268506d7a6998bf" src="https://github.com/user-attachments/assets/2eca12f9-2868-47d5-b8bb-d14f48743066" />


### 4\.3 迁移学习模型构建

#### 4\.3\.1 模型构建函数

基于 MobileNetV2 构建迁移学习模型，冻结基础网络，仅训练分类头：

```python
def build_model(num_classes, image_size, learning_rate):
    # 定义输入层
    inputs = tf.keras.Input(shape=(image_size, image_size, 3), name="image")
    # MobileNetV2专用预处理
    x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
    
    # 加载预训练MobileNetV2，仅保留特征提取部分
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(image_size, image_size, 3),
        include_top=False, weights="imagenet", pooling="avg",
    )
    
    # 冻结基础模型参数
    base_model.trainable = False
    x = base_model(x, training=False)
    x = tf.keras.layers.Dropout(0.2)(x)
    
    # 5分类输出层，softmax输出概率
    outputs = tf.keras.layers.Dense(
        num_classes, activation="softmax", name="predictions"
    )(x)
    model = tf.keras.Model(inputs, outputs)

    # 编译模型
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )
    return model
```


#### 4\.3\.2 模型结构查看

创建模型并打印网络结构：

```python
model = build_model(len(class_names), IMAGE_SIZE, LEARNING_RATE)
model.summary()
```

运行结果：模型总参数量约 226 万，其中可训练参数仅 6405 个，绝大多数参数来自冻结的 MobileNetV2，符合迁移学习特征。

<img width="2232" height="1118" alt="66e34d06116619ca0e1eb4429d39de77" src="https://github.com/user-attachments/assets/936557f5-02a2-4384-a706-1b4defe54c85" />


### 4\.4 模型训练与评估

#### 4\.4\.1 模型训练

调用`fit`方法启动训练，同步在验证集上评估性能：

```python
history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS)
```

训练过程逐轮输出损失与准确率，随着轮次增加，损失持续下降，准确率持续上升，最终趋于收敛。5 个 epoch 训练耗时约 2 分钟（M2 芯片 GPU 加速）。

<img width="2250" height="708" alt="19b4e9e33a79e3086a46f9aa69732c4d" src="https://github.com/user-attachments/assets/4ea0e621-3206-46d0-8742-bb6a6d2df91d" />


#### 4\.4\.2 测试集评估

使用未参与训练的测试集评估模型泛化能力：

```python
loss, accuracy = model.evaluate(test_ds)
print(f"测试集损失: {loss:.4f}, 测试集准确率: {accuracy:.4f}")
```

实验结果：测试集准确率达到 90\.8%，证明模型泛化能力良好，无明显过拟合。

<img width="1992" height="910" alt="f7688a6788e4eb3c9e4d03368637df11" src="https://github.com/user-attachments/assets/13465e3a-1daa-4422-8028-60e864795031" />


### 4\.5 TFLite 模型转换与导出

#### 4\.5\.1 模型转换函数

编写通用转换函数，支持多种量化方式，本实验使用动态范围量化：

```python
def convert_to_tflite(model, quantization, representative_ds):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    if quantization == "dynamic":
        # 动态范围量化
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
    elif quantization == "float16":
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
    elif quantization == "int8":
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        def representative_data_gen():
            for images, _ in representative_ds.take(100):
                for image in images:
                    yield [tf.expand_dims(tf.cast(image, tf.float32), 0)]
        converter.representative_dataset = representative_data_gen
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.uint8
        converter.inference_output_type = tf.uint8
    elif quantization != "none":
        raise ValueError(f"不支持的量化方式: {quantization}")
    return converter.convert()
```

#### 4\.5\.2 模型文件导出

创建导出目录，保存标签文件、Keras 原始模型与量化 TFLite 模型：

```python
export_dir = Path(EXPORT_DIR)
export_dir.mkdir(parents=True, exist_ok=True)

# 保存类别标签文件
labels_path = export_dir / "labels.txt"
labels_path.write_text("\n".join(class_names) + "\n", encoding="utf-8")

# 保存Keras原始模型
keras_path = export_dir / "flower_classifier.keras"
model.save(keras_path)

# 转换并保存TFLite量化模型
tflite_model = convert_to_tflite(model, QUANTIZATION, train_ds)
tflite_path = export_dir / "model.tflite"
tflite_path.write_bytes(tflite_model)

print(f"Keras模型已保存: {keras_path.resolve()}")
print(f"TFLite模型已保存: {tflite_path.resolve()}")
print(f"标签文件已保存: {labels_path.resolve()}")
```

运行结果：在`exported_flower_model`目录下生成三个文件，其中`model.tflite`体积约 2\.8MB，相比原始 Keras 模型缩小 68\.5%。


#### 4\.5\.3 TFLite 模型冒烟测试

加载 TFLite 模型并使用测试集图片推理，验证转换后功能正常：

```python
def smoke_test_tflite(tflite_path, test_ds, class_names):
    interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    images, labels = next(iter(test_ds.unbatch().batch(8)))
    input_data = tf.cast(images, input_details["dtype"]).numpy()

    if input_details["dtype"] == np.uint8:
        scale, zero_point = input_details["quantization"]
        if scale:
            input_data = images.numpy() / scale + zero_point
            input_data = np.clip(input_data, 0, 255).astype(np.uint8)

    predictions = []
    for image in input_data:
        interpreter.set_tensor(input_details["index"], np.expand_dims(image, 0))
        interpreter.invoke()
        predictions.append(interpreter.get_tensor(output_details["index"])[0])

    predicted_ids = np.argmax(np.asarray(predictions), axis=1)
    print("\nTFLite模型测试结果：")
    for expected, predicted in zip(labels.numpy()[:5], predicted_ids[:5]):
        print(f"真实类别: {class_names[expected]:<10}  预测类别: {class_names[predicted]:<10}")

smoke_test_tflite(tflite_path, test_ds, class_names)
```


### 4\.6 Android 端模型验证

#### 4\.6\.1 模型文件替换

1. 将训练生成的`model.tflite`复制到 Android 项目`start/src/main/ml/`目录，重命名为`flower_model.tflite`。

2. 执行`Build` → `Make Project`，让 Android Studio 自动生成模型包装类。

#### 4\.6\.2 推理逻辑适配

由于自定义训练的 TFLite 模型不含 Metadata 元数据，自动生成的模型接口与官方示例不同，需手动适配推理逻辑：

- 新增`TensorBuffer`导入，将`TensorImage`转换为张量缓冲区输入；

- 手动定义与训练顺序一致的类别列表，用于输出结果映射；

- 从输出张量读取原始概率数组，手动转换为标签与置信度并排序。

核心推理代码：

```kotlin
private val CLASS_NAMES = listOf("daisy", "dandelion", "roses", "sunflowers", "tulips")

override fun analyze(imageProxy: ImageProxy) {
    val items = mutableListOf<Recognition>()
    try {
        val tfImage = TensorImage.fromBitmap(toBitmap(imageProxy))
        val inputBuffer = tfImage.tensorBuffer
        val outputs = flowerModel.process(inputBuffer)
        val probabilities = outputs.outputFeature0AsTensorBuffer.floatArray

        val recognitionList = CLASS_NAMES.mapIndexed { index, label ->
            Recognition(label, probabilities[index])
        }.sortedByDescending { it.confidence }
         .take(MAX_RESULT_DISPLAY)

        items.addAll(recognitionList)
    } catch (e: Exception) {
        Log.e(TAG, "图像推理失败", e)
    } finally {
        imageProxy.close()
    }
    listener(items.toList())
}
```

#### 4\.6\.3 编译与运行

执行`Clean Project` → `Rebuild Project`，项目编译成功。启动 Pixel 7 模拟器运行应用，授予相机权限后，将摄像头对准玫瑰花测试图片，应用成功识别出
类别，置信度达 85% 以上；更换其他花卉图片均可正确识别。

<img width="432" height="830" alt="93515ac22abe836f090e9ac8e31dbba6" src="https://github.com/user-attachments/assets/4941be8b-a3e1-4b7d-a03a-ec3f283f71d9" />

<img width="290" height="626" alt="b78b16dd9c64d56bad28f7353a79a131" src="https://github.com/user-attachments/assets/8aadcc80-3564-4417-a897-8ed2abc0dcb9" />

### 4\.7 Jupyter Notebook 上传 GitHub

1. 将训练笔记本保存为`flower_model_training.ipynb`，复制到 Android 项目根目录。

2. 终端执行 Git 命令提交并推送：

```bash
cd /Users/unnn/AndroidStudioProjects/TFLClassify-main
git add flower_model_training.ipynb
git commit -m "添加TensorFlow花卉模型训练Notebook"
git push origin main
```

3. 打开 GitHub 仓库页面，确认 Notebook 文件上传成功，获取共享链接完成实验要求。

---

## 五、实验遇到的问题与解决方案

### 问题 1：Conda 环境 SSL 证书错误

**现象**：执行`conda create`时抛出`SSLError: X509: NO_CERTIFICATE_OR_CRL_FOUND`，无法下载依赖。
**原因**：macOS 系统证书链异常，国内网络下 HTTPS 校验失败。
**解决方案**：执行`conda config --set ssl_verify false`临时关闭 SSL 验证，配合清华镜像源完成安装。

### 问题 2：自定义模型替换后编译失败

**现象**：报错`Type mismatch: inferred type is TensorImage! but TensorBuffer was expected`与`Unresolved reference: probabilityAsCategoryList`。
**原因**：官方模型包含 Metadata，自动生成高级 API；自定义模型无 Metadata，仅支持基础张量接口。
**解决方案**：修改推理逻辑，手动完成张量转换、概率读取、标签映射与排序。

### 问题 3：Recognition 类属性名不匹配

**现象**：使用`it.score`排序时报错`Unresolved reference: score`。
**原因**：项目中`Recognition`数据类的置信度字段名为`confidence`。
**解决方案**：将代码中所有`score`替换为`confidence`，与数据类定义保持一致。

### 问题 4：TFLite 库命名空间重复警告

**现象**：编译时警告`Namespace 'org.tensorflow.lite.support' is used in multiple modules`。
**原因**：TensorFlow 官方依赖库自身的命名空间重复问题。
**解决方案**：该警告不影响编译与运行，直接忽略即可。

### 问题 5：模拟器摄像头黑屏

**现象**：模拟器设置 Webcam0 后相机界面黑屏。
**原因**：系统未授予 Android Studio 摄像头权限，或摄像头被其他应用占用。
**解决方案**：开启系统摄像头桌面应用权限，关闭占用摄像头的软件，冷启动模拟器。

---

## 六、实验总结

### 6\.1 实验成果

本次实验完整完成了全部要求内容：

1. 掌握了 TensorFlow 与 LiteRT 的基础概念与技术定位；

2. 了解了 Model Maker 方案的原理与废弃现状，选用了稳定的 Keras 原生训练方案；

3. 基于迁移学习完成了 5 分类花卉模型的训练，测试集准确率达 90\.8%；

4. 通过 TFLite Converter 完成了模型动态量化转换，模型体积压缩至 2\.8MB；

5. 在 Android 应用中成功集成本次训练的模型，实现了实时相机识别功能；

6. 将完整的 Jupyter Notebook 上传至 GitHub，完成了实验成果共享。

### 6\.2 实验收获

通过本次实验，完整掌握了端侧机器学习从训练到部署的全流程，深入理解了迁移学习、模型量化、端侧推理等核心技术；同时积累了 M2 芯片环境适配、依赖冲突解决、接口适配等实战经验，提升了问题排查与工程实践能力。

---
