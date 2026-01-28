# 多模态情感分类实验

## 项目简介

这是一个基于深度学习的多模态情感分类系统，结合文本和图像信息进行情感分类。项目支持三种情感类别：正面（positive）、中性（neutral）、负面（negative）。

## 项目结构

```
.
├── data/
│   ├── data/                    # 原始数据文件夹
│   │   ├── {guid}.txt          # 文本文件（GBK编码）
│   │   └── {guid}.jpg          # 图像文件
│   ├── train.txt               # 训练标签文件（4000个样本）
│   └── test_without_label.txt  # 测试集文件
├── bert_model/                  # BERT预训练模型（可选）
├── models/                      # 模型保存目录
├── results/                     # 结果保存目录
├── task.ipynb                   # 主实验笔记本
└── README.md                    # 项目说明
```

## 环境要求

### 硬件要求
- GPU: NVIDIA GPU（推荐，但CPU也可运行）
- 内存: ≥ 8GB RAM
- 显存: ≥ 4GB（使用完整模型时）

### 软件要求
- Python 3.7+
- PyTorch 1.8+
- Transformers 4.0+

### Python依赖
```bash
pip install torch torchvision transformers
pip install pandas numpy matplotlib pillow scikit-learn tqdm
```

## 快速开始

### 1. 数据准备
确保数据文件结构如下：
- `data/data/` 文件夹包含 `.txt` 和 `.jpg` 文件
- `data/train.txt` 包含训练标签（格式：guid,tag）
- `data/test_without_label.txt` 包含测试集GUID

### 2. 运行实验
打开 `task.ipynb` 笔记本，按顺序执行代码格：

1. **环境设置**（第1格）：检查GPU和安装依赖
2. **数据预处理**（第2-3格）：初始化文本和图像处理器
3. **模型定义**（第4格）：定义多模态融合模型
4. **训练函数**（第5格）：定义训练和验证函数
5. **主训练流程**（第6格）：训练模型并保存最佳结果
6. **测试预测**（第7格）：对测试集进行预测
7. **消融实验**（第8格）：比较不同模态的效果
8. **主执行流程**（第9格）：运行完整实验

### 3. 运行修复版本（推荐）
如果原代码运行有问题，可以直接运行第九格中的 **修复版完整实验**，它包含：
- 自动错误处理
- 模型加载的fallback机制
- 小规模数据测试模式
- 完整的训练验证循环

## 模型架构

### 特征提取器
1. **文本特征提取**：BERT模型（本地或在线）
   - 输入：文本序列（最大长度128）
   - 输出：768维文本特征

2. **图像特征提取**：ResNet-18
   - 输入：224×224 RGB图像
   - 输出：512维图像特征

### 融合策略
支持三种融合方法：
1. **早期融合（Early Fusion）**：特征级拼接
2. **中期融合（Middle Fusion）**：注意力机制
3. **晚期融合（Late Fusion）**：决策级融合（默认）

### 分类头
- 文本分类器：BERT特征 → 分类结果
- 图像分类器：ResNet特征 → 分类结果
- 融合分类器：融合特征 → 最终分类结果

## 实验结果

### 性能指标
- 仅文本模态准确率：XX.XX%
- 仅图像模态准确率：XX.XX%
- 多模态融合准确率：XX.XX%
- 多模态相对提升：+X.XX%

### 训练日志
```
Epoch 1/10
训练集 - Loss: 1.0234, Acc: 45.67%
验证集 - Loss: 0.9876, Acc: 48.92%

Epoch 2/10
训练集 - Loss: 0.8765, Acc: 58.34%
验证集 - Loss: 0.8456, Acc: 56.78%
...
```

## 配置选项

### 数据相关
```python
# 样本数量限制（用于快速测试）
max_samples = 1000  # 可调整为完整数据量

# 批处理大小
batch_size = 16  # 根据GPU内存调整

# 数据增强
augment = True  # 训练时启用数据增强
```

### 模型相关
```python
# 融合方法选择
fusion_method = 'late'  # 'early', 'middle', 'late'

# 文本序列长度
max_length = 128  # BERT输入长度

# 图像尺寸
img_size = 224  # ResNet输入尺寸
```

### 训练相关
```python
# 学习率
learning_rate = 2e-5

# 训练轮数
num_epochs = 10

# 早停耐心值
patience = 3
```

## 故障排除

### 常见问题

1. **BERT模型加载失败**
   ```
   解决方案：
   - 确保bert_model/文件夹存在且包含BERT模型
   - 或代码会自动使用在线模型
   - 或使用随机初始化的BERT（性能会下降）
   ```

2. **内存不足**
   ```
   解决方案：
   - 减少batch_size（如从32改为16）
   - 限制max_samples（如从4000改为1000）
   - 使用更小的图像尺寸（如从224改为128）
   ```

3. **多进程错误**
   ```
   解决方案：
   - 设置num_workers=0（Windows系统）
   - 设置TOKENIZERS_PARALLELISM=false环境变量
   ```

4. **编码问题**
   ```
   解决方案：
   - 代码自动尝试多种编码（utf-8, gbk, latin-1等）
   - 如果仍有问题，检查文本文件编码格式
   ```

### 调试模式
如需调试，可以：
1. 设置`max_samples=50`快速测试流程
2. 添加`debug=True`参数打印更多信息
3. 单独运行每个代码格检查输出

## 输出文件

### 模型文件
- `best_model.pth`：原版最佳模型
- `fixed_best_model.pth`：修复版最佳模型

### 结果文件
- `submission.txt`：原版预测结果
- `fixed_submission.txt`：修复版预测结果
- `ablation_results.txt`：消融实验结果

### 日志文件
- 训练过程中的准确率和损失记录
- 验证集的分类报告
- 模型参数统计

## 扩展功能

### 1. 添加新数据集
修改`MultiModalDataset`类以支持不同的数据格式：
```python
# 自定义数据读取逻辑
def load_custom_data(self, label_file):
    # 实现自定义数据加载
    pass
```

### 2. 尝试不同模型
```python
# 更换文本编码器
from transformers import RobertaModel, DistilBertModel

# 更换图像编码器
from torchvision.models import efficientnet_b0, vit_b_16
```

### 3. 实验不同融合策略
```python
# 实现新的融合方法
def custom_fusion(self, text_features, image_features):
    # 实现自定义融合逻辑
    return fused_features
```

## 项目贡献

### 开发路线图
- [ ] 添加更多预训练模型支持
- [ ] 实现交叉验证
- [ ] 添加模型解释性分析
- [ ] 支持实时预测API
- [ ] 添加Web演示界面

### 贡献指南
1. Fork本仓库
2. 创建功能分支
3. 提交更改
4. 推送到分支
5. 创建Pull Request