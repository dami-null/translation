# Transformer 中英翻译项目

这是一个基于 PyTorch `nn.Transformer` 实现的中文到英文翻译项目。项目包含数据预处理、词表构建、Dataset/DataLoader、Transformer 编码解码模型、训练、推理和 BLEU 评估流程。

## 项目结构

```text
translation/
├── data/
│   ├── raw/
│   │   └── cmn.txt                 # 原始中英平行语料
│   └── processed/
│       ├── train.jsonl             # 处理后的训练集
│       └── test.jsonl              # 处理后的测试集
├── logs/                           # TensorBoard 训练日志
├── models/
│   ├── best_model.pt               # 训练过程中保存的最佳模型权重
│   ├── en_vocab.txt                # 英文词表
│   └── zh_vocab.txt                # 中文词表
├── src/
│   ├── config.py                   # 路径、训练参数、模型参数
│   ├── tokenizer.py                # 中文/英文分词器和词表逻辑
│   ├── process.py                  # 原始数据预处理
│   ├── dataset.py                  # Dataset、DataLoader、padding
│   ├── model.py                    # Transformer 翻译模型
│   ├── train.py                    # 模型训练
│   ├── predict.py                  # 模型推理
│   └── evaluate.py                 # BLEU 评估
└── test/
    └── test-tensorboard.py         # TensorBoard 测试代码
```

## 环境依赖

当前项目使用的主要依赖包括：

```text
torch
pandas
scikit-learn
nltk
tqdm
tensorboard
```

如果使用当前本机 Conda 环境，可通过下面的解释器运行：

```bash
/opt/anaconda3/envs/nlp/bin/python
```

由于项目源码使用 `from src.xxx import ...` 这种导入方式，命令行运行时建议在项目根目录设置 `PYTHONPATH`：

```bash
cd /Users/xxxx/translation
PYTHONPATH=. /opt/anaconda3/envs/nlp/bin/python src/predict.py
```

## 数据格式

原始数据位于：

```text
data/raw/cmn.txt
```

文件是制表符分隔的中英平行语料，代码只读取前两列：

```text
英文句子\t中文句子\t其他来源信息
```

预处理后的数据保存为 JSON Lines：

```text
data/processed/train.jsonl
data/processed/test.jsonl
```

每一行的格式类似：

```json
{"en":[2,3459,3605,1557,6512,755,759,3478,5256,3],"zh":[668,1571,82,1928,2604,870,53,2482,1813,1463]}
```

其中：

- `zh` 是中文输入 token id 序列。
- `en` 是英文目标 token id 序列，包含 `<sos>` 和 `<eos>`。

## 核心流程

### 1. 数据预处理

入口文件：

```text
src/process.py
```

运行：

```bash
PYTHONPATH=. /opt/anaconda3/envs/nlp/bin/python src/process.py
```

主要步骤：

1. 读取 `data/raw/cmn.txt`。
2. 用 `train_test_split` 划分训练集和测试集。
3. 基于训练集构建中文词表和英文词表。
4. 将句子编码成 token id。
5. 保存 `train.jsonl` 和 `test.jsonl`。

中文分词逻辑是按字切分：

```python
return list(sentence)
```

英文分词逻辑使用 NLTK 的 `TreebankWordTokenizer`。

### 2. 构造数据集

入口文件：

```text
src/dataset.py
```

`TranslationDataset.__getitem__()` 返回三部分：

```text
encoder_input: 中文输入
decoder_input: 英文输入，去掉最后一个 token
label: 英文标签，去掉第一个 token
```

例如英文序列为：

```text
<sos> I like you . <eos>
```

训练时会拆成：

```text
decoder_input: <sos> I like you .
label:         I like you . <eos>
```

这样模型在每个位置学习预测下一个英文 token。

### 3. 模型结构

入口文件：

```text
src/model.py
```

模型主要组件：

- 中文 `Embedding`
- 英文 `Embedding`
- 正弦/余弦位置编码 `PositionEncoding`
- PyTorch `nn.Transformer`
- 输出到英文词表大小的 `Linear`

模型参数在 `src/config.py` 中配置：

```python
DIM_MODEL = 256
NUM_HEAD = 4
NUM_ENCODER_LAYERS = 2
NUM_DECODER_LAYERS = 2
DIM_FEEDFORWARD = 4 * DIM_MODEL
```

编码阶段：

```text
中文 token id -> 中文 Embedding -> 位置编码 -> Transformer Encoder -> memory
```

解码阶段：

```text
英文 token id -> 英文 Embedding -> 位置编码 -> Transformer Decoder -> Linear -> 英文词表 logits
```

### 4. 模型训练

入口文件：

```text
src/train.py
```

运行：

```bash
PYTHONPATH=. /opt/anaconda3/envs/nlp/bin/python src/train.py
```

训练配置：

```python
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
NUM_EPOCHS = 30
```

训练时使用：

```python
loss_fn = CrossEntropyLoss(ignore_index=en_tokenizer.pad_token_id)
```

含义是计算英文 token 分类损失时忽略 `<pad>` 位置，避免 padding 参与训练目标。

每个 epoch 会记录平均 loss 到 TensorBoard，并在 loss 降低时保存最佳权重：

```text
models/best_model.pt
```

查看 TensorBoard：

```bash
tensorboard --logdir logs
```

### 5. 模型推理

入口文件：

```text
src/predict.py
```

运行：

```bash
PYTHONPATH=. /opt/anaconda3/envs/nlp/bin/python src/predict.py
```

推理逻辑是自回归生成：

1. 先对中文句子编码，得到 `memory`。
2. 解码器输入从 `<sos>` 开始。
3. 每一步取最后一个位置的 logits。
4. 使用 `argmax` 选择概率最高的 token。
5. 拼接到解码器输入后继续生成。
6. 遇到 `<eos>` 或达到 `MAX_STEPS` 停止。

生成最大步数在 `src/config.py` 中配置：

```python
MAX_STEPS = 100
```

### 6. 模型评估

入口文件：

```text
src/evaluate.py
```

运行：

```bash
PYTHONPATH=. /opt/anaconda3/envs/nlp/bin/python src/evaluate.py
```

评估指标使用 NLTK 的 `corpus_bleu`。流程是：

1. 读取测试集。
2. 调用 `predict_batch()` 生成预测 token 序列。
3. 从 label 中截取到 `<eos>` 前。
4. 计算整个测试集的 BLEU 分数。

## 设备选择

`src/train.py` 中的 `get_device()` 会按下面优先级选择设备：

```text
cuda -> mps -> cpu
```

也就是：

- 有 NVIDIA GPU 时使用 `cuda`。
- Mac M 系列芯片可用时使用 `mps`。
- 否则使用 `cpu`。

## Mac MPS 注意事项

当前 PyTorch 在 MPS 上运行 `nn.Transformer` 并携带 `src_key_padding_mask` 时，可能触发 nested tensor 相关算子不支持的问题：

```text
aten::_nested_tensor_from_mask_left_aligned is not currently implemented for the MPS device
```

项目在 `src/model.py` 中关闭了 Transformer Encoder 的 nested tensor 路径：

```python
self.transformer.encoder.enable_nested_tensor = False
self.transformer.encoder.use_nested_tensor = False
```

其中 `use_nested_tensor` 是当前 PyTorch forward 过程中实际判断的开关。

如果仍然遇到 MPS 不支持的算子，也可以临时使用 CPU：

```python
device = torch.device("cpu")
```

或者设置环境变量让 PyTorch 自动回退到 CPU：

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1
```

## 常见问题

### ModuleNotFoundError: No module named 'src'

原因是命令行直接运行 `src/predict.py` 时，项目根目录没有加入 Python 模块搜索路径。

解决方式是在项目根目录运行，并加上 `PYTHONPATH=.`：

```bash
cd /Users/xxxx/translation
PYTHONPATH=. /opt/anaconda3/envs/nlp/bin/python src/predict.py
```

### padding token 为什么要忽略

batch 中句子长度不同，需要补 `<pad>` 到相同长度。`<pad>` 只是占位符，不是真实翻译目标，所以训练 loss 中需要忽略：

```python
CrossEntropyLoss(ignore_index=en_tokenizer.pad_token_id)
```

### 中文输入 padding 应该用哪个 tokenizer

中文 encoder 输入应该使用中文 tokenizer 的 pad id：

```python
padding_value=zh_tokenizer.pad_token_id
```

英文 decoder 输入和 label 才使用英文 tokenizer 的 pad id。

## 推荐运行顺序

第一次完整运行项目时，建议按下面顺序执行：

```bash
cd /Users/xxxx/translation

# 1. 预处理数据，生成词表和 train/test jsonl
PYTHONPATH=. /opt/anaconda3/envs/nlp/bin/python src/process.py

# 2. 训练模型，生成 models/best_model.pt
PYTHONPATH=. /opt/anaconda3/envs/nlp/bin/python src/train.py

# 3. 推理测试
PYTHONPATH=. /opt/anaconda3/envs/nlp/bin/python src/predict.py

# 4. BLEU 评估
PYTHONPATH=. /opt/anaconda3/envs/nlp/bin/python src/evaluate.py
```

