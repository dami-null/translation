import time

import torch
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.config import MODELS_DIR, LEARNING_RATE, NUM_EPOCHS, LOGS_DIR
from src.dataset import get_dataloader
from src.model import TranslationModel
from src.tokenizer import ChineseTokenizer, EnglishTokenizer


def train_one_epoch(model, dataloader, zh_tokenizer, en_tokenizer, loss_fn, optimizer, device):
    """
    一个轮次的训练
    :param model: 模型
    :param dataloader: 数据集
    :param zh_tokenizer: zh_分词器
    :param en_tokenizer: en_分词器
    :param loss_fn: 损失函数
    :param optimizer: 优化器
    :param device: 设备
    :return: 平均损失
    """
    model.to(device)
    # 训练模式
    model.train()

    total_loss = 0

    # tqdm进度条
    for encoder_inputs, decoder_inputs, labels in tqdm(dataloader, desc='Training'):
        # 将数据集放在目标设备上
        # encoder_inputs.shape: [batch_size, src_len]
        encoder_inputs = encoder_inputs.to(device)
        # decoder_inputs.shape: [batch_size, tgt_len]
        decoder_inputs = decoder_inputs.to(device)
        # labels.shape: [batch_size, tgt_len]
        labels = labels.to(device)

        # 前向传播
        # src_key_padding_mask：计算得分矩阵时使pad token对应的列趋近于负无穷 shape同encoder_inputs bool类型的张量
        src_key_padding_mask = (encoder_inputs == zh_tokenizer.pad_token_id)
        # tgt_mask：因果掩码，用来遮住结果 shape：方阵（seq_len）左下角都是0右上角都是负无穷
        tgt_mask = model.transformer.generate_square_subsequent_mask(sz=decoder_inputs.shape[1], device=device)
        # outputs.shape: [batch_size, tgt_len, en_vocab_size]
        outputs = model(
            src=encoder_inputs,
            src_key_padding_mask=src_key_padding_mask,
            tgt=decoder_inputs,
            tgt_mask=tgt_mask
        )
        # 计算损失 input：(batch_size * tgt_len, en_vocab_size), target:(batch_size * tgt_len)
        loss = loss_fn(outputs.reshape(-1, outputs.shape[-1]), labels.reshape(-1))

        # 反向传播：计算梯度
        loss.backward()
        # 更新参数
        optimizer.step()
        # 梯度清零
        optimizer.zero_grad()

        total_loss += loss.item()

    return total_loss / len(dataloader)



def get_device():
    """
    英伟达：cuda
    Mac M1/M2/M3 mps
    啥都没有那就cpu
    :return: device
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def train():
    """
    训练
    :return: None
    """
    # 设备
    device = get_device()
    # 分词器
    zh_tokenizer = ChineseTokenizer.from_vocab(MODELS_DIR / 'zh_vocab.txt')
    en_tokenizer = EnglishTokenizer.from_vocab(MODELS_DIR / 'en_vocab.txt')
    # 模型
    model = TranslationModel(zh_vocab_size=zh_tokenizer.vocab_size, zh_pad_index=zh_tokenizer.pad_token_id,
                             en_vocab_size=en_tokenizer.vocab_size, en_pad_index=en_tokenizer.pad_token_id)
    # 数据集
    dataloader = get_dataloader(train=True)

    # 损失函数：用于多分类任务，衡量模型预测结果和真实类别之间的差距。
    # 内部包含Softmax函数：把一组分数转换成一组概率，而且所有概率加起来等于 1。
    # 计算loss时要屏蔽pad token
    loss_fn = CrossEntropyLoss(ignore_index=en_tokenizer.pad_token_id)

    # 优化器：根据 loss 的梯度，自动更新模型参数，让 loss 越来越小。
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)

    # 创建writer
    writer = SummaryWriter(log_dir=LOGS_DIR / time.strftime("%Y-%m-%d_%H-%M-%S"))

    # 训练循环
    best_loss = float('inf')
    for epoch in range(1, NUM_EPOCHS + 1):
        print(f'========== Epoch {epoch}/{NUM_EPOCHS} ==========')
        avg_loss = train_one_epoch(model, dataloader, zh_tokenizer, en_tokenizer, loss_fn, optimizer, device)
        print(f'Average Loss: {avg_loss:.4f}')

        # 记录日志
        writer.add_scalar('loss', avg_loss, epoch)

        # 保存权重
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), MODELS_DIR / 'best_model.pt')
            print('Saved best model.')

    writer.close()



if __name__ == '__main__':
    train()
