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
    model.to(device)
    model.train()

    total_loss = 0

    for encoder_inputs, decoder_inputs, labels in tqdm(dataloader, desc='Training'):
        encoder_inputs = encoder_inputs.to(device)  # encoder_inputs.shape: [batch_size, src_len]
        decoder_inputs = decoder_inputs.to(device)  # decoder_inputs.shape: [batch_size, tgt_len]
        labels = labels.to(device)  # labels.shape: [batch_size, tgt_len]

        # 前向传播
        src_key_padding_mask = (encoder_inputs == zh_tokenizer.pad_token_id)
        tgt_mask = model.transformer.generate_square_subsequent_mask(sz=decoder_inputs.shape[1], device=device)
        outputs = model(
            src=encoder_inputs,
            src_key_padding_mask=src_key_padding_mask,
            tgt=decoder_inputs,
            tgt_mask=tgt_mask
        )
        # outputs.shape: [batch_size, tgt_len, en_vocab_size]

        loss = loss_fn(outputs.reshape(-1, outputs.shape[-1]), labels.reshape(-1))

        # 反向传播
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def train():
    # 设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 分词器
    zh_tokenizer = ChineseTokenizer.from_vocab(MODELS_DIR / 'zh_vocab.txt')
    en_tokenizer = EnglishTokenizer.from_vocab(MODELS_DIR / 'en_vocab.txt')

    # 模型
    model = TranslationModel(zh_vocab_size=zh_tokenizer.vocab_size, zh_pad_index=zh_tokenizer.pad_token_id,
                             en_vocab_size=en_tokenizer.vocab_size, en_pad_index=en_tokenizer.pad_token_id)

    # 数据
    dataloader = get_dataloader(train=True)

    # 损失函数
    loss_fn = CrossEntropyLoss(ignore_index=en_tokenizer.pad_token_id)

    # 优化器
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)

    # writer
    writer = SummaryWriter(log_dir=LOGS_DIR / time.strftime("%Y-%m-%d_%H-%M-%S"))

    # 训练循环
    best_loss = float('inf')
    for epoch in range(1, NUM_EPOCHS + 1):
        print(f'========== Epoch {epoch}/{NUM_EPOCHS} ==========')
        avg_loss = train_one_epoch(model, dataloader, zh_tokenizer, en_tokenizer, loss_fn, optimizer, device)
        print(f'Average Loss: {avg_loss:.4f}')

        # 记录日志
        writer.add_scalar('loss', avg_loss, epoch)

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), MODELS_DIR / 'best_model.pt')
            print('Saved best model.')

    writer.close()



if __name__ == '__main__':
    train()
