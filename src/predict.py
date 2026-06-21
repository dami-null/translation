import torch
from torch.nn.utils.rnn import pad_sequence

from src.config import MODELS_DIR, MAX_STEPS
from src.model import TranslationModel
from src.tokenizer import ChineseTokenizer, EnglishTokenizer
from src.train import get_device


def predict_batch(encoder_inputs, model, zh_tokenizer, en_tokenizer, device) -> list[list[int]]:
    """
    批处理（自回归生成）
    :param encoder_inputs: tensor编码器输入 shape: [batch_size, src_len]
    :param model: 模型
    :param zh_tokenizer: 分词器
    :param en_tokenizer: 分词器
    :param device: 设备
    :return: token_ids
    """
    encoder_inputs = encoder_inputs.to(device)

    model.to(device)
    # 把模型切换到“评估 / 推理模式”。
    model.eval()

    # 预测阶段关闭梯度计算
    with torch.no_grad():
        # 编码
        src_key_padding_mask = (encoder_inputs == zh_tokenizer.pad_token_id)
        # memory.shape: [batch_size, src_len, dim_model]
        memory = model.encode(src=encoder_inputs, src_key_padding_mask=src_key_padding_mask)

        # 解码
        batch_size = encoder_inputs.shape[0]
        # 构造decoder_inputs，初始的input为<sos>形状->有几句话（即batch_size）就有几行<sos>（列数为1）----> (batch_size, 1)
        decoder_inputs = torch.full(size=[batch_size, 1], fill_value=en_tokenizer.sos_token_id, device=device)

        # 句子完成标识
        is_finish = torch.full(size=[batch_size], fill_value=False, device=device)

        for step in range(MAX_STEPS):
            # tgt_mask形状由decoder_inputs的形状决定，方阵 shape：decoder_inputs（seq_len）
            tgt_mask = model.transformer.generate_square_subsequent_mask(sz=decoder_inputs.shape[-1], device=device)

            # logits.shape: [batch_size, tgt_len, en_vocab_size]
            logits = model.decode(
                memory=memory,
                tgt=decoder_inputs,
                tgt_mask=tgt_mask,
                memory_key_padding_mask=src_key_padding_mask
            )

            # 切片 只需要tgt_len的最后一个，最后一个包含前面所有的隐藏状态  shape: [batch_size, en_vocab_size]（降维了）
            next_token_logits = logits[:, -1, :]

            # 取出next_token_logits中的最大值（预测概率最高的）的索引
            # shape: [batch_size, 1] ------> 维度由keepdim=True控制 keepdim=False则降维[batch_size]
            next_token_ids = torch.argmax(next_token_logits, dim=-1, keepdim=True)

            # 拼接 输入 + 新预测的 = new decoder_inputs  shape: [batch_size, tgt_len]
            decoder_inputs = torch.cat([decoder_inputs, next_token_ids], dim=-1)

            # 判断eos  注意：不一定只有一个句子（batch_size），每个长短不一，到达eos的时机不同，定义中间状态is_finish
            # squeeze(-1)：干掉最后一维 ----> [batch_size]
            # |=：或运算，为了保留true，不会把true替换为false导致继续预测
            is_finish |= next_token_ids.squeeze(-1) == en_tokenizer.eos_token_id

            if is_finish.all():
                break

        # 取出预测结果 --- [[2,3,4,5],[4,5,6,7],[4,5,6,7]]
        token_ids_list = decoder_inputs.tolist()
        batch_result = []
        for token_ids in token_ids_list:
            # 切片：干掉头和尾的sos，eos  注意：不一定每个token_ids都有eos
            if en_tokenizer.eos_token_id in token_ids:
                batch_result.append(token_ids[1:token_ids.index(en_tokenizer.eos_token_id)])
            else:
                batch_result.append(token_ids[1:])
        return batch_result


def predict(text: str | list[str], model, zh_tokenizer, en_tokenizer, device) -> str | list[str]:
    """
    推理
    :param text: 语料
    :param model: 模型
    :param zh_tokenizer: 分词器
    :param en_tokenizer: 分词器
    :param device: 设备
    :return: 预测结果
    """
    # 处理输入
    is_str = isinstance(text, str)
    if is_str:
        text = [text]

    # encode
    indexes_list = [torch.tensor(zh_tokenizer.encode(sentence), dtype=torch.long) for sentence in text]
    # 填充pad token   shape: [batch_size, seq_len]
    encoder_inputs = pad_sequence(indexes_list, batch_first=True, padding_value=zh_tokenizer.pad_token_id)
    batch_result = predict_batch(encoder_inputs, model, zh_tokenizer, en_tokenizer, device)
    # 处理输出
    result = [en_tokenizer.decode(result) for result in batch_result]

    if is_str:
        return result[0]
    return result


if __name__ == '__main__':
    text = ['我喜欢你。', '我喜欢打篮球。', '我不爱吃饭。']

    # 设备
    device = get_device()

    # 分词器
    zh_tokenizer = ChineseTokenizer.from_vocab(MODELS_DIR / 'zh_vocab.txt')
    en_tokenizer = EnglishTokenizer.from_vocab(MODELS_DIR / 'en_vocab.txt')

    # 模型
    model = TranslationModel(zh_vocab_size=zh_tokenizer.vocab_size, zh_pad_index=zh_tokenizer.pad_token_id,
                             en_vocab_size=en_tokenizer.vocab_size, en_pad_index=en_tokenizer.pad_token_id)
    # 加载模型权重
    model.load_state_dict(torch.load(MODELS_DIR / 'best_model.pt'))

    print(predict(text, model, zh_tokenizer, en_tokenizer, device))
