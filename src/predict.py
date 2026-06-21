import torch
from torch.nn.utils.rnn import pad_sequence

from src.config import MODELS_DIR, MAX_STEPS
from src.model import TranslationModel
from src.tokenizer import ChineseTokenizer, EnglishTokenizer


def predict_batch(encoder_inputs, model, zh_tokenizer, en_tokenizer, device) -> list[list[int]]:
    # encoder_inputs.shape:  [batch_size, src_len]
    encoder_inputs = encoder_inputs.to(device)

    model.to(device)
    model.eval()

    with torch.no_grad():
        # 编码
        src_key_padding_mask = (encoder_inputs == zh_tokenizer.pad_token_id)
        memory = model.encode(src=encoder_inputs, src_key_padding_mask=src_key_padding_mask)
        # memory.shape: [batch_size, src_len, dim_model]

        # 解码
        batch_size = encoder_inputs.shape[0]
        decoder_inputs = torch.full(size=[batch_size, 1], fill_value=en_tokenizer.sos_token_id, device=device)

        # 句子完成标识
        is_finish = torch.full(size=[batch_size], fill_value=False, device=device)

        for step in range(MAX_STEPS):
            tgt_mask = model.transformer.generate_square_subsequent_mask(sz=decoder_inputs.shape[-1], device=device)

            logits = model.decode(
                memory=memory,
                tgt=decoder_inputs,
                tgt_mask=tgt_mask,
                memory_key_padding_mask=src_key_padding_mask
            )

            # logits.shape: [batch_size, tgt_len, en_vocab_size]
            next_token_logits = logits[:, -1, :]
            # next_token_logits.shape: [batch_size, en_vocab_size]

            next_token_ids = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            # next_token_ids.shape: [batch_size, 1]

            decoder_inputs = torch.cat([decoder_inputs, next_token_ids], dim=-1)
            # decoder_inputs.shape: [batch_size, tgt_len]

            # 判断eos
            is_finish |= next_token_ids.squeeze(-1) == en_tokenizer.eos_token_id

            if is_finish.all():
                break

        # 取出预测结果
        token_ids_list = decoder_inputs.tolist()
        # [[2,3,4,5],[4,5,6,7],[4,5,6,7]]
        batch_result = []
        for token_ids in token_ids_list:
            if en_tokenizer.eos_token_id in token_ids:
                batch_result.append(token_ids[1:token_ids.index(en_tokenizer.eos_token_id)])
            else:
                batch_result.append(token_ids[1:])
        return batch_result


def predict(text: str | list[str], model, zh_tokenizer, en_tokenizer, device) -> str | list[str]:
    # 处理输入
    is_str = isinstance(text, str)
    if is_str:
        text = [text]

    indexes_list = [torch.tensor(zh_tokenizer.encode(sentence), dtype=torch.long) for sentence in text]
    encoder_inputs = pad_sequence(indexes_list, batch_first=True, padding_value=en_tokenizer.pad_token_id)
    # encoder_inputs.shape: [batch_size, seq_len]
    batch_result = predict_batch(encoder_inputs, model, zh_tokenizer, en_tokenizer, device)
    # 处理输出
    result = [en_tokenizer.decode(result) for result in batch_result]

    if is_str:
        return result[0]
    return result


if __name__ == '__main__':
    text = ['我喜欢你。', '我喜欢打篮球。']
    # 设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 分词器
    zh_tokenizer = ChineseTokenizer.from_vocab(MODELS_DIR / 'zh_vocab.txt')
    en_tokenizer = EnglishTokenizer.from_vocab(MODELS_DIR / 'en_vocab.txt')

    # 模型
    model = TranslationModel(zh_vocab_size=zh_tokenizer.vocab_size, zh_pad_index=zh_tokenizer.pad_token_id,
                             en_vocab_size=en_tokenizer.vocab_size, en_pad_index=en_tokenizer.pad_token_id)
    model.load_state_dict(torch.load(MODELS_DIR / 'best_model.pt'))

    print(predict(text, model, zh_tokenizer, en_tokenizer, device))
