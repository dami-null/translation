import torch
from nltk.translate.bleu_score import corpus_bleu
from tqdm import tqdm

from src.config import MODELS_DIR
from src.dataset import get_dataloader
from src.model import TranslationModel
from src.predict import predict_batch
from src.tokenizer import ChineseTokenizer, EnglishTokenizer
from src.train import get_device


def evaluate(dataloader, model, zh_tokenizer, en_tokenizer, device) -> float:
    """
    评估模型
    :param dataloader: 数据集
    :param model: 模型
    :param zh_tokenizer: 分词器
    :param en_tokenizer: 分词器
    :param device: 设备
    :return: bleu score
    """
    # 预测结果
    all_predictions = []  # [[1,2,3],[1,2,3],[1,2,3],[1,2,3],[1,2,3],[1,2,3]]
    # 参考译文
    all_references = []

    # 进度条tqdm
    for encoder_inputs, decoder_inputs, labels in tqdm(dataloader, desc='Evaluate'):
        batch_result = predict_batch(encoder_inputs, model, zh_tokenizer, en_tokenizer, device)
        all_predictions.extend(batch_result)

        # labels切片[0, eos]
        all_references.extend(
            [[token_ids[:token_ids.index(en_tokenizer.eos_token_id)]] for token_ids in labels.tolist()])
        # [[[1, 2, 3]], [[1, 2, 3]], [[1, 2, 3]], [[1, 2, 3]],[[1, 2, 3]], [[1, 2, 3]]]
    return corpus_bleu(all_references, all_predictions)


if __name__ == '__main__':
    # 设备
    device = get_device()

    # 分词器
    zh_tokenizer = ChineseTokenizer.from_vocab(MODELS_DIR / 'zh_vocab.txt')
    en_tokenizer = EnglishTokenizer.from_vocab(MODELS_DIR / 'en_vocab.txt')

    # 模型
    model = TranslationModel(zh_vocab_size=zh_tokenizer.vocab_size, zh_pad_index=zh_tokenizer.pad_token_id,
                             en_vocab_size=en_tokenizer.vocab_size, en_pad_index=en_tokenizer.pad_token_id)
    model.load_state_dict(torch.load(MODELS_DIR / 'best_model.pt'))

    # 数据（测试集）
    dataloader = get_dataloader(train=False)

    print(evaluate(dataloader, model, zh_tokenizer, en_tokenizer, device))

