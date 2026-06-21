from src.config import DATA_DIR, MODELS_DIR
import pandas as pd
from sklearn.model_selection import train_test_split

from src.tokenizer import ChineseTokenizer, EnglishTokenizer


def process():
    # 读取文件
    df = pd.read_csv(DATA_DIR / 'raw' / 'cmn.txt', sep='\t', header=None, usecols=[0, 1], names=['en', 'zh'],
                     encoding='utf-8').dropna()

    # 划分数据集
    train_df, test_df = train_test_split(df, test_size=0.2)

    # 基于训练集构建词表
    ChineseTokenizer.build_vocab(train_df['zh'].tolist(), MODELS_DIR / 'zh_vocab.txt')
    EnglishTokenizer.build_vocab(train_df['en'].tolist(), MODELS_DIR / 'en_vocab.txt')

    # 创建分词器
    zh_tokenizer = ChineseTokenizer.from_vocab(MODELS_DIR / 'zh_vocab.txt')
    en_tokenizer = EnglishTokenizer.from_vocab(MODELS_DIR / 'en_vocab.txt')

    # 处理数据集
    train_df['zh'] = train_df['zh'].apply(lambda x: zh_tokenizer.encode(x, add_sos_eos=False))
    train_df['en'] = train_df['en'].apply(lambda x: en_tokenizer.encode(x, add_sos_eos=True))

    test_df['zh'] = test_df['zh'].apply(lambda x: zh_tokenizer.encode(x, add_sos_eos=False))
    test_df['en'] = test_df['en'].apply(lambda x: en_tokenizer.encode(x, add_sos_eos=True))

    # 保存处理结果
    train_df.to_json(DATA_DIR / 'processed' / 'train.jsonl', orient='records', lines=True)
    test_df.to_json(DATA_DIR / 'processed' / 'test.jsonl', orient='records', lines=True)


if __name__ == '__main__':
    process()
