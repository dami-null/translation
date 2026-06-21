from pathlib import Path

from nltk import TreebankWordDetokenizer, TreebankWordTokenizer


class BaseTokenizer:
    pad_token = '<pad>'
    unk_token = '<unk>'
    sos_token = '<sos>'
    eos_token = '<eos>'

    def __init__(self, vocab_list: list[str]):
        self.vocab = vocab_list
        self.vocab_size = len(self.vocab)
        self.word2id = {word: id for id, word in enumerate(self.vocab)}
        self.id2word = {id: word for id, word in enumerate(self.vocab)}

        self.pad_token_id = self.word2id.get(self.pad_token)
        self.unk_token_id = self.word2id.get(self.unk_token)
        self.sos_token_id = self.word2id.get(self.sos_token)
        self.eos_token_id = self.word2id.get(self.eos_token)

    @classmethod
    def build_vocab(cls, sentences: list[str], vocab_file: Path):
        unique_tokens = set()
        for sentence in sentences:
            tokens = cls.tokenize(sentence)
            unique_tokens.update(tokens)

        all_tokens = [cls.pad_token, cls.unk_token, cls.sos_token, cls.eos_token] + list(unique_tokens)

        vocab_file.write_text('\n'.join(all_tokens), encoding='utf-8')

    @classmethod
    def from_vocab(cls, vocab_file: Path):
        text = vocab_file.read_text(encoding='utf-8')
        vocab_list = text.split('\n')
        return cls(vocab_list)

    @classmethod
    def tokenize(cls, sentence: str) -> list[str]:
        pass

    def detokenize(self, tokens: list[str]) -> str:
        pass

    def encode(self, sentence: str, add_sos_eos: bool = False) -> list[int]:
        # 分词
        tokens = self.tokenize(sentence)

        if add_sos_eos:
            tokens = [self.sos_token] + tokens + [self.eos_token]

        # token->id
        return [self.word2id.get(token, self.unk_token_id) for token in tokens]

    def decode(self, token_ids: list[int]) -> str:
        # id->token
        tokens = [self.id2word[token_id] for token_id in token_ids]
        # 反分词
        return self.detokenize(tokens)


class ChineseTokenizer(BaseTokenizer):

    @classmethod
    def tokenize(cls, sentence: str) -> list[str]:
        return list(sentence)

    def detokenize(self, tokens: list[str]) -> str:
        return ''.join(tokens)


class EnglishTokenizer(BaseTokenizer):
    detokenizer = TreebankWordDetokenizer()
    tokenizer = TreebankWordTokenizer()

    @classmethod
    def tokenize(cls, sentence: str) -> list[str]:
        return cls.tokenizer.tokenize(sentence)

    def detokenize(self, tokens: list[str]) -> str:
        return self.detokenizer.detokenize(tokens)


if __name__ == '__main__':
    sentences = [
        '我爱北京天安门',
        '我爱自然语言处理',
        '我爱北京']

    # 构建词表
    ChineseTokenizer.build_vocab(sentences, Path('vocab.txt'))

    # 创建分词器
    zh_tokenizer = ChineseTokenizer.from_vocab(Path('vocab.txt'))

    # encode
    ids = zh_tokenizer.encode('我爱上海')
    print(ids)
