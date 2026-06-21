import math

import torch
from torch import nn

from src.config import DIM_MODEL, NUM_HEAD, NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, DIM_FEEDFORWARD


class PositionEncoding(nn.Module):

    # 每个 token 在句子里的位置

    def __init__(self, max_length=500):
        super().__init__()
        pe = torch.zeros(size=[max_length, DIM_MODEL], dtype=torch.float)
        for pos in range(max_length):
            for _2i in range(0, DIM_MODEL, 2):
                # 位置函数（sin，cos）
                pe[pos, _2i] = math.sin(pos / (10000 ** (_2i / DIM_MODEL)))
                pe[pos, _2i + 1] = math.cos(pos / (10000 ** (_2i / DIM_MODEL)))
        # 要随着to方法放到指定设备去使用，不是parameter（会被优化器更新）是个buffer（不会更新），当作属性去访问self.名字
        self.register_buffer('pe', pe)

    def forward(self, embed):
        # embed.shape: [batch_size, seq_len, dim_model]
        sel_len = embed.shape[1]

        part_pe = self.pe[:sel_len, :]
        # part_pe.shape: [seq_len, dim_model]

        return embed + part_pe


class TranslationModel(nn.Module):

    # batch_size：一次喂几句话
    # seq_len：每一行的token_id数
    # embedding_dim：词向量维度

    def __init__(self, zh_vocab_size, zh_pad_index, en_vocab_size, en_pad_index):
        super().__init__()
        # 定义模型
        self.transformer = nn.Transformer(
            d_model=DIM_MODEL,
            nhead=NUM_HEAD,
            num_encoder_layers=NUM_ENCODER_LAYERS,
            num_decoder_layers=NUM_DECODER_LAYERS,
            dim_feedforward=DIM_FEEDFORWARD,
            # batch_first=True: batch_size作为第一维
            batch_first=True
        )

        # todo 关闭 Transformer 的 nested tensor 优化
        # todo PyTorch中Transformer + src_key_padding_mask 里有个算子，MPS 暂时不支持。
        self.transformer.encoder.enable_nested_tensor = False
        self.transformer.encoder.use_nested_tensor = False

        # Embedding：把token_id转为词向量 随机初始化
        self.zh_embedding = nn.Embedding(num_embeddings=zh_vocab_size, embedding_dim=DIM_MODEL,
                                         padding_idx=zh_pad_index)
        self.en_embedding = nn.Embedding(num_embeddings=en_vocab_size, embedding_dim=DIM_MODEL,
                                         padding_idx=en_pad_index)

        # 位置编码
        self.position_encoding = PositionEncoding()

        # 线性层 输入的是隐藏状态的维度
        self.linear = nn.Linear(in_features=DIM_MODEL, out_features=en_vocab_size)

    def forward(self, src, src_key_padding_mask, tgt, tgt_mask):
        """
        前向传播
        :param src: 原始序列
        :param src_key_padding_mask: 掩码
        :param tgt: 目标序列
        :param tgt_mask: 掩码
        :return: 预测token
        """
        memory = self.encode(src, src_key_padding_mask)
        output = self.decode(memory, tgt, tgt_mask, src_key_padding_mask)
        return output

    def encode(self, src, src_key_padding_mask):
        """
        编码
        :param src: 原始序列 shape: [batch_size, seq_len]
        :param src_key_padding_mask: 掩码
        :return: 隐藏状态
        """
        # src_embed.shape: [batch_size, seq_len, dim_model]
        src_embed = self.zh_embedding(src)
        src_embed = self.position_encoding(src_embed)

        # memory：隐藏状态 shape: [batch_size, seq_len, dim_model]
        memory = self.transformer.encoder(
            src=src_embed,
            src_key_padding_mask=src_key_padding_mask
        )

        return memory

    def decode(self, memory, tgt, tgt_mask, memory_key_padding_mask):
        """
        解码
        :param memory: encode输出的隐藏状态 shape: [batch_size, seq_len, dim_model]
        :param tgt: 目标序列 shape: [batch_size, seq_len]
        :param tgt_mask: 掩码
        :param memory_key_padding_mask: pad掩码
        :return: 预测token
        """

        # tgt_embed.shape: [batch_size, seq_len, dim_model]
        tgt_embed = self.en_embedding(tgt)
        tgt_embed = self.position_encoding(tgt_embed)

        # output.shape: [batch_size, seq_len, dim_model]
        output = self.transformer.decoder(
            tgt=tgt_embed,
            memory=memory,
            tgt_mask=tgt_mask,
            memory_key_padding_mask=memory_key_padding_mask
        )

        # 线性变换，处理输入张量的最后一维 shape: [batch_size, seq_len, en_vocab_size]
        output = self.linear(output)

        return output
