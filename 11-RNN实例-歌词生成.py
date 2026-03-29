import torch
import re
import torch.nn as nn
import time
from torch.utils.data import DataLoader
import torch.optim as optim
import jieba


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# todo:获取数据，进行分词，建立词表
def build_vocab():
    unique_words, all_words = [], []
    for line in open('./data/jaychou_lyrics.txt', 'r', encoding='utf-8'):
        words = jieba.lcut(line)
        # print(words)
        all_words.append(words)
        for word in words:
            if word not in unique_words:
                unique_words.append(word)

    word_count = len(unique_words)
    # print(word_count)

    # 构建词表，字典形式，key是词，value是索引
    word_to_index = {word: i for i, word in enumerate(unique_words)}
    # print(word_to_index)

    # 生成词向量
    corpus_idx = []
    for words in all_words:
        tmp = []
        for word in words:
            tmp.append(word_to_index[word])

        # 每段词之间添加空格隔开
        tmp.append(word_to_index[' '])
        corpus_idx.extend(tmp)
        # print(corpus_idx)

    # 去重后的词表，词对应索引的字典，去重后词的数量，文档中把词转换为索引的结果
    return unique_words, word_to_index, word_count, corpus_idx


# todo:构建数据集
class LyricsDataset(torch.utils.data.Dataset):
    def __init__(self, corpus_idx, seq_len):
        self.corpus_idx = corpus_idx
        self.seq_len = seq_len
        # 🔥 正确样本数：总长度 - 序列长度（防止y越界）
        self.total_samples = len(self.corpus_idx) - self.seq_len

    def __len__(self):
        # 返回真实样本数量
        return self.total_samples

    def __getitem__(self, idx):
        # 滑动窗口：idx从0到total_samples-1
        start = idx
        end = start + self.seq_len
        x = self.corpus_idx[start:end]
        y = self.corpus_idx[start+1:end+1]
        return torch.tensor(x), torch.tensor(y)


class TextGenerrator(nn.Module):
    def __init__(self, word_count):
        super().__init__()
        # 初始化嵌入层
        self.emd = nn.Embedding(word_count, 128)
        # 循环网络层：词向量维度， 隐藏层维度， 网络层数
        self.rnn = nn.RNN(128, 256, 1, batch_first=True)
        # 全连接层：特征向量维度， 词表中词个数，
        self.out = nn.Linear(256, word_count)


    def forward(self, inputs, hidden):
        embd = self.emd(inputs)
        output, hidden = self.rnn(embd, hidden)
        # 输入维度 seq_len每个句子数量 * batch句子个数， 词向量维度
        output = self.out(output.reshape(shape = (-1, output.shape[-1])))

        return output, hidden


    def init_hidden(self, batch_size):
        # 网络层数，batch，隐藏层向量维度
        return torch.zeros(1, batch_size, 256, device=self.emd.weight.device)


def train():
    unique_words, word_to_index, word_count, corpus_idx = build_vocab()
    lyrics = LyricsDataset(corpus_idx, 32)
    model = TextGenerrator(word_count)
    model.train()
    model = model.to(device)
    lyrics_dataloader = DataLoader(lyrics, batch_size=5, shuffle=True)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    epochs = 10
    for epoch in range(epochs):
        start, iternum, total_loss = time.time(), 0, 0.0
        for x, y in lyrics_dataloader:
            x = x.to(device)    # [5,8]
            y = y.to(device)
            hidden = model.init_hidden(x.shape[0])
            output, hidden = model(x, hidden)
            loss = criterion(output, y.reshape(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            iternum += 1

        print(f'epoch:{epoch+1}, time:{time.time()-start:.2f}s, loss:{total_loss / iternum:.4f}')

    torch.save(model.state_dict(), './model/textgenerator.pth')


def evaluate(start_word, want_len):
    unique_words, word_to_index, word_count, corpus_idx = build_vocab()
    model = TextGenerrator(word_count)
    model.load_state_dict(torch.load('./model/textgenerator.pth',  map_location=device))
    model.eval()
    hidden = model.init_hidden(1)
    word_idx = word_to_index[start_word]
    generate_sentence = [word_idx, ]
    with torch.no_grad():
        for _ in range(want_len):
            output, hidden = model(torch.tensor([[word_idx]]), hidden)
            word_idx = torch.argmax(output)
            generate_sentence.append(word_idx)

    for i in generate_sentence:
        print(unique_words[i], end='')


if __name__ == '__main__':
    # unique_words, word_to_index, word_count, corpus_idx = build_vocab()
    # print(f'词的数量：{word_count}')
    # print(f'去重后的词：{unique_words}')
    # print(f'每个词索引：{word_to_index}')
    # print(f'文档中每个词的索引：{corpus_idx}')
    # dataset = LyricsDataset(corpus_idx,5)
    # print(f'句子数量：{len(dataset)}')
    # x, y = dataset[0]
    # print(f'输入：{x}')
    # print(f'输出：{y}')
    #
    # model = TextGenerator(word_count)
    train()
    evaluate('下雨', 80)
