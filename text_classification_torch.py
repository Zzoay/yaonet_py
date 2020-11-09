import torch
# import torchtext
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.nn.functional as F


class MyDataset(Dataset):
    
    def __init__(self, data_path, vocab_path=None, labels_path=False):
        self.vocab = None
        if vocab_path:
            self.vocabs = self.get_vocab(vocab_path)
        self.features = self.read_features(data_path, self.vocabs)
        self.labels = self.read_labels(labels_path)
        self.data = [item for item in zip(self.features, self.labels)]
        
    def __len__(self):
        return len(self.data)

    def get_vocab(self, vocab_path):
        with open(vocab_path, encoding='utf-8') as f:
            words = []
            for line in f.readlines():
                words.append(line.strip("\n").split(' '))
            return {item[0]:int(item[1]) for item in words}
    
    def read_features(self, data_path, vocabs=None):
        words_list = []
        with open(data_path, encoding='utf-8') as f:
            for line in f.readlines():
                words = line.strip('\n').split(' ')
                if vocabs:
                    words_list.append(torch.tensor([vocabs.get(word) for word in words]))
            return words_list
    
    def read_labels(self, labels_path):
        labels_list = []
        with open(labels_path, encoding='utf-8') as f:
            for line in f.readlines():
                words = line.strip('\n').split(' ')
                tmp = torch.tensor(int(words[0]), dtype=torch.long)
                labels_list.append(tmp)
            return labels_list
    
    def __getitem__(self, idx):
        return self.data[idx]


class TextCNN(nn.Module):
    
    def __init__(self, params):
        super(TextCNN, self).__init__()
        
        embed_dim = params['embed_dim']
        kernel_num = params['kernel_num']
        kernel_sizes = params['kernel_sizes']
        dropout = params['dropout']
        class_nums = params['class_num']
        
        self.emb = nn.Embedding(params['vocab_size'], embed_dim)
        self.convs = nn.ModuleList([nn.Conv2d(1, kernel_num, (K, embed_dim)) for K in kernel_sizes])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(kernel_num*len(kernel_sizes), class_nums)

    def forward(self, x):
        x = self.emb(x)  # [batch_size, sentence_len, embed_dim]
        
        x = x.unsqueeze(1)  # [batch_size, 1, sentence_len, embed_dim]
        
        conv_outputs = []
        for conv in self.convs:
            tmp = conv(x)
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs] 
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # [(N, Co), ...]*len(Ks)
        x = torch.cat(x, 1)
        
        x = self.dropout(x)
        logit = self.fc(x)

        return logit


def eval(val_iter, model, test=False):
    model.eval()
    corrects, avg_loss = 0, 0
    for batch in val_iter:
        batch_x, batch_y = batch
        batch_x, batch_y = batch_x.cuda(), batch_y.cuda()

        logit = model(batch_x)
        loss = F.cross_entropy(logit, batch_y)
        
        avg_loss += loss.data
        predicts = torch.max(logit, 1)[1]

        corrects += (predicts.view(batch_y.size()).data == batch_y.data).sum()
    
    if test:
        size = test_size
    else:
        size = val_size
    avg_loss /= size
    corrects = float(corrects)
    accuracy = 100.0 * corrects/size
    print('\t loss: {:.6f}  acc: {:.2f}%({}/{}) \n'.format(avg_loss, accuracy, int(corrects), size))


def train(train_iter, val_iter, test_iter, model, epochs):
    model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    steps = 0
    model.train()
    
    for epoch in range(1, epochs+1):
        for batch in train_iter:
            batch_x, batch_y = batch
            batch_x, batch_y = batch_x.cuda(), batch_y.cuda()

            optimizer.zero_grad()
            logit = model(batch_x)

            loss = F.cross_entropy(logit, batch_y)
            loss.backward()
            optimizer.step()

            # train acc
            if steps % 100 == 0:
                predicts = torch.max(logit, 1)[1]
                corrects = (predicts.view(batch_y.size()).data == batch_y.data).float().sum()
                accuracy = 100.0 * float(corrects/batch_size)
                print('step {} , training accuracy : {:.2f} %'.format(steps, accuracy))
                # dev
                print("----Validation:")
                eval(val_iter=val_iter, model=model, test=False)
    
            steps += 1

            # test
            if steps % 1000 == 0:
                print("----------------------")
                print("----Test:")
                eval(val_iter=test_iter, model=model, test=True)


if __name__ == "__main__":
    corpus = 'sst1'

    data_path = '{}/data.txt'.format(corpus)
    labels_path = '{}/labels.txt'.format(corpus)
    vocab_path = '{}/vocab.txt'.format(corpus)

    train_test_bound = 7680
    validation = True
    val_ratio = 0.1
    val_size = int(train_test_bound * val_ratio)
    train_size = train_test_bound - val_size

    batch_size = 128
            
    model_params = {
        'embed_dim': 32,
        'class_num': 5,
        'kernel_num' :100,
        'kernel_sizes' : [3, 4, 5],
        'dropout' : 0.5,
    }

    data = MyDataset(data_path, vocab_path, labels_path)
    features = data.features
    labels = data.labels
    vocabs = data.vocabs

    features_size = len(features)
    labels_size = len(labels)
    print("labels_size : {}".format(labels_size))
    assert features_size == labels_size

    vocab_size = len(vocabs)
    del vocabs

    test_size = features_size - train_test_bound

    val = []
    train_data, test_data = data[:train_test_bound], data[train_test_bound:]
    if validation:
        train_data, val_data = torch.utils.data.random_split(train_data, [train_size, val_size])

    print("vocab_size : {}".format(vocab_size))
    print("train_size : {}".format(len(train_data)))
    print("val_size : {}".format(len(val_data)))
    print("test_size : {}".format(len(test_data)))

    train_iter = DataLoader(train_data, batch_size=batch_size)
    val_iter = DataLoader(val_data, batch_size=batch_size)
    test_iter = DataLoader(test_data, batch_size=batch_size)

    # 构建模型
    model_params['vocab_size'] = vocab_size
    model = TextCNN(model_params)
    print(model)

    # 开始训练
    train(train_iter, val_iter, test_iter, model, epochs=10)