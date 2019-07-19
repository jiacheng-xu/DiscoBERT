import torch, os

if __name__ == '__main__':
    dataset = torch.load(os.path.join('/datadrive/data/cnndm/valid', 'dailymail.valid.8.bert.pt'))
    for d in dataset:
        print(d)