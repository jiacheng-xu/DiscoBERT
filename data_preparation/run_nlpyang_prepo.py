if __name__ == '__main__':
    # tokenization
    from data_preparation.nlpyang_prepo import tokenize

    raw_path = '/datadrive/data/cnn/stories'
    TOKENIZED_PATH = '/datadrive/data/cnn/tokenized_stories'
    tokenize(raw_path=raw_path, save_path=TOKENIZED_PATH)
