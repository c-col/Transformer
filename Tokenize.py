# import re
from Utils import remove_whitespace


class ListTokenizer(object):
    def __init__(self, opt):
        train_src = [x.replace('\n', '').lower() for x in
                     open(opt.data_path + '/train_in.txt', 'r', encoding='utf-8').readlines()]
        val_src = [x.replace('\n', '').lower() for x in open(opt.data_path + '/val_in.txt', 'r', encoding='utf-8').readlines()]
        test_src = [x.replace('\n', '').lower() for x in
                    open(opt.data_path + '/test_in.txt', 'r', encoding='utf-8').readlines()]

        train_src = ' '.join(train_src).split()
        val_src = ' '.join(val_src).split()
        test_src = ' '.join(test_src).split()

        src_tokens = list(set(train_src + val_src + test_src))
        vocab_tokens = ["'", " '", "(", " (", ")", " )", ","]
        for token in src_tokens:
            if "'" not in token:
                vocab_tokens.append(token)

        self.vocab_tokens = vocab_tokens

    def tokenizer(self, sentence):
        sentence = sentence.lower()
        sentence = sentence.replace("'", " ' ").replace("(", " ( ").replace(")", " ) ").replace(",", " , ")
        sentence = remove_whitespace(sentence)
        temp_tokenization = sentence.split()
        tokenization = []
        for tok in temp_tokenization:
            if tok not in self.vocab_tokens:
                spaced = True
                while len(tok):
                    if spaced:
                        tokenization.append(tok[:3])
                    else:
                        tokenization.append('%' + tok[:3])
                    tok = tok[3:]
                    spaced = False
            else:
                tokenization.append(tok)

        return tokenization


class StringPieceTokenizer(object):
    def __init__(self):
        pass

    def tokenizer(self, sentence):
        sentence = sentence.lower()
        sentence = sentence.replace("'", " ' ").replace(",", " , ")
        sentence = remove_whitespace(sentence)
        return sentence.split()