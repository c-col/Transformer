import pandas as pd
import torchtext
from torchtext import data
from collections import Counter, defaultdict
from Tokenize import ListTokenizer, StringPieceTokenizer
from Batch import MyIterator, batch_size_fn
import os
import dill as pickle


def read_data(opt):
    if opt.data_path is not None:
        try:
            opt.src_data = open(opt.data_path + '/train_in.txt').read().strip().split('\n')
        except:
            print("error: '" + opt.data_path + '/train_in.txt' + "' file not found")
            quit()

        try:
            opt.trg_data = open(opt.data_path + '/train_out.txt', encoding='utf-8').read().strip().split('\n')
        except:
            print("error: '" + opt.data_path + '/train_out.txt' + "' file not found")
            quit()


def create_fields(opt):
    print("loading task vocabulary and tokenizer...")

    train_src = [x.replace('\n', '').lower() for x in open(opt.data_path + '/train_in.txt', 'r', encoding='utf-8').readlines()]
    val_src = [x.replace('\n', '').lower() for x in open(opt.data_path + '/val_in.txt', 'r', encoding='utf-8').readlines()]
    test_src = [x.replace('\n', '').lower() for x in open(opt.data_path + '/test_in.txt', 'r', encoding='utf-8').readlines()]

    train_src = ' '.join(train_src).split()
    val_src = ' '.join(val_src).split()
    test_src = ' '.join(test_src).split()

    src_tokens = list(set(train_src + val_src + test_src))
    vocab_tokens = ["'", " '", "(", " (", ")", " )", ","]
    for token in src_tokens:
        if "'" not in token:
            vocab_tokens.append(token)

    # add string piece tokenization vocabulary
    vocab_tokens += [x.replace('\n', '') for x in open('string_piece_vocabulary.txt', 'r', encoding='utf-8')]
    vocab_tokens = list(set(vocab_tokens))
    vocab_counter = Counter(vocab_tokens)
    vocab_tokens = ['<unk>', '<pad>', '<sos>', '<eos>'] + vocab_tokens

    tokenizer_ = StringPieceTokenizer()
    TRG = data.Field(lower=True, tokenize=tokenizer_.tokenizer, init_token='<sos>', eos_token='<eos>')
    SRC = data.Field(lower=True, tokenize=tokenizer_.tokenizer)

    # instantiate vocab list
    TRG.build_vocab(vocab_counter)
    SRC.build_vocab(vocab_counter)

    # create mappings between vocab and integers (numericalizer ?)
    stoi = defaultdict()
    itos = []
    # print('vocab size', len(vocab_tokens))
    for i, t in enumerate(vocab_tokens):
        stoi[t] = i
        itos.append(t)
    TRG.vocab.stoi = stoi
    SRC.vocab.stoi = stoi
    TRG.vocab.itos = itos
    SRC.vocab.itos = itos

    if opt.load_weights is not None:
        try:
            print("loading presaved fields...")
            SRC = pickle.load(open(f'{opt.load_weights}/SRC.pkl', 'rb'))
            TRG = pickle.load(open(f'{opt.load_weights}/TRG.pkl', 'rb'))
        except:
            print("error opening SRC.pkl and TXT.pkl field files, please ensure they are in " + opt.load_weights + "/")
            quit()
        
    return SRC, TRG


def create_dataset(opt, SRC, TRG):
    print("creating dataset and iterator... ")

    # Load in validation data
    f_in, f_out = open(opt.data_path + '/val_in.txt', 'r', encoding='utf-8'), open(opt.data_path + '/val_out.txt', 'r', encoding='utf-8')
    in_ = [x.replace('\n', '') for x in f_in.readlines()]
    out_ = [x.replace('\n', '') for x in f_out.readlines()]

    raw_data = {'src': in_, 'trg': out_}
    df = pd.DataFrame(raw_data, columns=["src", "trg"])

    mask = (df['src'].str.count(' ') < opt.max_strlen) & (df['trg'].str.count(' ') < opt.max_strlen)
    df = df.loc[mask]

    df.to_csv("translate_transformer_temp.csv", index=False)
    data_fields = [('src', SRC), ('trg', TRG)]
    val = data.TabularDataset('./translate_transformer_temp.csv', format='csv', fields=data_fields)
    os.remove('translate_transformer_temp.csv')

    val_iter = MyIterator(val, batch_size=opt.batchsize,
                          repeat=False, sort_key=lambda x: (len(x.src), len(x.trg)),
                          batch_size_fn=batch_size_fn, train=False, shuffle=False)

    ##### TRAIN DATA #####
    raw_data = {'src' : [line for line in opt.src_data], 'trg': [line for line in opt.trg_data]}
    df = pd.DataFrame(raw_data, columns=["src", "trg"])
    
    mask = (df['src'].str.count(' ') < opt.max_strlen) & (df['trg'].str.count(' ') < opt.max_strlen)
    df = df.loc[mask]

    df.to_csv("translate_transformer_temp.csv", index=False)
    data_fields = [('src', SRC), ('trg', TRG)]
    train = data.TabularDataset('./translate_transformer_temp.csv', format='csv', fields=data_fields)

    train_iter = MyIterator(train, batch_size=opt.batchsize, # device=opt.device,
                        repeat=False, sort_key=lambda x: (len(x.src), len(x.trg)),
                        batch_size_fn=batch_size_fn, train=True, shuffle=True)

    os.remove('translate_transformer_temp.csv')

    if opt.load_weights is None:
        if opt.checkpoint > 0:
            try:
                os.mkdir("weights")
            except:
                print("weights folder already exists, run program with -load_weights weights to load them")
                quit()
            pickle.dump(SRC, open('weights/SRC.pkl', 'wb'))
            pickle.dump(TRG, open('weights/TRG.pkl', 'wb'))

    opt.src_pad = SRC.vocab.stoi['<pad>']
    opt.trg_pad = TRG.vocab.stoi['<pad>']

    opt.train_len = get_len(train_iter)

    return train_iter, val_iter

def get_len(train):

    for i, b in enumerate(train):
        pass
    
    return i
