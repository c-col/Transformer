import argparse
import time
import torch
from Models import get_model
from Process import *
from Utils import *
import torch.nn.functional as F
from Optim import CosineWithRestarts
from Batch import create_masks
from Beam import beam_search
from torch.autograd import Variable
import dill as pickle


def simple_em(pred, gold):
    pred, gold = remove_whitespace(pred), remove_whitespace(gold)
    if pred.lower().strip() == gold.lower().strip():
        return 1
    return 0


def difficulty_em(src, pred, gold):
    # return steps # , em
    if '@@sep@@' in src.lower():
        step_count = src.lower().count('@@sep@@') + 1
        return step_count, simple_em(pred, gold)
    else:
        return polish_n_steps(src), simple_em((pred, gold))


def train_model(model, opt, SRC, TRG):
    # val_data = zip_io_data('list_data_pol/val')
    test_data = zip_io_data(opt.data_path + '/test')
    test_predictions = ''

    print("training model...")
    model.train()
    start = time.time()
    if opt.checkpoint > 0:
        cptime = time.time()
                 
    for epoch in range(opt.epochs):
        model.train()  # TODO: this ok?
        total_loss = 0
        if opt.floyd is False:
            print("   %dm: epoch %d [%s]  %d%%  loss = %s" %\
            ((time.time() - start)//60, epoch + 1, "".join(' '*20), 0, '...'), end='\r')
        
        if opt.checkpoint > 0:
            torch.save(model.state_dict(), 'weights/model_weights')

        for i, batch in enumerate(opt.train): 

            src = batch.src.transpose(0, 1).cuda()
            trg = batch.trg.transpose(0, 1).cuda()
            trg_input = trg[:, :-1]
            src_mask, trg_mask = create_masks(src, trg_input, opt)
            preds = model(src, trg_input, src_mask, trg_mask)
            ys = trg[:, 1:].contiguous().view(-1)
            opt.optimizer.zero_grad()
            loss = F.cross_entropy(preds.view(-1, preds.size(-1)), ys, ignore_index=opt.trg_pad)
            loss.backward()
            opt.optimizer.step()
            if opt.SGDR == True: 
                opt.sched.step()
            
            total_loss += loss.item()
            
            if (i + 1) % opt.printevery == 0:
                 p = int(100 * (i + 1) / opt.train_len)
                 avg_loss = total_loss/opt.printevery
                 if opt.floyd is False:
                    print("   %dm: epoch %d [%s%s]  %d%%  loss = %.3f" %\
                    ((time.time() - start)//60, epoch + 1, "".join('#'*(p//5)), "".join(' '*(20-(p//5))), p, avg_loss), end='\r')
                 else:
                    print("   %dm: epoch %d [%s%s]  %d%%  loss = %.3f" %\
                    ((time.time() - start)//60, epoch + 1, "".join('#'*(p//5)), "".join(' '*(20-(p//5))), p, avg_loss))
                 total_loss = 0
            
            if opt.checkpoint > 0 and ((time.time()-cptime)//60) // opt.checkpoint >= 1:
                torch.save(model.state_dict(), 'weights/model_weights')
                cptime = time.time()
   
   
        print("%dm: epoch %d [%s%s]  %d%%  loss = %.3f\nepoch %d complete, loss = %.03f" %\
        ((time.time() - start)//60, epoch + 1, "".join('#'*(100//5)), "".join(' '*(20-(100//5))), 100, avg_loss, epoch + 1, avg_loss))

        model.eval()  # TODO: this ok?
        val_losses = []
        for i, batch in enumerate(opt.val):

            src = batch.src.transpose(0, 1).cuda()
            trg = batch.trg.transpose(0, 1).cuda()
            trg_input = trg[:, :-1]
            src_mask, trg_mask = create_masks(src, trg_input, opt)
            preds = model(src, trg_input, src_mask, trg_mask)
            ys = trg[:, 1:].contiguous().view(-1)
            opt.optimizer.zero_grad()
            loss = F.cross_entropy(preds.view(-1, preds.size(-1)), ys, ignore_index=opt.trg_pad)
            val_losses.append(loss.item())

        print('validation loss:', sum(val_losses)/len(val_losses), '\n')

        if epoch != opt.epochs - 1:
            continue

        val_acc, val_success = 0, 0
        for j, e in enumerate(test_data[:1000]):
            e_src, e_tgt = e[0], e[1]
            indexed = []
            sentence = SRC.preprocess(e_src)
            pass_bool = False
            for tok in sentence:
                if SRC.vocab.stoi[tok] != 0:
                    indexed.append(SRC.vocab.stoi[tok])
                else:
                    print('no tokenization for', tok)
                    pass_bool = True
                    break
            if pass_bool:
                test_predictions += '\n'
                continue

            sentence = Variable(torch.LongTensor([indexed]))
            if opt.device == 0:
                sentence = sentence.cuda()

            try:
                sentence = beam_search(sentence, model, SRC, TRG, opt)
            except Exception as e:
                test_predictions += '\n'
                continue
                # beam_errors += 1
                # print('beam error:', sentence, e_src, e_tgt)
                # print(e)

            try:
                val_acc += simple_em(sentence, e_tgt)
                val_success += 1
                test_predictions += sentence + '\n'
                print(sentence, '@@@@', e_tgt)
            except Exception as e:
                # print('acc add error:', sentence, e_src, e_tgt)
                # print(e)
                test_predictions += '\n'
                continue

        if val_success == 0:
            val_success = 1
        val_acc = val_acc / val_success
        print('test accuracy:', val_acc, '; successful attempts:', val_success)
        print()
        with open('pol_generations.txt', 'w', encoding='utf-8') as f:
            f.write(test_predictions)


def main():

    parser = argparse.ArgumentParser()
    # parser.add_argument('-src_data', required=True)
    # parser.add_argument('-trg_data', required=True)
    parser.add_argument('-data_path', required=True)
    parser.add_argument('-no_cuda', action='store_true')
    parser.add_argument('-SGDR', action='store_true')
    parser.add_argument('-epochs', type=int, default=15)
    parser.add_argument('-d_model', type=int, default=512)
    parser.add_argument('-n_layers', type=int, default=6)
    parser.add_argument('-heads', type=int, default=8)
    parser.add_argument('-dropout', type=int, default=0.1)
    parser.add_argument('-batchsize', type=int, default=3000)
    parser.add_argument('-printevery', type=int, default=100)
    parser.add_argument('-lr', type=int, default=0.00006)
    parser.add_argument('-load_weights')
    parser.add_argument('-create_valset', action='store_true')
    parser.add_argument('-max_strlen', type=int, default=512)
    parser.add_argument('-floyd', action='store_true')
    parser.add_argument('-checkpoint', type=int, default=0)

    opt = parser.parse_args()
    
    opt.device = 0 if opt.no_cuda is False else -1
    if opt.device == 0:
        assert torch.cuda.is_available()

    read_data(opt)
    SRC, TRG = create_fields(opt)
    opt.train, opt.val = create_dataset(opt, SRC, TRG)
    model = get_model(opt, len(SRC.vocab), len(TRG.vocab))

    # beam search parameters
    opt.k = 3
    opt.max_len = 500

    opt.optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, betas=(0.9, 0.98), eps=1e-9)
    if opt.SGDR == True:
        opt.sched = CosineWithRestarts(opt.optimizer, T_max=opt.train_len)

    if opt.checkpoint > 0:
        print("model weights will be saved every %d minutes and at end of epoch to directory weights/"%(opt.checkpoint))
    
    if opt.load_weights is not None and opt.floyd is not None:
        os.mkdir('weights')
        pickle.dump(SRC, open('weights/SRC.pkl', 'wb'))
        pickle.dump(TRG, open('weights/TRG.pkl', 'wb'))
    
    train_model(model, opt, SRC, TRG)

    if opt.floyd is False:
        promptNextAction(model, opt, SRC, TRG)

def yesno(response):
    while True:
        if response != 'y' and response != 'n':
            response = input('command not recognised, enter y or n : ')
        else:
            return response

def promptNextAction(model, opt, SRC, TRG):

    saved_once = 1 if opt.load_weights is not None or opt.checkpoint > 0 else 0
    
    if opt.load_weights is not None:
        dst = opt.load_weights
    if opt.checkpoint > 0:
        dst = 'weights'

    while True:
        save = yesno(input('training complete, save results? [y/n] : '))
        if save == 'y':
            while True:
                if saved_once != 0:
                    res = yesno("save to same folder? [y/n] : ")
                    if res == 'y':
                        break
                dst = input('enter folder name to create for weights (no spaces) : ')
                if ' ' in dst or len(dst) < 1 or len(dst) > 30:
                    dst = input("name must not contain spaces and be between 1 and 30 characters length, enter again : ")
                else:
                    try:
                        os.mkdir(dst)
                    except:
                        res= yesno(input(dst + " already exists, use anyway? [y/n] : "))
                        if res == 'n':
                            continue
                    break
            
            print("saving weights to " + dst + "/...")
            torch.save(model.state_dict(), f'{dst}/model_weights')
            if saved_once == 0:
                pickle.dump(SRC, open(f'{dst}/SRC.pkl', 'wb'))
                pickle.dump(TRG, open(f'{dst}/TRG.pkl', 'wb'))
                saved_once = 1
            
            print("weights and field pickles saved to " + dst)

        res = yesno(input("train for more epochs? [y/n] : "))
        if res == 'y':
            while True:
                epochs = input("type number of epochs to train for : ")
                try:
                    epochs = int(epochs)
                except:
                    print("input not a number")
                    continue
                if epochs < 1:
                    print("epochs must be at least 1")
                    continue
                else:
                    break
            opt.epochs = epochs
            train_model(model, opt, SRC, TRG)
        else:
            print("exiting program...")
            break

    # for asking about further training use while true loop, and return
if __name__ == "__main__":
    main()
