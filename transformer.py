### HW2 solution (Spring 2021)
### transformer.py
###
### Please do not distribute to anyone else (especially students taking this course in future years).
from nltk.tokenize import TweetTokenizer
tweet_tokenizer = TweetTokenizer()

import torch
device = 'cpu'

import math, collections.abc, random, copy

from layers import *

# If installed, this prints progress bars
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable):
        return iterable

class Vocab(collections.abc.MutableSet):
    """Set-like data structure that can change words into numbers and back."""
    def __init__(self):
        words = {'<BOS>', '<EOS>', '<UNK>'}
        self.num_to_word = list(words)    
        self.word_to_num = {word:num for num, word in enumerate(self.num_to_word)}
    def add(self, word):
        if word in self: return
        num = len(self.num_to_word)
        self.num_to_word.append(word)
        self.word_to_num[word] = num
    def discard(self, word):
        raise NotImplementedError()
    def __contains__(self, word):
        return word in self.word_to_num
    def __len__(self):
        return len(self.num_to_word)
    def __iter__(self):
        return iter(self.num_to_word)

    def numberize(self, word):
        """Convert a word into a number."""
        if word in self.word_to_num:
            return self.word_to_num[word]
        else: 
            return self.word_to_num['<UNK>']

    def denumberize(self, num):
        """Convert a number into a word."""
        return self.num_to_word[num]

def read_parallel(filename):
    """Read data from the file named by 'filename.'

    The file should be in the format:

    Tweet \t Outcome

    where \t is a tab character.
    """
    data = []
    for line in open(filename):
        try:
        #print(line)
            fline, eline = line.split('\t')
            fwords = ['<BOS>'] + tweet_tokenizer.tokenize(fline) + ['<EOS>']
            ewords = ['<BOS>'] + tweet_tokenizer.tokenize(eline) + ['<EOS>']
            data.append((fwords, ewords))
        except ValueError:
            continue
    return data

def read_mono(filename):
    """Read sentences from the file named by 'filename.' """
    data = []
    for line in open(filename):
        words = ['<BOS>'] + tweet_tokenizer.tokenize(line) + ['<EOS>']
        data.append(words)
    return data
    
class Encoder(torch.nn.Module):
    """Transformer encoder."""
    def __init__(self, vocab_size, dims):
        super().__init__()
        self.emb = Embedding(vocab_size, dims)
        self.pos = torch.nn.Parameter(torch.empty(1000, dims))
        torch.nn.init.normal_(self.pos, std=0.01)
        self.att1 = SelfAttention(dims)
        self.ffnn1 = ResidualTanhLayer(dims)
        self.att2 = SelfAttention(dims)
        self.ffnn2 = ResidualTanhLayer(dims)

    def forward(self, fnums):
        e = self.emb(fnums) + self.pos[:len(fnums)]
        h = self.att1(e)
        h = self.ffnn1(h)
        h = self.att2(h)
        h = self.ffnn2(h)
        return h
    
class Decoder(torch.nn.Module):
    """Transformer decoder."""

    def __init__(self, dims, vocab_size):
        super().__init__()
        self.emb = Embedding(vocab_size, dims)
        self.pos = torch.nn.Parameter(torch.empty(1000, dims))
        torch.nn.init.normal_(self.pos, std=0.01)
        self.att = MaskedSelfAttention(dims)
        self.ffnn = ResidualTanhLayer(dims)
        self.merge = TanhLayer(dims+dims, dims)
        self.out = SoftmaxLayer(dims, vocab_size)

    def start(self):
        """Return the initial state of the decoder.

        Since the only layer that has state is self.att, we just use
        self.rnn's state. If there were more than one self-attention
        layer, this would be more complicated.
        """
        
        return self.att.start()

    def step(self, fencs, h, enum):
        """Run one step of the decoder:

        1. Read in an English word (enum) and compute a new state from the old state (h).
        2. Compute a probability distribution over the next English word.

        Arguments:
            fencs: Chinese word encodings (tensor of size n,d)
            h: Old state of decoder
            enum:  Next English word (int)

        Returns (logprobs, h), where
            logprobs: Vector of log-probabilities (tensor of size len(evocab))
            h: New state of decoder
        """
        
        e = self.emb(enum) + self.pos[len(h)]
        a, h = self.att.step(h, e)
        a = self.ffnn(a)
        c = attention(a, fencs, fencs)
        m = self.merge(torch.cat([c, a]))
        o = self.out(m)
        return (o, h)

class Model(torch.nn.Module):
    def __init__(self, fvocab, dims, evocab):
        super().__init__()
        
        # Store the vocabularies inside the Model object
        # so that they get loaded and saved with it.
        self.fvocab = fvocab
        self.evocab = evocab
        
        self.enc = Encoder(len(fvocab), dims)
        self.dec = Decoder(dims, len(evocab))
        
        # This is just so we know what device to create new tensors on
        self.dummy = torch.nn.Parameter(torch.empty(0))

    def logprob(self, fwords, ewords):
        """Return the log-probability of a sentence pair.

        Arguments:
            fwords: source sentence (list of str)
            ewords: target sentence (list of str)

        Return:
            log-probability of ewords given fwords (scalar)"""
        
        fnums = torch.tensor([self.fvocab.numberize(f) for f in fwords], device=self.dummy.device)
        fencs = self.enc(fnums)
        h = self.dec.start()
        logprob = 0.
        assert ewords[0] == '<BOS>'
        enum = self.evocab.numberize(ewords[0])
        for i in range(1, len(ewords)):
            o, h = self.dec.step(fencs, h, enum)
            enum = self.evocab.numberize(ewords[i])
            logprob += o[enum]
        return logprob

    def translate(self, fwords):
        """Translate a sentence using greedy search.

        Arguments:
            fwords: source sentence (list of str)

        Return:
            ewords: target sentence (list of str)
        """
        
        fnums = torch.tensor([self.fvocab.numberize(f) for f in fwords], device=self.dummy.device)
        fencs = self.enc(fnums)
        h = self.dec.start()
        ewords = []
        enum = self.evocab.numberize('<BOS>')
        for i in range(100):
            o, h = self.dec.step(fencs, h, enum)
            enum = torch.argmax(o).item()
            eword = self.evocab.denumberize(enum)
            if eword == '<EOS>': break
            ewords.append(eword)
        return ewords

if __name__ == "__main__":
    import argparse, sys
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', type=str, help='training data')
    parser.add_argument('--dev', type=str, help='development data')
    parser.add_argument('infile', nargs='?', type=str, help='test data to translate')
    parser.add_argument('-o', '--outfile', type=str, help='write translations to file')
    parser.add_argument('--load', type=str, help='load model from file')
    parser.add_argument('--save', type=str, help='save model in file')
    args = parser.parse_args()

    if args.train:
        # Read training data and create vocabularies
        traindata = read_parallel(args.train)

        fvocab = Vocab()
        evocab = Vocab()
        for fwords, ewords in traindata:
            fvocab |= fwords
            evocab |= ewords

        # Create model
        m = Model(fvocab, 64, evocab) # try increasing 64 to 128 or 256
        
        if args.dev is None:
            print('error: --dev is required', file=sys.stderr)
            sys.exit()
        devdata = read_parallel(args.dev)
            
    elif args.load:
        if args.save:
            print('error: --save can only be used with --train', file=sys.stderr)
            sys.exit()
        if args.dev:
            print('error: --dev can only be used with --train', file=sys.stderr)
            sys.exit()
        m = torch.load(args.load)

    else:
        print('error: either --train or --load is required', file=sys.stderr)
        sys.exit()

    if args.infile and not args.outfile:
        print('error: -o is required', file=sys.stderr)
        sys.exit()

    if args.train:
        opt = torch.optim.Adam(m.parameters(), lr=0.0003)

        best_dev_loss = None
        for epoch in range(10):
            random.shuffle(traindata)

            ### Update model on train

            train_loss = 0.
            train_ewords = 0
            for fwords, ewords in tqdm(traindata):
                loss = -m.logprob(fwords, ewords)
                opt.zero_grad()
                loss.backward()
                opt.step()
                train_loss += loss.item()
                train_ewords += len(ewords)-1 # -1 for BOS

            ### Validate on dev set and print out a few translations
            
            dev_loss = 0.
            dev_ewords = 0
            for line_num, (fwords, ewords) in enumerate(devdata):
                dev_loss -= m.logprob(fwords, ewords).item()
                dev_ewords += len(ewords)-1 # -1 for BOS
                if line_num < 10:
                    translation = m.translate(fwords)
                    print(' '.join(translation))

            if best_dev_loss is None or dev_loss < best_dev_loss:
                best_model = copy.deepcopy(m)
                if args.save:
                    torch.save(m, args.save)
                best_dev_loss = dev_loss

            print(f'[{epoch+1}] train_loss={train_loss} train_ppl={math.exp(train_loss/train_ewords)} dev_ppl={math.exp(dev_loss/dev_ewords)}', flush=True)
            
        m = best_model

    ### Translate test set

    if args.infile:
        with open(args.outfile, 'w') as outfile:
            for fwords in read_mono(args.infile):
                translation = m.translate(fwords)
                print(' '.join(translation), file=outfile)
