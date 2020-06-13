from ast import literal_eval
import numpy as np
import pickle


def filter_data(tokens_list):
    return [token for token in tokens_list if token != ' ']  


def reverse_vocab(word_dictionary):
    return {value: key for key, value in word_dictionary.items()}


def word_level(sentence, word_dictionary, mode, checker):
    new = []
    for word in sentence.split(" "):
        if checker == 'decoding':
            word = int(word)
        
        new.append(word if word_dictionary is None else word_dictionary[word])
        
        if mode != 'output':
            new.append(' ' if word_dictionary is None else word_dictionary[' '])
    return new[-1] if mode != 'output' else new


def char_level(sentence, word_dictionary):
    return [char if word_dictionary is None else word_dictionary[char] for word in sentence for char in sentence]


def build_vocab(sentences):
    dictionary = {' ': 1}
    for sentence in sentences:
        for word in sentence.split(" "):
            words_set = list(dictionary.keys())
            if word not in words_set:
                i = len(dictionary) + 1
                dictionary[word] = i
    return dictionary       


def save_config(tokenizer):
    with open('tokenizer_output.pickle', 'wb') as file:
        pickle.dump(tokenizer, file)        


class Oper(object):
    def __init__(self, 
                 sentence,
                 mode, 
                 word_dictionary,
                 max_length,
                 checker):
        self.sentence = sentence  
        self.word_dictionary = word_dictionary
        self.mode = mode
        self.max_length = max_length
        self.checker = checker
        
    def do(self, word_dictionary, checker=None):
        if self.mode == 'char_level':
            tokenized = char_level(self.sentence, word_dictionary)
        tokenized = word_level(self.sentence, word_dictionary, self.mode, self.checker)     
        
        if self.mode == 'output':
            return filter_data(tokenized)
        return tokenized 
    
    @property
    def tokens(self):
        word_dictionary = None
        return self.do(word_dictionary) 
    
    @property
    def ids(self):
        word_dictionary = self.word_dictionary
        return self.do(word_dictionary)
    
    @property
    def padded_tokens(self):
        word_dictionary = self.word_dictionary
        tokens = self.do(word_dictionary)
        return np.pad(tokens, (0, self.max_length - len(tokens)), 'constant',
                      constant_values=(self.word_dictionary['<pad>']))
    
    
class TokenizerBase(object):
    def __init__(self,
                 files,
                 max_length,
                 special_tokens,
                 mode='word_level',
                 checker='output',
                 tokenizer=None):
        self.mode = mode
        self.checker = checker
        self.max_length = max_length
        self.vocab = {}
                
        with open(files, 'r') as file:
                sentences = file.read()
                sentences = literal_eval(sentences)
                sentences = sentences + special_tokens
                self.sentences = sentences
    
    @property
    def get_vocab(self):
        return self.vocab
            
    def train(self):
        self.vocab = build_vocab(self.sentences)
        self.reverse_vocab = reverse_vocab(self.vocab)
        save_config(self)
    
    def encode(self, sentence):
        word_dictionary = self.vocab
        sentence = '<start> ' + sentence + ' <end>'
        return Oper(sentence=sentence, mode=self.checker, word_dictionary=word_dictionary,
                    max_length=self.max_length, checker=None) 
    
    def decode(self, sentence):
        word_dictionary = self.reverse_vocab
        sentence = [str(item) for item in sentence]
        sentence = " ".join(sentence)
        return " ".join(Oper(sentence=sentence, mode=self.checker, word_dictionary=word_dictionary,
                    max_length=self.max_length, checker='decoding').ids)
    

class Tokenizer(object):
    def __init__(self,
                 files,
                 max_length,
                 special_tokens,
                 mode='word_level',
                 checker='output',
                 tokenizer=None):
        if tokenizer is None:
            self.tokenizer = TokenizerBase(files, max_length, special_tokens,
                                           mode, checker, tokenizer)            
        else:
            with open(tokenizer, 'rb') as file:
                tokenizer = pickle.load(file)
            self.tokenizer = tokenizer
            
    @property
    def get_vocab(self):
        return self.tokenizer.vocab
    
    @property
    def get_reverse_vocab(self):
        return self.tokenizer.reverse_vocab
            
    def train(self):
        self.tokenizer.train()
    
    def encode(self, sentence):
        return self.tokenizer.encode(sentence)    
    
    def decode(self, tokens):
        return self.tokenizer.decode(tokens)
    
