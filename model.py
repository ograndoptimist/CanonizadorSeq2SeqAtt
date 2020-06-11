import math

import torch
import torch.cuda
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.rnn import (pack_padded_sequence, pad_packed_sequence)
from torch.utils.data import Dataset


def padding_tensor(sequences, null_id):
    num = len(sequences)
    max_len = max([s.shape[0] for s in sequences])
    out_dims = (num, max_len, *sequences[0].shape[1:])
    out_tensor = sequences[0].data.new(*out_dims).fill_(null_id)   
    for i, tensor in enumerate(sequences):
        length = tensor.size(0)
        out_tensor[i, :length] = tensor
    return out_tensor


def collate_fn_padd(batch):
    input_ = []
    output_ = []        
    for item in batch:
        input_.append(torch.Tensor(item['input']))
        output_.append(torch.Tensor(item['output']))
    
    batch_input = padding_tensor(input_, batch[0]['null_input'])
    batch_output = padding_tensor(output_, batch[0]['null_output'])
    
    batch = {'input': batch_input.long(), 'output': batch_output.long()}
    
    return batch


def sort_for_rnn(x, null):
    lengths = torch.sum(x != null, dim=1).long()
    sorted_lengths, sorted_idx = torch.sort(lengths, dim=0, descending=True)
    sorted_lengths = sorted_lengths.data.tolist() 
    inverse_sorted_idx = torch.LongTensor(sorted_idx.shape).fill_(null)
    for i, v in enumerate(sorted_idx):
        inverse_sorted_idx[v.data] = i

    return x[sorted_idx], sorted_lengths, inverse_sorted_idx


class NLPDataset(Dataset):
    def __init__(self,
                 data,
                 input_tokenizer,
                 output_tokenizer):
        self.data = data
        self.data.index = range(len(data))
        self.input_tokenizer = input_tokenizer
        self.output_tokenizer = output_tokenizer        
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        input_data = self.data.loc[idx, 'query_string']
        output_data = self.data.loc[idx, 'output']
        
        x = self.input_tokenizer.encode(input_data).ids
        y = self.output_tokenizer.encode(output_data).ids
        
        null_input = self.input_tokenizer.get_vocab()['<pad>']
        null_output = self.output_tokenizer.get_vocab()['<pad>']
        
        sample = {'input': x, 'output': y, 'null_input': null_input, 'null_output':null_output}

        return sample


class Attn(nn.Module):
    def __init__(self, hidden_size):
        super(Attn, self).__init__()
        self.hidden_size = hidden_size
        self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
        self.v = nn.Parameter(torch.rand(hidden_size))
        stdv = 1. / math.sqrt(self.v.size(0))
        self.v.data.normal_(mean=0, std=stdv)

    def forward(self, encoder_outputs, hidden):
        seq_len = encoder_outputs.size(1)
        H = hidden.repeat(seq_len, 1, 1).transpose(0,1)
        attn_energies = self.score(H, encoder_outputs) # B*1*T
        return F.softmax(attn_energies, dim=2)

    def score(self, hidden, encoder_outputs):
        energy = torch.tanh(self.attn(torch.cat([hidden, encoder_outputs], 2))) # [B*T*2H]->[B*T*H]
        energy = energy.transpose(2,1) # [B*H*T]
        v = self.v.repeat(encoder_outputs.data.shape[0],1).unsqueeze(1) #[B*1*H]
        energy = torch.bmm(v, energy) # [B*1*T]
        return energy

    
class Seq2SeqAtt(nn.Module):
    def __init__(self,
      null_token=0,
      start_token=1,
      end_token=2,
      encoder_vocab_size=100,
      decoder_vocab_size=100,
      wordvec_dim=300,
      hidden_dim=256,
      rnn_num_layers=2,
      rnn_dropout=0,
    ):
        super().__init__()
        self.encoder_embed = nn.Embedding(encoder_vocab_size, wordvec_dim)
        self.encoder_rnn = nn.LSTM(wordvec_dim, hidden_dim, rnn_num_layers,
                                   dropout=rnn_dropout, batch_first=True)
        self.decoder_embed = nn.Embedding(decoder_vocab_size, wordvec_dim)
        self.decoder_rnn = nn.LSTM(wordvec_dim + hidden_dim, hidden_dim, rnn_num_layers,
                                   dropout=rnn_dropout, batch_first=True)
        self.decoder_linear = nn.Linear(hidden_dim, decoder_vocab_size)
        self.decoder_attn = Attn(hidden_dim)
        self.rnn_num_layers = rnn_num_layers
        self.NULL = null_token
        self.START = start_token
        self.END = end_token
        self.multinomial_outputs = None

    def expand_encoder_vocab(self, token_to_idx, word2vec=None, std=0.01):
        expand_embedding_vocab(self.encoder_embed, token_to_idx,
                               word2vec=word2vec, std=std)

    def get_dims(self, x=None, y=None):
        V_in = self.encoder_embed.num_embeddings
        V_out = self.decoder_embed.num_embeddings
        D = self.encoder_embed.embedding_dim
        H = self.encoder_rnn.hidden_size
        L = self.encoder_rnn.num_layers

        N = x.size(0) if x is not None else None
        N = y.size(0) if N is None and y is not None else N
        T_in = x.size(1) if x is not None else None
        T_out = y.size(1) if y is not None else None
        return V_in, V_out, D, H, L, N, T_in, T_out

    def encoder(self, x):
        x, x_lengths, inverse_index = sort_for_rnn(x, null=self.NULL)
        embed = self.encoder_embed(x)
        packed = pack_padded_sequence(embed, x_lengths, batch_first=True)
        out_packed, hidden = self.encoder_rnn(packed)
        out, _ = pad_packed_sequence(out_packed, batch_first=True)

        out = out[inverse_index]
        hidden = [h[:,inverse_index] for h in hidden]

        return out, hidden

    def decoder(self, word_inputs, encoder_outputs, prev_hidden):
        hn, cn = prev_hidden
        word_embedded = self.decoder_embed(word_inputs).unsqueeze(1) # batch x 1 x embed

        attn_weights = self.decoder_attn(encoder_outputs, hn[-1])
        context = attn_weights.bmm(encoder_outputs) # batch x 1 x hidden

        rnn_input = torch.cat((word_embedded, context), 2)
        output, hidden = self.decoder_rnn(rnn_input, prev_hidden)

        output = output.squeeze(1) # batch x hidden
        output = self.decoder_linear(output)

        return output, hidden

    def compute_loss(self, output_logprobs, y):
        self.multinomial_outputs = None
        V_in, V_out, D, H, L, N, T_in, T_out = self.get_dims(y=y)
        mask = y.data != self.NULL
        y_mask = Variable(torch.Tensor(N, T_out).fill_(0).type_as(mask))
        y_mask[:, 1:] = mask[:, 1:]
        y_masked = y[y_mask]
        out_mask = Variable(torch.Tensor(N, T_out).fill_(0).type_as(mask))
        out_mask[:, :-1] = mask[:, 1:]
        out_mask = out_mask.view(N, T_out, 1).expand(N, T_out, V_out)
        out_masked = output_logprobs[out_mask].view(-1, V_out)
        loss = F.cross_entropy(out_masked, y_masked)
        return loss

    def forward(self, x, y):
        max_target_length = y.size(1)

        encoder_outputs, encoder_hidden = self.encoder(x)
        decoder_inputs = y
        decoder_hidden = encoder_hidden
        decoder_outputs = []
        for t in range(max_target_length):
            decoder_out, decoder_hidden = self.decoder(
                decoder_inputs[:,t], encoder_outputs, decoder_hidden)
            decoder_outputs.append(decoder_out)

        decoder_outputs = torch.stack(decoder_outputs, dim=1)
        loss = self.compute_loss(decoder_outputs, y)
        return loss

    def sample(self, x, max_length=50):
        self.multinomial_outputs = None
        assert x.size(0) == 1, "Sampling minibatches not implemented"

        encoder_outputs, encoder_hidden = self.encoder(x)
        decoder_hidden = encoder_hidden
        sampled_output = [self.START]
        for t in range(max_length):
            decoder_input = Variable(torch.LongTensor([sampled_output[-1]]))
            decoder_out, decoder_hidden = self.decoder(
                decoder_input, encoder_outputs, decoder_hidden)
            _, argmax = decoder_out.data.max(1)
            output = argmax[0]
            sampled_output.append(output)
            if output == self.END:
                break

        return sampled_output
    
