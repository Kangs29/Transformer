
# coding: utf-8

# In[ ]:

import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

from google.colab import drive
drive.mount('/content/drive')

import os
import re
import math
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

def clean_str(string):
    # Yoon kim English Preprocessing Revise
    string = re.sub(r"[^A-Za-z0-9().,!?\'\`]", " ", string)     
    string = re.sub(r"\'s", " \'s", string) 
    string = re.sub(r"\'ve", " \'ve", string) 
    string = re.sub(r"n\'t", " n\'t", string) 
    string = re.sub(r"\'re", " \'re", string) 
    string = re.sub(r"\'d", " \'d", string) 
    string = re.sub(r"\'ll", " \'ll", string) 
    string = re.sub(r"\.", " .", string) 
    string = re.sub(r",", " , ", string) 
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string) 
    string = re.sub(r"\)", " \) ", string) 
    string = re.sub(r"\?", " \? ", string) 
    string = re.sub(r"\s{2,}", " ", string)    
    return string.strip()

# masking function
def mask_key(seq_k,seq_q):
    # seq_k, seq_q : embedding전 padding되어있는 문장의 sequence
    # [[1,2,0]] -> [[[0,0,1],[0,0,1],[0,0,1]]]
    
    length_query = seq_q.size()[1]
    mask_padding = seq_k.eq(0)                                          # .eq(0)에서 0은 
                                                                        # dictionary["PADDING"]=0
    mask_padding = mask_padding.unsqueeze(1).expand(-1,length_query,-1) # [batch, length, length]
    
    return mask_padding

def mask_seq(seq):
    # seq : embedding전 padding되어있는 문장의 sequence
    # [1,2,3,4,5,6,0,0,0,0,0] -> [1,1,1,1,1,1,0,0,0,0,0]
    
    return seq.ne(0).type(torch.float).unsqueeze(-1) # .ne(0)에서 0은 dictionary["PADDING"]=0


def mask_decoder(seq):
    # seq : embedding전 padding되어있는 문장의 sequence
    # [[1,2,0,0]]
    #
    # -> [[[0,1,1,1],
    #      [0,0,1,1],
    #      [0,0,0,1],
    #      [0,0,0,0]]]
    
    batch, length = seq.size()
    mask_deco = torch.triu(torch.ones((length,length),dtype=torch.uint8),diagonal=1).to(device)
    mask_deco = mask_deco.unsqueeze(0).expand(batch,-1,-1)
    
    return mask_deco


class EmbeddingWeight(nn.Module):
    def __init__(self, voca_size, model_dimension):
        super().__init__()
        self.embedding = nn.Embedding(voca_size, model_dimension)
        self.embedding.weight.data.uniform_(-0.01,0.01)
        
    def forward(self, x):
        return self.embedding(x)



class ScaledDotProductAttention(nn.Module):
    def __init__(self,MODEL_DIMENSION,ATTENTION_DROPOUT=0.1):
        super().__init__()
        
        self.scaling = math.sqrt(MODEL_DIMENSION)
        self.dropout = nn.Dropout(ATTENTION_DROPOUT)
        self.softmax = nn.Softmax(dim=2)
        
    def forward(self, q, k, v, mask=None):
        # Step 1 : Query와 Key의 상대적 중요도를 반영한 확률값을 구한다 - Self-attention
        # Step 1-1 : Query와 Key의 dot product를 진행한다. - Dot product attention
        # Step 1-2 : Dot product attention을 scaling한다 (model dimension에 따라 크기가 다르기때문에 이를 보정하는 역할)
        # Step 1-3 : Softmax를 통하여 Dot prodcut attention에서 얻은 값으로 Query 단어마다의 key와의 상대적 중요도를 구한다.

        # Step 2 : 상대적 중요도를 반영한 Value를 구한다.
        # Step 2-1 : Query에 대한 Key의 확률값을 Value와 Dot product를 통해 반영한다
        
        # Step 3 : 최종 값 Query가 Self-Attention을 반영한 새로운 단어의 표현이 된다. 
        #           -> [Batch size, Query(words of sentence), model_dimension]
        
        
        # Step 1
        # [batch_size, Query_length, model_dimension] X [batch_size, model_dimension, Key_length]
        Attention_DotProduct = torch.bmm(q,k.transpose(1,2)) # [batch_size, Query_length, Key_length] 
        Attention_Scaling = Attention_DotProduct / self.scaling
        
        if mask is not None:
            Attention_Scaling = Attention_Scaling.masked_fill_(mask,1e-9) # -float('inf')와 1e-9중 선택
            
        Attention_Softmax = F.softmax(Attention_Scaling,dim=2) # [batch_size, Query_length, Key_length] 
        Attention_Softmax = nn.Dropout(0.1)(Attention_Softmax) # [batch_size, Query_length, Key_length] 

        # Step 2
        # [batch_size, Query_length, Key_length] X [batch_size, Value_length, model_dimension]
        Output = torch.bmm(Attention_Softmax,v) # [batch_size, Query_length, model_dimension]

        # Step 3
        return Output, Attention_Softmax
    
class MultiHeadAttention(nn.Module):
    def __init__(self, NUMBER_HEAD, MODEL_DIMENSION, KEY_DIMENSION, VALUE_DIMENSION, MULTI_DROPOUT=0.1):
        super().__init__()
        
        self.NUMBER_HEAD = NUMBER_HEAD                # The number of HEAD
        self.QUERY_DIMENSION = KEY_DIMENSION          # Query Dimension == Key Dimension
        self.KEY_DIMENSION = KEY_DIMENSION            # Key Dimension
        self.VALUE_DIMENSION = VALUE_DIMENSION        # Value Dimension
        self.MODEL_DIMENSION = MODEL_DIMENSION        # Model dimension
        self.MULTI_DROPOUT = MULTI_DROPOUT
        
        self.Multi_Query = nn.Linear(self.MODEL_DIMENSION,self.NUMBER_HEAD*self.QUERY_DIMENSION) # Multi-Head Query
        self.Multi_Key = nn.Linear(self.MODEL_DIMENSION,self.NUMBER_HEAD*self.KEY_DIMENSION)     # Multi-Head Key
        self.Multi_Value = nn.Linear(self.MODEL_DIMENSION,self.NUMBER_HEAD*self.VALUE_DIMENSION) # Multi-Head Value
        nn.init.normal_(self.Multi_Query.weight, 
                        mean=0, std=math.sqrt(2.0 / (self.MODEL_DIMENSION + self.QUERY_DIMENSION)))
        nn.init.normal_(self.Multi_Key.weight, 
                        mean=0, std=math.sqrt(2.0 / (self.MODEL_DIMENSION + self.KEY_DIMENSION)))
        nn.init.normal_(self.Multi_Value.weight, 
                        mean=0, std=math.sqrt(2.0 / (self.MODEL_DIMENSION + self.VALUE_DIMENSION)))
        
        self.ScaledAttention = ScaledDotProductAttention(self.MODEL_DIMENSION) # Scaled Dot-Product Attetion
        self.Fully_Connected = nn.Linear(self.NUMBER_HEAD*self.VALUE_DIMENSION,self.MODEL_DIMENSION)
        nn.init.xavier_normal_(self.Fully_Connected.weight)
        
        self.Dropout = nn.Dropout(self.MULTI_DROPOUT)
        self.Layer_Norm = nn.LayerNorm(self.MODEL_DIMENSION)
        
        
    def forward(self, q, k, v, mask=None): # input으로 들어오는 q, k, v는 같은 값이다.
        # Step 1 : 문장에서 각 단어의 임베딩을 input으로 받아 query, key, value를 표현하고 residual을 만들어둔다.
        # Step 1-1 : 처음에 input값으로 들어오는 q, k, v는 모두 같은 값이다.
        # Step 1-2 : residual을 query, key, value를 만들기 전 임베딩 상태를 가지고 있는다.
        # Step 1-3 : query, key, value의 fully-connected를 지나 같은 값이였던 q, k, v는 각각 query, key, value로 각기 다른 표현을 나뉘게된다.
        
        # Step 2 : 각각의 query, key, value는 ScaledDotProductAttention을 지나 query값을 얻고 여러개의 MultiHead를 fully connected로 연산한다
        # Step 2-1 : 앞서 얻은 query, key, value를 self-Attention하기 위해 ScaledDotProductAttention을 지난다.
        # Step 2-2 : ScaledDotProduct를 통해 Head 개수만큼 얻은 것을 [Head 개수 X query 차원, model 차원]인 fully connected와 연산한다
        
        # Step 3 : dropout=0.1로 드랍아웃하고, residual을 더한 후 정규화하기 위해 layerNormalization을 사용한다.
        # Step 3-1 : dropout
        # Step 3-2 : Add residual
        # Step 3-3 : Layer normalization
        
        batch, length_query, _ = q.size()
        batch, length_key, _ = k.size()
        batch, length_value, _ = v.size()
        
        residual = q  # residual
        q = self.Multi_Query(q).view(batch, length_query, self.NUMBER_HEAD, self.QUERY_DIMENSION)
        k = self.Multi_Key(k).view(batch, length_key, self.NUMBER_HEAD, self.KEY_DIMENSION)
        v = self.Multi_Value(v).view(batch, length_value, self.NUMBER_HEAD, self.VALUE_DIMENSION)

        q = q.permute(2,0,1,3).contiguous().view(-1,length_query,self.QUERY_DIMENSION)   # [batch*NUMBER_HEAD, length_query, QUERY_DIMENSION]
        k = k.permute(2,0,1,3).contiguous().view(-1,length_key,self.KEY_DIMENSION)       # [batch*NUMBER_HEAD, length_key, KEY_DIMENSION]
        v = v.permute(2,0,1,3).contiguous().view(-1,length_value,self.VALUE_DIMENSION)   # [batch*NUMBER_HEAD, length_value, VALUE_DIMENSION]
        
        
        
        mask=mask.repeat(NUMBER_HEAD,1,1)
        Output, attention_softmax = self.ScaledAttention(q,k,v,mask=mask)

        Output = Output.view(self.NUMBER_HEAD,batch,length_query,self.VALUE_DIMENSION)
        Output = Output.permute(1,2,0,3).contiguous().view(batch, length_query,-1) # [batch, length_query, NUMBER_HEAD*VALUE_DIMENSION]

        Output = self.Dropout(self.Fully_Connected(Output))
        Output = self.Layer_Norm(Output+residual)

        
        return Output, attention_softmax
    
class PositionWiseFeedForward(nn.Module):
    def __init__(self,FULLY_CONNECTED_DIMENSION,MODEL_DIMENSION,POSITIONWISE_DROPOUT=0.1):
        super().__init__()
        self.FULLY_CONNECTED_DIMENSION = FULLY_CONNECTED_DIMENSION
        self.MODEL_DIMENSION = MODEL_DIMENSION
        self.POSITIONWISE_DROPOUT = POSITIONWISE_DROPOUT
        
        self.PositionWise_FFN1=nn.Linear(self.MODEL_DIMENSION,self.FULLY_CONNECTED_DIMENSION)
        self.PositionWise_FFN2=nn.Linear(self.FULLY_CONNECTED_DIMENSION,self.MODEL_DIMENSION)

        self.Layer_Norm = nn.LayerNorm(self.MODEL_DIMENSION)
        self.Dropout = nn.Dropout(self.POSITIONWISE_DROPOUT)

        
    def forward(self, MultiHead_output):
        # Step 1 : residual을 따로 두고 Multi-Head Attention 결과를 이용하여 Position-wise Feed-Forward Networks를 진행한다.
        # Step 1-1 : residual을 따로 만들어둔다.
        # Step 1-2 : Multi-Head Attention을 통해 얻은 결과값을 Fully connected layer을 거친다. (hidden = 2048)
        # Step 1-3 : ReLU를 지난다.
        # Step 1-4 : Fully connected layer를 마지막으로 거친다.
        
        # Step 2 : dropout=0.1로 드랍아웃하고, residual을 더한 후 정규화하기 위해 layerNormalization을 사용한다.
        # Step 2-1 : dropout
        # Step 2-2 : Add residual
        # Step 2-3 : Layer normalization
        
        residual = MultiHead_output
        
        FeedForward1 = self.PositionWise_FFN1(MultiHead_output)
        FeedForward1 = F.relu(FeedForward1)
        FeedForward2 = self.PositionWise_FFN2(FeedForward1)

        Output = self.Dropout(FeedForward2)
        Output = self.Layer_Norm(Output+residual)

        return Output
    
    
class EncoderLayer(nn.Module):
    def __init__(self,NUMBER_HEAD, MODEL_DIMENSION, KEY_DIMENSION, VALUE_DIMENSION,
                FULLY_CONNECTED_DIMENSION, MULTI_DROPOUT=0.1, POSITIONWISE_DROPOUT=0.1):
        super(EncoderLayer, self).__init__()
        
        self.NUMBER_HEAD = NUMBER_HEAD
        self.MODEL_DIMENSION = MODEL_DIMENSION
        self.KEY_DIMENSION = KEY_DIMENSION
        self.VALUE_DIMENSION = VALUE_DIMENSION
        self.MULTI_DROPOUT = MULTI_DROPOUT
        
        self.FULLY_CONNECTED_DIMENSION = FULLY_CONNECTED_DIMENSION
        self.POSITIONWISE_DROPOUT = POSITIONWISE_DROPOUT
        
        
        self.MultiHead =  MultiHeadAttention(self.NUMBER_HEAD,self.MODEL_DIMENSION,
                                             self.KEY_DIMENSION,self.VALUE_DIMENSION,self.MULTI_DROPOUT)
        self.PositionFFN = PositionWiseFeedForward(self.FULLY_CONNECTED_DIMENSION, self.MODEL_DIMENSION,
                                                   self.POSITIONWISE_DROPOUT)
        
    def forward(self, encoder_input, non_pad_mask=None, MultiHead_mask=None):
        encoder_output, encoder_attention = self.MultiHead(q=encoder_input,
                                                   k=encoder_input,
                                                   v=encoder_input,
                                                   mask = MultiHead_mask)

        encoder_output *= non_pad_mask
        
        encoder_output = self.PositionFFN(encoder_output)
        encoder_output *= non_pad_mask
        
        return encoder_output, encoder_attention
    
class DecoderLayer(nn.Module):
    def __init__(self,NUMBER_HEAD, MODEL_DIMENSION, KEY_DIMENSION, VALUE_DIMENSION,
                FULLY_CONNECTED_DIMENSION, MULTI_DROPOUT=0.1, POSITIONWISE_DROPOUT=0.1):
        super(DecoderLayer, self).__init__()
        
        self.NUMBER_HEAD = NUMBER_HEAD
        self.MODEL_DIMENSION = MODEL_DIMENSION
        self.KEY_DIMENSION = KEY_DIMENSION
        self.VALUE_DIMENSION = VALUE_DIMENSION
        self.MULTI_DROPOUT = MULTI_DROPOUT
        
        self.FULLY_CONNECTED_DIMENSION = FULLY_CONNECTED_DIMENSION
        self.POSITIONWISE_DROPOUT = POSITIONWISE_DROPOUT
        
        
        self.De_MultiHead =  MultiHeadAttention(self.NUMBER_HEAD,self.MODEL_DIMENSION,
                                             self.KEY_DIMENSION,self.VALUE_DIMENSION,self.MULTI_DROPOUT)
        self.En_MultiHead =  MultiHeadAttention(self.NUMBER_HEAD,self.MODEL_DIMENSION,
                                             self.KEY_DIMENSION,self.VALUE_DIMENSION,self.MULTI_DROPOUT)
        self.PositionFFN = PositionWiseFeedForward(self.FULLY_CONNECTED_DIMENSION, self.MODEL_DIMENSION,
                                                   self.POSITIONWISE_DROPOUT)
        
    def forward(self, decoder_input, encoder_output, 
                non_pad_mask=None, De_MultiHead_mask=None, En_MultiHead_mask=None):
        
        decoder_output, decoder_attention = self.De_MultiHead(q=decoder_input,
                                                   k=decoder_input,
                                                   v=decoder_input,
                                                   mask = De_MultiHead_mask)
        decoder_output *= non_pad_mask

        
        decoder_output, decoder_attention = self.En_MultiHead(q=decoder_input,
                                                   k=encoder_output,
                                                   v=encoder_output,
                                                   mask = En_MultiHead_mask)
        decoder_output *= non_pad_mask
        
        decoder_output = self.PositionFFN(decoder_output)
        decoder_output *= non_pad_mask
        
        return decoder_output, decoder_attention


class PositionalEncoder(nn.Module):
    def __init__(self, MODEL_DIMENSION, MAX_LENGTH):
        super().__init__()

        self.MODEL_DIMENSION = MODEL_DIMENSION
        self.MAX_LENGTH = MAX_LENGTH
        
    def forward(self):
        get_pos = torch.tensor([[pos/(10000**((2*i)/self.MODEL_DIMENSION)) 
                                 for i in range(self.MODEL_DIMENSION)]
                                 for pos in range(self.MAX_LENGTH)]).to(device)
        position = torch.zeros(self.MAX_LENGTH,self.MODEL_DIMENSION).to(device)
        position[:,0::2] += torch.sin(get_pos[:,0::2])
        position[:,1::2] += torch.cos(get_pos[:,1::2])
        
        return position
    
class Encoder(nn.Module):
    def __init__(self,ENCODER_VOCA_SIZE,MODEL_DIMENSION,NUMBER_HEAD,
                KEY_DIMENSION,VALUE_DIMENSION,FULLY_CONNECTED_DIMENSION,N_LAYER,MAX_LENGTH):
        super(Encoder,self).__init__()

        self.ENCODER_VOCA_SIZE = ENCODER_VOCA_SIZE
        self.MODEL_DIMENSION = MODEL_DIMENSION
        self.NUMBER_HEAD = NUMBER_HEAD
        self.KEY_DIMENSION = KEY_DIMENSION
        self.VALUE_DIMENSION = VALUE_DIMENSION
        self.FULLY_CONNECTED_DIMENSION = FULLY_CONNECTED_DIMENSION
        self.N_LAYER = N_LAYER
        self.MAX_LENGTH = MAX_LENGTH
        
        self.En_Embedding = EmbeddingWeight(self.ENCODER_VOCA_SIZE,self.MODEL_DIMENSION)
        
        self.positional_encoding = PositionalEncoder(self.MODEL_DIMENSION, self.MAX_LENGTH)
        self.Encoder_ = nn.ModuleList([EncoderLayer(self.NUMBER_HEAD, self.MODEL_DIMENSION, self.KEY_DIMENSION, 
                                self.VALUE_DIMENSION,self.FULLY_CONNECTED_DIMENSION) for _ in range(N_LAYER)])

    def forward(self,seq_encoder):
        att_mask = mask_key(seq_encoder,seq_encoder)
        non_mask = mask_seq(seq_encoder)
        
        # [batch size, max_length, dimension] + [max_length, dimension]
        encoder_output = self.En_Embedding(seq_encoder) + self.positional_encoding()


        # enc_slf_attn_list = []?????
        for enc_layer in self.Encoder_:
            encoder_output, encoder_attention = enc_layer(encoder_output,
                                                          non_pad_mask=non_mask,
                                                          MultiHead_mask = att_mask)
            
        return encoder_output

    
class Decoder(nn.Module):
    def __init__(self,DECODER_VOCA_SIZE,MODEL_DIMENSION,NUMBER_HEAD,
                 KEY_DIMENSION,VALUE_DIMENSION,FULLY_CONNECTED_DIMENSION,N_LAYER,MAX_LENGTH):
        super(Decoder,self).__init__()
        
        self.DECODER_VOCA_SIZE = DECODER_VOCA_SIZE
        self.MODEL_DIMENSION = MODEL_DIMENSION
        self.NUMBER_HEAD = NUMBER_HEAD
        self.KEY_DIMENSION = KEY_DIMENSION
        self.VALUE_DIMENSION = VALUE_DIMENSION
        self.FULLY_CONNECTED_DIMENSION = FULLY_CONNECTED_DIMENSION
        self.N_LAYER = N_LAYER
        self.MAX_LENGTH = MAX_LENGTH

        self.De_Embedding = EmbeddingWeight(self.DECODER_VOCA_SIZE,self.MODEL_DIMENSION)

        self.positional_encoding = PositionalEncoder(self.MODEL_DIMENSION, self.MAX_LENGTH)
        self.Decoder_ = nn.ModuleList([DecoderLayer(self.NUMBER_HEAD, self.MODEL_DIMENSION, self.KEY_DIMENSION, 
                                self.VALUE_DIMENSION,self.FULLY_CONNECTED_DIMENSION) for _ in range(self.N_LAYER)])


    def forward(self,seq_decoder,seq_encoder,encoder_output):
        non_mask = mask_seq(seq_decoder)

        deco_mask = mask_decoder(seq_decoder)
        att_mask = mask_key(seq_decoder,seq_decoder)
        attn_mask = (deco_mask+att_mask).gt(0)

        de_en_mask = mask_key(seq_encoder,seq_decoder)

        # [batch size, max_length, dimension] + [max_length, dimension]
        decoder_output = self.De_Embedding(seq_decoder) + self.positional_encoding()
        # enc_slf_attn_list = []?????
        for dec_layer in self.Decoder_:
            decoder_output, decoder_attention = dec_layer(decoder_output,
                                                          encoder_output,
                                                          non_pad_mask=non_mask,
                                                          De_MultiHead_mask = attn_mask,
                                                          En_MultiHead_mask = de_en_mask)
        
        return decoder_output

class Transformer(nn.Module):
    def __init__(self,MODEL_DIMENSION,ENCODER_VOCA_SIZE,DECODER_VOCA_SIZE,NUMBER_HEAD,
                KEY_DIMENSION,VALUE_DIMENSION,FULLY_CONNECTED_DIMENSION,N_LAYER,MAX_LENGTH):
        super(Transformer,self).__init__()
        
        self.ENCODER_VOCA_SIZE = ENCODER_VOCA_SIZE
        self.DECODER_VOCA_SIZE = DECODER_VOCA_SIZE
        self.MODEL_DIMENSION = MODEL_DIMENSION
        self.NUMBER_HEAD = NUMBER_HEAD
        self.KEY_DIMENSION = KEY_DIMENSION
        self.VALUE_DIMENSION = VALUE_DIMENSION
        self.FULLY_CONNECTED_DIMENSION = FULLY_CONNECTED_DIMENSION
        self.N_LAYER = N_LAYER
        self.MAX_LENGTH = MAX_LENGTH

        self.encoder = Encoder(self.ENCODER_VOCA_SIZE,self.MODEL_DIMENSION,self.NUMBER_HEAD,
                               self.KEY_DIMENSION,self.VALUE_DIMENSION,self.FULLY_CONNECTED_DIMENSION,
                               self.N_LAYER,self.MAX_LENGTH)
        self.decoder = Decoder(self.DECODER_VOCA_SIZE,self.MODEL_DIMENSION,self.NUMBER_HEAD,
                               self.KEY_DIMENSION,self.VALUE_DIMENSION,self.FULLY_CONNECTED_DIMENSION,
                               self.N_LAYER,self.MAX_LENGTH)
        self.FC = nn.Linear(self.MODEL_DIMENSION,self.DECODER_VOCA_SIZE)
        torch.nn.init.xavier_uniform_(self.FC.weight)
        
    def forward(self,seq_encoder, seq_decoder):
        encoder_output = self.encoder(seq_encoder)
        decoder_output = self.decoder(seq_decoder,seq_encoder,encoder_output)
        logit = self.FC(decoder_output)
        
        return logit
    
import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

from google.colab import drive
drive.mount('/content/drive')

import re

def measure_token_length(token):
    if token[-4:] == '</w>':
        return len(token[:-4]) + 1
    else:
        return len(token)


def tokenize_word(string, sorted_tokens, unknown_token='</u>'):
    
    if string == '':
        return []
    if sorted_tokens == []:
        return [unknown_token]

# 이 함수안에 넣는 것은 아니고 필요할 경우, if-else문으로 tokenized_word함수를 사용할 지, 안할지 결정
# -> 모델에 사용 시
#    if string in sorted_tokens:
#        return [string]  # 이거 안할수없나... 나중에 찾아보자
    
    string_tokens = []
    for i in range(len(sorted_tokens)):
        token = sorted_tokens[i]
        token_reg = re.escape(token.replace('.', '[.]')) # . -> [.], 
        
        matched_positions = [(m.start(0), m.end(0)) for m in re.finditer(token_reg, string)]
        if len(matched_positions) == 0:
            continue
        substring_end_positions = [matched_position[0] for matched_position in matched_positions]

        substring_start_position = 0
        for substring_end_position in substring_end_positions:
            substring = string[substring_start_position:substring_end_position]
            string_tokens += tokenize_word(string=substring, sorted_tokens=sorted_tokens[i+1:], unknown_token=unknown_token)
            string_tokens += [token]
            substring_start_position = substring_end_position + len(token)
        remaining_substring = string[substring_start_position:]
        string_tokens += tokenize_word(string=remaining_substring, sorted_tokens=sorted_tokens[i+1:], unknown_token=unknown_token)
        break
    return string_tokens

def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Every dataset is lower cased except for TREC
    """
    string = re.sub(r"[^A-Za-z0-9(),!.?\'\`]", " ", string)     
    string = re.sub(r"\'s", " \'s", string) 
    string = re.sub(r"\'ve", " \'ve", string) 
    string = re.sub(r"n\'t", " n\'t", string) 
    string = re.sub(r"\'re", " \'re", string) 
    string = re.sub(r"\'d", " \'d", string) 
    string = re.sub(r"\'ll", " \'ll", string) 
    string = re.sub(r"\,", " \, ", string) 
    string = re.sub(r"\.", " \. ", string) 
    string = re.sub(r"\!", " \! ", string) 
    string = re.sub(r"\(", " \( ", string) 
    string = re.sub(r"\)", " \) ", string) 
    string = re.sub(r"\?", " \? ", string) 
    string = re.sub(r"\s{2,}", " ", string)    
    return string.strip()

class Constant():
    def pad():
        return 0
    def sos():
        return 1
    def eos():
        return 2


vocab_en_={}
with open('/content/drive/My Drive/Tranformer_BPE_English.txt','r',encoding='utf-8') as f:
    for lines in f.readlines():
        lines = lines.split('\t')
        word = lines[0]
        freq = int(lines[1])

        vocab_en_[word] = freq 


sorted_tokens_tuple = sorted(vocab_en_.items(), 
                             key=lambda item: (measure_token_length(item[0]), item[1]), reverse=True)
sorted_tokens = [token for (token, freq) in sorted_tokens_tuple]

vocab_en={}
vocab_en['#PAD'] = 0
vocab_en["<SOS>"] = 1
vocab_en["<EOS>"] = 2
for lines in sorted_tokens:
    vocab_en[lines]=len(vocab_en)


vocab_de_={}
with open('/content/drive/My Drive/Tranformer_BPE_Deutsch.txt','r',encoding='utf-8') as f:
    for lines in f.readlines():
        lines = lines.split('\t')
        word = lines[0]
        freq = int(lines[1])

        vocab_de_[word] = freq 


sorted_tokens_tuple = sorted(vocab_de_.items(), 
                             key=lambda item: (measure_token_length(item[0]), item[1]), reverse=True)
sorted_tokens = [token for (token, freq) in sorted_tokens_tuple]

vocab_de={}
vocab_de['#PAD'] = 0
vocab_de["<SOS>"] = 1
vocab_de["<EOS>"] = 2
for lines in sorted_tokens:
    vocab_de[lines]=len(vocab_de)


# In[ ]:

#filenames = ["commoncrawl.de-en.en","europarl-v7.de-en.en","news-commentary-v9.de-en.en"]
filenames = ["commoncrawl.de-en.en","europarl-v7.de-en.en"]
data_en=[]
for filename in filenames:
    print(filename)
    with open('/content/drive/My Drive/'+filename,'r',encoding='utf-8') as f:
        for lines in f.readlines():
            data_en.append(clean_str(lines))
            
filenames = ["commoncrawl.de-en.de","europarl-v7.de-en.de"]
data_de=[]
for filename in filenames:
    print(filename)
    with open('/content/drive/My Drive/'+filename,'r',encoding='utf-8') as f:
        for lines in f.readlines():
            data_de.append(clean_str(lines))
            
import random

def slice_(word,x,vocab_):
    lens=len(word)
    if lens == 0:
        return x
    for i in range(lens):
        slice_word=word[:(lens-i)]
        if slice_word in vocab_:
            x.append(slice_word)
            break
    word = word[(lens-i):]
    slice_(word,x,vocab_)

def Sentence_to_token(sentence,vocab_):
    slice_sentence = []
    for word in sentence.split():
        word = word.lower()+'</w>'
        slice_(word,slice_sentence,vocab_)
    return slice_sentence


data=[]
length = 50
j=0
for i in range(len(data_en)):
    enco=data_en[i]
    deco=data_de[i]
    j+=1
    if len(Sentence_to_token(enco,vocab_en))>=length:
        continue
    if len(Sentence_to_token(deco,vocab_de))>=length:
        continue
    data.append([data_en[i],data_de[i]])
#    if j % 1000000 == 0:
#        print(j)


# In[ ]:

#train
# data setting
length = 50
ENCODER_WORDDICT = vocab_en
DECODER_WORDDICT = vocab_de 


# hyperparameter
ENCODER_VOCA_SIZE = len(ENCODER_WORDDICT)
DECODER_VOCA_SIZE = len(DECODER_WORDDICT)
MODEL_DIMENSION = 512
NUMBER_HEAD = 8
KEY_DIMENSION = 64
VALUE_DIMENSION = 64
FULLY_CONNECTED_DIMENSION = 2048
N_LAYER = 3
MAX_LENGTH=length

#checkpoint= torch.load("/content/drive/My Drive/Transformer-200113-8.pth")
model=Transformer(MODEL_DIMENSION,ENCODER_VOCA_SIZE,DECODER_VOCA_SIZE,NUMBER_HEAD,
                KEY_DIMENSION,VALUE_DIMENSION,FULLY_CONNECTED_DIMENSION,N_LAYER,MAX_LENGTH)
model.to(device)
#model.load_state_dict(checkpoint['model'])

parameters = filter(lambda p: p.requires_grad, model.parameters())
loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr =5e-5, betas=(0.9, 0.999), eps=1e-09, weight_decay=0)
#optimizer.load_state_dict(checkpoint['optimizer'])

#for param_group in optimizer.param_groups:
#    param_group['lr']=0.000001


import time
t1=time.time()
MAX_LENGTH=length
batch_size = 50
iterations = 1000000
for iters in range(iterations):

    batch_data = random.sample(data,batch_size)
    batch_encoder_, batch_decoder_ = zip(*batch_data)

    # 하기전에 length 이상인건 다 쳐내놓고 시작해야됨

    batch_encoder=[]
    for lines in batch_encoder_:
        edit = Sentence_to_token(lines,vocab_en)
        tokens=[]
        for token in edit:
            tokens.append(vocab_en[token])
        batch_encoder.append(tokens+[Constant.eos()]+[Constant.pad()]*(length-len(tokens)-1))

    batch_decoder=[]
    for lines in batch_decoder_:
        edit = Sentence_to_token(lines,vocab_de)
        tokens=[]
        for token in edit:
            tokens.append(vocab_de[token])
        batch_decoder.append([Constant.sos()]+tokens+[Constant.eos()]+[Constant.pad()]*(length-len(tokens)-1))

    batch_encoder = torch.tensor(batch_encoder).to(device)
    batch_decoder = torch.tensor(batch_decoder).to(device)
    batch_decoder_input = batch_decoder[:,:length]
    batch_decoder_output = batch_decoder[:,1:]


    logit = model(batch_encoder, batch_decoder_input)
    loss = loss_function(logit.view(-1,DECODER_VOCA_SIZE),batch_decoder_output.contiguous().view(-1))
    
    optimizer.zero_grad()
    loss.backward()
    
    optimizer.step()

    if iters % 1000 == 0:
        print(round(iters*batch_size/len(data),4)*100,"%", "Time :",int(time.time()-t1))
        print("Iteration :",iters,"Loss :",float(loss),'\n')
        print('ANSWER :\n', batch_decoder_input[:2])
        print('PREDICT :\n',torch.max(F.softmax(logit,dim=2),2)[1][:2],'\n\n')
        t1=time.time()


# In[ ]:

#model save
torch.save({'model': model.state_dict(),
            'optimizer': optimizer.state_dict()}, "/content/drive/My Drive/Transformer-200103-15.pth")


# In[ ]:

# Testing
import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

from google.colab import drive
drive.mount('/content/drive')

import os
import re
import math
import time
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

get_ipython().system('pip install nltk==3.4.5')
import nltk
print(nltk.__version__)


def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Every dataset is lower cased except for TREC
    """
    string = re.sub(r"[^A-Za-z0-9(),!.?\'\`]", " ", string)     
    string = re.sub(r"\'s", " \'s", string) 
    string = re.sub(r"\'ve", " \'ve", string) 
    string = re.sub(r"n\'t", " n\'t", string) 
    string = re.sub(r"\'re", " \'re", string) 
    string = re.sub(r"\'d", " \'d", string) 
    string = re.sub(r"\'ll", " \'ll", string) 
    string = re.sub(r"\,", " \, ", string) 
    string = re.sub(r"\.", " \. ", string) 
    string = re.sub(r"\!", " \! ", string) 
    string = re.sub(r"\(", " \( ", string) 
    string = re.sub(r"\)", " \) ", string) 
    string = re.sub(r"\?", " \? ", string) 
    string = re.sub(r"\s{2,}", " ", string)    
    return string.strip()


class Constant():
    def pad():
        return 0
    def sos():
        return 1
    def eos():
        return 2


def measure_token_length(token):
    if token[-4:] == '</w>':
        return len(token[:-4]) + 1
    else:
        return len(token)
    
    
def BPE_vocab(filename):

    # Loading the vocab file
    vocab_={}
    with open('/content/drive/My Drive/%s.txt' % filename,'r',encoding='utf-8') as f:
        for lines in f.readlines():
            lines = lines.split('\t')
            word = lines[0]
            freq = int(lines[1])

            vocab_[word] = freq 


    # Sorting by frequency
    sorted_tokens_tuple = sorted(vocab_.items(), 
                                key=lambda item: (measure_token_length(item[0]), item[1]), reverse=True)
    sorted_tokens = [token for (token, freq) in sorted_tokens_tuple]

    # Making a vocab
    vocab={}
    vocab['#PAD'] = Constant.pad()
    vocab["<SOS>"] = Constant.sos()
    vocab["<EOS>"] = Constant.eos()
    for lines in sorted_tokens:
        vocab[lines]=len(vocab)

    del vocab_
    return vocab

def BPE_slice(word,return_set,vocab):
    # 단어를 Byte-Pair Encoding으로 slice하는 함수
    word_len = len(word)
    if word_len == 0:
        return return_set
    for len_ in range(word_len):
        slice_word_set = word[:(word_len-len_)]
        if slice_word_set in vocab:
            return_set.append(slice_word_set)
            break
    word = word[(word_len-len_):]
    BPE_slice(word,return_set,vocab)

def Sentence_to_token(sentence,vocab):
    # 문장을 Byte-Pair Encoding으로 slice하는 함수
    slice_sentence = []
    for word in sentence.split():
        word = word.lower()+'</w>'
        BPE_slice(word,slice_sentence,vocab)
    return slice_sentence

def TestData(filename_encoder,filename_decoder):
  
    # encoder-testset
    test_open=open("/content/drive/My Drive/%s.sgm" % filename_encoder,'r',encoding='utf-8')
    test_raw=[]
    for lines in test_open.readlines():
        test_raw.append(lines)

    test_encoder=[]
    for lines in test_raw:
        if '</seg>' in lines:
            test_encoder.append(clean_str(lines.split('>')[1][:-5]))

    # decoder-testset
    test_open=open("/content/drive/My Drive/%s.sgm" % filename_decoder,'r',encoding='utf-8')
    test_raw=[]
    for lines in test_open.readlines():
        test_raw.append(lines)

    test_decoder=[]
    for lines in test_raw:
        if '</seg>' in lines:
            test_decoder.append(clean_str(lines.split('>')[1][:-5]))

    test_data=[]
    for num in range(len(test_encoder)):
        test_data.append([test_encoder[num],test_decoder[num]])

    return test_data

# LOADING TESTING SETTING
filename="Tranformer_BPE_English"
vocab_encoder = BPE_vocab(filename)
filename="Tranformer_BPE_Deutsch"
vocab_decoder = BPE_vocab(filename)

filename_encoder="newstest2014-deen-ref.en"
filename_decoder="newstest2014-deen-ref.de"
test_data = TestData(filename_encoder,filename_decoder)

# data setting
length = 50
ENCODER_WORDDICT = vocab_encoder
DECODER_WORDDICT = vocab_decoder

decoder_index = {}
for i in DECODER_WORDDICT:
    decoder_index[DECODER_WORDDICT[i]]=i

# hyperparameter
ENCODER_VOCA_SIZE = len(ENCODER_WORDDICT)
DECODER_VOCA_SIZE = len(DECODER_WORDDICT)
MODEL_DIMENSION = 512
NUMBER_HEAD = 8
KEY_DIMENSION = 64
VALUE_DIMENSION = 64
FULLY_CONNECTED_DIMENSION = 2048
N_LAYER = 6
MAX_LENGTH=50


# In[ ]:

import numpy as np
def bleu_scores(hypo,infer,epsilon=0.1,weight = [0.25,0.25,0.25,0.25]):

    hypo=hypo.split()
    hypo_set=[]
    for nu in range(len(hypo)):
        hypo_set.append(hypo[nu])
    for nu in range(len(hypo)-1):
        hypo_set.append(hypo[nu:nu+2])
    for nu in range(len(hypo)-2):
        hypo_set.append(hypo[nu:nu+3])
    for nu in range(len(hypo)-3):
        hypo_set.append(hypo[nu:nu+4])


    infer=infer.split()
    n1=[]; n2=[]; n3=[]; n4=[]
    for nu in range(len(infer)):
        if infer[nu] in hypo_set:
            n1.append(1)
        else:
            n1.append(0)

    for nu in range(len(infer)-1):
        if infer[nu:nu+2] in hypo_set:
            n2.append(1)
        else:
            n2.append(0)

    for nu in range(len(infer)-2):
        if infer[nu:nu+3] in hypo_set:
            n3.append(1)
        else:
            n3.append(0)

    for nu in range(len(infer)-3):
        if infer[nu:nu+4] in hypo_set:
            n4.append(1)
        else:
            n4.append(0)

    if len(n4)!=0:
        score= pow((sum(n1)+epsilon)/len(n1),weight[0]*4/3)*        pow((sum(n2)+epsilon)/len(n2),weight[1])*            pow((sum(n3)+epsilon)/len(n3),weight[2])*            pow((sum(n4)+epsilon)/len(n4),weight[3])


    if len(n4)==0 and len(n3)!=0: 
        score= pow((sum(n1)+epsilon)/len(n1),weight[0]*4/3)*          pow((sum(n2)+epsilon)/len(n2),weight[1]*4/3)*            pow((sum(n3)+epsilon)/len(n3),weight[2]*4/3)
          
    if len(n3)==0 and len(n2)!=0: 
        score= pow((sum(n1)+epsilon)/len(n1),weight[0]*4/2)*          pow((sum(n2)+epsilon)/len(n2),weight[1]*4/2)

    if len(n2)==0: 
        score= pow((sum(n1)+epsilon)/len(n1),weight[0]*4)


    return score*np.exp(1-len(infer)/len(hypo))


# In[ ]:

def BLUE_score(test_data,ENCODER_WORDDICT,DECODER_WORDDICT,decoder_index,sequence_length=50,BPE_TEST=True):

    global model

    BLUE = []
    BATCH = 30
    iters = int(len(test_data)/BATCH)

    test_e,test_d = zip(*test_data)
    data_encoder=[]
    for lines in test_e:
        edit = Sentence_to_token(lines,ENCODER_WORDDICT)
        tokens=[]
        for token in edit:
            tokens.append(ENCODER_WORDDICT[token])
        
        if len(tokens) < sequence_length:
            data_encoder.append(tokens+[Constant.eos()]+[Constant.pad()]*(length-len(tokens)-1))
        else:
            data_encoder.append(tokens[:sequence_length])

    data_decoder = test_d


    for num in range(iters-1):
        test_D=[[[DECODER_WORDDICT["<SOS>"]]+[0]*(sequence_length-1)] for _ in range(BATCH)]
        test_E=torch.tensor(data_encoder[BATCH*num:BATCH*(num+1)]).to(device)
        test_D=torch.tensor(test_D).to(device)
        test_D=test_D.squeeze(1)


        for i in range(sequence_length-1):
            logit = model(test_E,test_D)
            p=torch.max(F.softmax(logit,dim=2),2)[1]
            test_D[:,i+1]+=p[:,i]


        if BPE_TEST==True:
            for i in range(BATCH):

                if "#PAD" in [decoder_index[int(k)] for k in test_D[i]]:
                    if "<EOS>" in [decoder_index[int(k)] for k in test_D[i]]:
                        idx = min([decoder_index[int(k)] for k in test_D[i]].index("#PAD"),[decoder_index[int(k)] for k in test_D[i]].index("<EOS>"))
                    else:
                        idx = [decoder_index[int(k)] for k in test_D[i]].index("#PAD")

                else:
                    if "<END>" in [decoder_index[int(k)] for k in test_D[i]]:
                        idx = [decoder_index[int(k)] for k in test_D[i]].index("<EOS>")
                    else:
                        idx = 50

#                predict = [decoder_index[int(k)] for k in test_D[i]][1:idx]
                predict = " ".join([decoder_index[int(i)] for i in test_D[i]][1:idx])
#                chencherry = SmoothingFunction()
#                score = nltk.translate.bleu_score.sentence_bleu([predict], Sentence_to_token(data_decoder[BATCH*num+i].lower(),DECODER_WORDDICT)+['EOS'],
#                                                                         smoothing_function=chencherry.method1)
#                score = bleu_scores(data_decoder[BATCH*num+i].lower()+" <EOS>"," ".join(predict)+" <EOS>")
                score = bleu_scores(" ".join(Sentence_to_token(data_decoder[BATCH*num+i].lower(),DECODER_WORDDICT)+[" <EOS>"]),predict+" <EOS>")
                print('corpus :',score)
                BLUE.append(score)
                if i==0:
                    print(predict,'\n')
                    print(data_decoder[BATCH*num+i])
                    print("BLUE SCORE :",sum(BLUE)/(len(BLUE)))

        else:
            for i in range(BATCH):
                predict = " ".join("".join([decoder_index[int(i)] for i in test_D[i]])[5:].split('</w>')[:-1])+" <EOS>"
                
#                chencherry = SmoothingFunction()
#                score = nltk.translate.bleu_score.sentence_bleu([predict.split()], data_decoder[BATCH*num+i].lower().split(),
#                                                                         smoothing_function=chencherry.method1)
                score = bleu_scores(data_decoder[BATCH*num+i].lower()+" <EOS>",predict)
                print('corpus :',score)
                BLUE.append(score)
                if i==0:
                    print(predict,'\n')
                    print(data_decoder[BATCH*num+i])
                    print("BLUE SCORE :",sum(BLUE)/(len(BLUE)))

    print("BLUE SCORE except < 0.001 :",sum(BLUE)/(len(BLUE)-sum(np.array(BLUE)<0.001)))
    print("BLUE SCORE :",sum(BLUE)/(len(BLUE)))

BLUE_score(test_data,ENCODER_WORDDICT,DECODER_WORDDICT,decoder_index,BPE_TEST=True)

