from difflib import get_close_matches
from fuzzywuzzy import process


def align_2(text,text2, nlp):
    text_doc = nlp(text)
    text2_doc = nlp(text2) 
    window = len(text2_doc)
    chunks = [text_doc[i:i+window+j] for j in range(-2,4) for i in range(len(text_doc)-window+1+j)]
    chunks = ["".join(chunk.text_with_ws) for chunk in chunks]
    x = get_close_matches(text2,chunks,1)
    if len(x) > 0:
        return x[0].strip()
    else:
        return -1



def align_3(text,text2, nlp):
    text_doc = nlp(text)
    text2_doc = nlp(text2) 
    window = len(text2_doc)
    chunks = [text_doc[i:i+window+j] for j in range(-2,4) for i in range(len(text_doc)-window+1+j)]
    chunks = ["".join(chunk.text_with_ws) for chunk in chunks]
    x = process.extract(text2, chunks, limit=10)
    if len(x) > 0:
        return x[0][0]
    else:
        return -1
    
    

import torch
import transformers
import itertools


model = transformers.BertModel.from_pretrained('bert-base-multilingual-uncased')
tokenizer = transformers.BertTokenizerFast.from_pretrained('bert-base-multilingual-uncased')


def align_tokens3(sent_src,sent_tgt):



    token_src, token_tgt = [[tokenizer.tokenize(word)[0]] for word in sent_src], [[tokenizer.tokenize(word)[0]]  for word in sent_tgt]
    
    wid_src, wid_tgt = [tokenizer.convert_tokens_to_ids(x) for x in token_src], [tokenizer.convert_tokens_to_ids(x) for x in token_tgt]
    ids_src, ids_tgt = tokenizer.prepare_for_model(list(itertools.chain(*wid_src)), return_tensors='pt', model_max_length=tokenizer.model_max_length, truncation=True)['input_ids'], tokenizer.prepare_for_model(list(itertools.chain(*wid_tgt)), return_tensors='pt', truncation=True, model_max_length=tokenizer.model_max_length)['input_ids']
    sub2word_map_src = []

    sent_tgt.append("Not Found")
    for i, word_list in enumerate(token_src):
        sub2word_map_src += [i for x in word_list]
    sub2word_map_tgt = []
    for i, word_list in enumerate(token_tgt):
        sub2word_map_tgt += [i for x in word_list]

    # alignment
    align_layer = 8
    model.eval()


    with torch.no_grad():
        out_src = model(ids_src.unsqueeze(0), output_hidden_states=True)[2][align_layer][0, 1:-1]
        out_tgt = model(ids_tgt.unsqueeze(0), output_hidden_states=True)[2][align_layer][0, 1:-1]

        dot_prod = torch.matmul(out_src, out_tgt.transpose(-1, -2))

        align_words = set()
        max_indices_srctgt = torch.argmax(dot_prod, dim=-1)
        for i, j in enumerate(max_indices_srctgt):
            align_words.add( (sub2word_map_src[i], sub2word_map_tgt[j]) )

        align_words = list(align_words)
        x,y = list(zip(*align_words))
        for i, t in enumerate(sent_src):
            if i not in x:
                align_words.append((i,-1))
        return(sorted(align_words))



def find_token_spans(scope_tokens,arg_tokens):

    start = -1
    end = -1
    for i in range(len(scope_tokens)-len(arg_tokens) +1 ):
        if scope_tokens[i:i+len(arg_tokens)] == arg_tokens:
            start = i
            end = i + len(arg_tokens) - 1
            break
    return start,end


def wordAligner(scope_en,scope_pt,arg_en, nlp):
    
    scope_en_doc =  nlp(scope_en)
    scope_pt_doc = nlp(scope_pt)

    arg_en_doc = nlp(arg_en)
    sent_scope, sent_tgt, sent_arg = [t.text.lower() for t in scope_en_doc if t.text.strip() != ""], [t.text.lower() for t in scope_pt_doc if t.text.strip() != ""], [t.text.lower() for t in arg_en_doc if t.text.strip() != ""]

    start, end = find_token_spans(sent_scope,sent_arg)

    if start == -1: return None
    #print(sent_scope,sent_tgt)
    align_words = align_tokens3(sent_scope,sent_tgt)

    
    z = align_words[start:end+1]
    
    _, tgt_ = zip(*z)
    k = list(filter(lambda x: x!= -1 ,tgt_))
    if len(k) == 0: return
    start_tg = min(k)
    
    end_tgt = max(k)
    return ''.join([token.text_with_ws for token in scope_pt_doc][start_tg:end_tgt+1]).strip()

def word_align_safe(tgt,src, nlp):
    len_align, len_en = len([t.text for t in nlp(tgt)]), len([t.text for t in nlp(src)])
    range_ = 3 + int(len_en * 0.2)
    if not (len_align > len_en + range_ or len_align < len_en - range_):
        return True
    return None
    

def chooseArg(choices, en_arg, nlp):
    #choices = filter(lambda x: x!= -1,choices)
    if not choices[2]: # normalmente este caso corresponde a sujeito omitido ou a casos dificeis de alinhar
        return "", "not_found"
    if word_align_safe(choices[2],en_arg, nlp):
        return choices[2], "word_aligner"
    if choices[1] != -1:
        return choices[1], "gestalt"
    elif choices[0] != -1:
        return choices[0], "levenstein"
    return "", "not_found"



