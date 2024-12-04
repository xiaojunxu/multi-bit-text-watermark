# Refer to: ~/anaconda3/lib/python3.11/site-packages/transformers/generation/utils.py
import numpy as np
import torch

def gen_split_tokens(tokenizer):
    # Current solution: tokens of EOS punctuations will be viewed as splitting.
    if 'OPT' in type(tokenizer).__name__ or 'GPT' in type(tokenizer).__name__:
        split_tokens = []
        #for eos in ['.', '?', '!']:
        for eos in ['.', '?', '!', ':']:
            tok = tokenizer.encode(eos)[1]
            split_tokens.append(tok)
    elif 'Llama' in type(tokenizer).__name__:
        split_tokens = []
        #for eos in ['.', '?', '!']:
        for eos in ['.', '?', '!', ':']:
            tok = tokenizer.encode(eos)[-1]
            split_tokens.append(tok)
            tok = tokenizer.encode('Yes%s'%eos)[-1] # llama tokenizer has different token for the punctuations in sentence
            split_tokens.append(tok)
            #print ("eos: %s, tok: %s"%(eos, tok))
    elif 'Gemma' in type(tokenizer).__name__:
        split_tokens = []
        for eos in ['.', '?', '!', ':']:
            tok = tokenizer.encode(eos)[1]
            split_tokens.append(tok)
    else:
        raise NotImplementedError(type(tokenizer).__name__)
    return split_tokens


def DM_generate_with_key(model0, model1, key, split_tokens, input_ids, attention_mask, max_length, pad_token_id, do_sample=True, use_cache=False):
    unfinished_sequences = torch.ones(input_ids.shape[0], dtype=torch.long, device=model0.device)
    eos_token_id_tensor = torch.tensor([pad_token_id]).to(model0.device)
    key_idx = [0 for _ in range(input_ids.shape[0])]  # which key the i-th sentence is on
    prompt_len = input_ids.shape[1]

    last_st = [prompt_len for _ in range(input_ids.shape[0])]
    last_ed = [-1 for _ in range(input_ids.shape[0])]
    split_info = [[] for _ in range(input_ids.shape[0])]
    while True:
        # TODO: can be done in parallel
        input_ids_m0, attention_mask_m0 = input_ids.to(model0.device), attention_mask.to(model0.device)
        input_ids_m1, attention_mask_m1 = input_ids.to(model1.device), attention_mask.to(model1.device)
        outputs0 = model0(input_ids=input_ids_m0, attention_mask=attention_mask_m0, past_key_values=None, use_cache=use_cache, return_dict=True)
        outputs1 = model1(input_ids=input_ids_m1, attention_mask=attention_mask_m1, past_key_values=None, use_cache=use_cache, return_dict=True)
        next_tokens_scores0 = outputs0.logits[:,-1,:]
        next_tokens_scores1 = outputs1.logits[:,-1,:].to(model0.device)
        cur_key = torch.FloatTensor([key[cur_id] for cur_id in key_idx]).to(next_tokens_scores0).unsqueeze(1)
        next_tokens_scores = (1-cur_key)*next_tokens_scores0 + cur_key*next_tokens_scores1  # key=0 -> use model0, key=1 -> use model1

        if do_sample:
            # Step 1: top-50 logit warper
            indices_to_remove = (next_tokens_scores<torch.topk(next_tokens_scores, 50)[0][..., -1, None])
            next_tokens_scores = next_tokens_scores.masked_fill(indices_to_remove, -1000)
            # Step 2: run sample
            probs = torch.nn.functional.softmax(next_tokens_scores, dim=-1)
            if torch.isnan(probs).any():
                print (probs)
                print (next_tokens_scores)
                print (torch.isnan(probs).float().sum())
                print (torch.isnan(next_tokens_scores).float().sum())
            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
        else:
            # run greedy search
            next_tokens = torch.argmax(next_tokens_scores, dim=-1)
        log_prob = torch.nn.functional.log_softmax(next_tokens_scores,-1)
        next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)
        input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
        attention_mask = torch.cat([attention_mask, attention_mask.new_ones((attention_mask.shape[0],1))], dim=-1)
        unfinished_sequences = unfinished_sequences.mul(
            next_tokens.tile(eos_token_id_tensor.shape[0], 1).ne(eos_token_id_tensor.unsqueeze(1)).prod(dim=0)
        )

        # update current key if switch
        for i in range(input_ids.shape[0]):
            if last_ed[i] == -1 and next_tokens[i] == pad_token_id:
                last_ed[i] = input_ids.shape[1]-1
            if next_tokens[i].item() in split_tokens:
                split_info[i].append( (last_st[i], input_ids.shape[1], key[key_idx[i]]) )
                last_st[i] = input_ids.shape[1]
                key_idx[i] = (key_idx[i]+1)%len(key)
        if input_ids.shape[1] >= max_length:
            break
        if unfinished_sequences.sum() == 0:
            break
    # update the final split info
    for i in range(input_ids.shape[0]):
        if last_ed[i] == -1:
            last_ed[i] = input_ids.shape[1]
        if last_st[i] < last_ed[i]:
            split_info[i].append( (last_st[i], last_ed[i], key[key_idx[i]]) )

    return input_ids, split_info
