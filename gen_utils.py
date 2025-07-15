# Refer to: ~/anaconda3/lib/python3.11/site-packages/transformers/generation/utils.py
import numpy as np
import torch

def gen_split_tokens(tokenizer):
    raise NotImplementedError("deprecated; use is_split_point() and split_sentence()")

ABBRV_LIST = ["Dr", "Mr", "Mrs", "Ms", "Prof", "St", "Rev", "Jr", "Sr", "i.e", "e.g", "U.S","etc", "vs", "M.D", "Ph.D", "B.A", "A.M", "R.S.V.P", "A.D", "N.A.S.A", "F.B.I", "C.I.A", "No"]
EN_EOS_STR = ['. ', '! ', '? ', '." ', '!" ', '?" ', '.\' ', '!\' ', '?\' ']
CN_EOS_STR = ['。', '！', '？', '。\n', '！\n', '？\n', '。 ', '！ ', '？ ']
def is_en_split_point(text):
    text = text.replace('\n',' ').replace('\t',' ')
    is_eos = False
    for eos in EN_EOS_STR:
        if text.endswith(eos):
            is_eos = True
            break
    if is_eos:
        for abbrv in ABBRV_LIST:
            if text.endswith(' '+abbrv+'. '):
                is_eos = False
                break
    return is_eos
def is_cn_split_point(text):
    is_eos = False
    for eos in CN_EOS_STR:
        if text.endswith(eos):
            is_eos = True
            break
    return is_eos
def is_split_point(text):
    return is_en_split_point(text) or is_cn_split_point(text)
    text = text.replace('\n',' ').replace('\t',' ')
    is_eos = False
    for eos in EOS_STR:
        if text.endswith(eos):
            is_eos = True
            break
    if is_eos:
        for abbrv in ABBRV_LIST:
            if text.endswith(' '+abbrv+'. '):
                is_eos = False
                break
    return is_eos
def split_sentence(text):
    last_i = 0
    sentences = []
    for i in range(len(text)):
        if is_split_point(text[:i+1]) and len(text[last_i:i+1].strip()) != 0:
            sentences.append(text[last_i:i+1])
            last_i = i+1
    if last_i != len(text) and len(text[last_i:].strip()) != 0:
        sentences.append(text[last_i:])
    return sentences

def DM_generate_with_key(model0, model1, tokenizer, key, input_ids, attention_mask, max_length, pad_token_id, synced_gpus=False, do_sample=False, keep_pert=True, use_cache=False):
    assert keep_pert
    assert synced_gpus==False, "multi card not supported"

    unfinished_sequences = torch.ones(input_ids.shape[0], dtype=torch.long, device=model0.device)
    eos_token_id_tensor = torch.tensor([pad_token_id]).to(model0.device)
    if tokenizer.end_of_turn_token is not None:
        eot_token_id = tokenizer.end_of_turn_token
        eot_token_id_tensor = torch.tensor([eot_token_id]).to(model0.device)
    else:
        eot_token_id = None
    key_idx = [0 for _ in range(input_ids.shape[0])]  # which key the i-th sentence is on
    prompt_len = input_ids.shape[1]

    last_st = [prompt_len for _ in range(input_ids.shape[0])]
    last_ed = [-1 for _ in range(input_ids.shape[0])]
    split_info = [[] for _ in range(input_ids.shape[0])]
    while True:
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
            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
        else:
            # run greedy search
            next_tokens = torch.argmax(next_tokens_scores, dim=-1)
        next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)
        input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
        attention_mask = torch.cat([attention_mask, attention_mask.new_ones((attention_mask.shape[0],1))], dim=-1)
        unfinished_sequences = unfinished_sequences.mul(
            next_tokens.tile(eos_token_id_tensor.shape[0], 1).ne(eos_token_id_tensor.unsqueeze(1)).prod(dim=0)
        )
        if tokenizer.end_of_turn_token is not None:
            unfinished_sequences = unfinished_sequences.mul(
                next_tokens.tile(eot_token_id_tensor.shape[0], 1).ne(eot_token_id_tensor.unsqueeze(1)).prod(dim=0)
            )

        # Update key if switch, complicated version. We determine sentence split by ". ", so for a sentence "aaa.", the switch will be detected in the next step "aaa. bbb".
        # Logic: if current is split point, then switch; otherwise, if there exist split point in the middle of generated string, then switch and change the current generate token to be the corresponding one.

        for i in range(input_ids.shape[0]):
            if last_ed[i] == -1 and next_tokens[i] == pad_token_id:
                last_ed[i] = input_ids.shape[1]-1
            if eot_token_id is not None and next_tokens[i] == eot_token_id:
                last_ed[i] = input_ids.shape[1]-1
            cur_sent = tokenizer.decode(input_ids[i][last_st[i]:])
            if len(cur_sent.strip()) > 0 and is_split_point(cur_sent):
                # if current is split point, then switch
                split_info[i].append( (last_st[i], input_ids.shape[1], key[key_idx[i]]) )
                last_st[i] = input_ids.shape[1]
                key_idx[i] = (key_idx[i]+1)%len(key)
            elif len(cur_sent.strip()) > 0:
                cur_tok_st_idx = len(tokenizer.decode(input_ids[i][last_st[i]:-1]))
                for idx in range(cur_tok_st_idx, len(cur_sent)-1):
                    check_sent = cur_sent[:idx+1]
                    if is_cn_split_point(check_sent):
                        split_info[i].append( (last_st[i], input_ids.shape[1], key[key_idx[i]]) )
                        last_st[i] = input_ids.shape[1]
                        key_idx[i] = (key_idx[i]+1)%len(key)
                        break
                    elif is_en_split_point(check_sent):
                        # if split point in middle, switch to be the last idx, and change the current token
                        split_info[i].append( (last_st[i], input_ids.shape[1]-1, key[key_idx[i]]) )
                        last_st[i] = input_ids.shape[1]-1
                        key_idx[i] = (key_idx[i]+1)%len(key)

                        # Re-do the current token generation process
                        one_cur_key = key[key_idx[i]]
                        one_next_tokens_scores = (1-one_cur_key)*next_tokens_scores0[i] + one_cur_key*next_tokens_scores1[i]  # key=0 -> use model0, key=1 -> use model1
                        if do_sample:
                            # Step 1: top-50 logit warper
                            indices_to_remove = (one_next_tokens_scores<torch.topk(one_next_tokens_scores, 50)[0][..., -1, None])
                            one_next_tokens_scores = one_next_tokens_scores.masked_fill(indices_to_remove, -1000)
                            # Step 2: run sample
                            probs = torch.nn.functional.softmax(one_next_tokens_scores, dim=-1)
                            one_next_token = torch.multinomial(probs, num_samples=1).squeeze(0)
                        else:
                            # run greedy search
                            one_next_token = torch.argmax(next_tokens_scores, dim=-1)
                        one_next_token = one_next_token * unfinished_sequences[i] + pad_token_id * (1 - unfinished_sequences[i])
                        input_ids[i,-1] = one_next_token
                        break

        if input_ids.shape[1] >= max_length:
            break
        if unfinished_sequences.sum() == 0:
            break
    # update the final split info
    for i in range(input_ids.shape[0]):
        if last_ed[i] == -1:
            last_ed[i] = input_ids.shape[1]
        if last_st[i] < last_ed[i]:
            remain_sent = tokenizer.decode(input_ids[i][last_st[i]:last_ed[i]])
            if len(remain_sent.strip()) > 0:
                split_info[i].append( (last_st[i], last_ed[i], key[key_idx[i]]) )
    return input_ids, split_info
