import os
import numpy as np
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import random

from datasets import load_dataset, Dataset
from transformers import AutoConfig, AutoTokenizer, AutoModel, AutoModelForCausalLM, DataCollatorForLanguageModeling
import gen_utils

def get_model(model_name, model_class=AutoModel, model_path=None, dropout=0.0, new_vocab_size=None, bf16=True, **kwargs):
    if "llama" in model_name or "gemma" in model_name or "qwen" in model_name:
        if model_path is None:
            if model_name == "llama2-7b":
                model_path = "meta-llama/Llama-2-7b-hf"
            elif model_name == "llama2-7b-chat":
                model_path = "meta-llama/Llama-2-7b-chat-hf"
            elif model_name == "llama2-1.1b":
                model_path = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"
            elif model_name == "llama2-1.1b-chat":
                model_path = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
            elif model_name == "llama3-8b":
                model_path = "meta-llama/Llama-3.1-8B"
            elif model_name == "llama3-8b-chat":
                model_path = "meta-llama/Llama-3.1-8B-Instruct"
            elif model_name == 'gemma2-2b':
                model_path = "google/gemma-2-2b"
            elif model_name == 'gemma2-2b-it':
                model_path = "google/gemma-2-2b-it"
            elif model_name == 'gemma2-9b':
                model_path = "google/gemma-2-9b"
            elif model_name == 'gemma2-9b-it':
                model_path = "google/gemma-2-9b-it"
            elif model_name == 'qwen2.5-1.5b-it':
                model_path = "Qwen/Qwen2.5-1.5B-Instruct"
            elif model_name == 'qwen2.5-3b-it':
                model_path = "Qwen/Qwen2.5-3B-Instruct"
            elif model_name == 'qwen2.5-7b-it':
                model_path = "Qwen/Qwen2.5-7B-Instruct"
            elif model_name == 'qwen2.5-1.5b':
                model_path = "Qwen/Qwen2.5-1.5B"
            elif model_name == 'qwen2.5-3b':
                model_path = "Qwen/Qwen2.5-3B"
            elif model_name == 'qwen2.5-7b':
                model_path = "Qwen/Qwen2.5-7B"
            else:
                raise NotImplementedError()
        model_config = AutoConfig.from_pretrained(model_path)
        for key in ('dropout', 'attention_dropout', 'hidden_dropout', 'activation_dropout'):
            if hasattr(model_config, key):
                setattr(model_config, key, dropout)
        if bf16:
            setattr(model_config, 'torch_dtype', 'bfloat16')
            model = model_class.from_pretrained(model_path, config=model_config, torch_dtype=torch.bfloat16)
        else:
            model = model_class.from_pretrained(model_path, config=model_config)
        if new_vocab_size is not None:
            if getattr(model_config, 'vocab_size') < new_vocab_size:
                new_vocab_size = ((new_vocab_size-1)//64+1)*64
                model.resize_token_embeddings(new_vocab_size)
                setattr(model_config, 'vocab_size', new_vocab_size)
    else:
        raise NotImplementedError()
    return model

def get_tokenizer(model_name):
    if "llama" in model_name or "gemma" in model_name or "qwen" in model_name:
        #model_path = "meta-llama/Llama-2-7b-hf"   # unified tokenizer for llama #TODO: use if we do sft for models
        if model_name == "llama2-7b":
            model_path = "meta-llama/Llama-2-7b-hf"
        elif model_name == "llama2-7b-chat":
            model_path = "meta-llama/Llama-2-7b-chat-hf"
        elif model_name == "llama2-1.1b":
            model_path = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"
        elif model_name == "llama2-1.1b-chat":
            model_path = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        elif model_name == "llama3-8b":
            model_path = "meta-llama/Llama-3.1-8B"
        elif model_name == "llama3-8b-chat":
            model_path = "meta-llama/Llama-3.1-8B-Instruct"
        elif model_name == 'gemma2-2b':
            model_path = "google/gemma-2-2b"
        elif model_name == 'gemma2-2b-it':
            model_path = "google/gemma-2-2b-it"
        elif model_name == 'gemma2-9b':
            model_path = "google/gemma-2-9b"
        elif model_name == 'gemma2-9b-it':
            model_path = "google/gemma-2-9b-it"
        elif model_name == 'qwen2.5-1.5b-it':
            model_path = "Qwen/Qwen2.5-1.5B-Instruct"
        elif model_name == 'qwen2.5-3b-it':
            model_path = "Qwen/Qwen2.5-3B-Instruct"
        elif model_name == 'qwen2.5-7b-it':
            model_path = "Qwen/Qwen2.5-7B-Instruct"
        elif model_name == 'qwen2.5-1.5b':
            model_path = "Qwen/Qwen2.5-1.5B"
        elif model_name == 'qwen2.5-3b':
            model_path = "Qwen/Qwen2.5-3B"
        elif model_name == 'qwen2.5-7b':
            model_path = "Qwen/Qwen2.5-7B"
        else:
            raise NotImplementedError()
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = 'right'
        if 'gemma' in model_name:
            tokenizer.end_of_turn_token = 107
            assert tokenizer.decode([tokenizer.end_of_turn_token]) == '<end_of_turn>'
        else:
            tokenizer.end_of_turn_token = None
    else:
        raise NotImplementedError()
    return tokenizer

QWEN_PROMPT = 'Paraphrase the user provided text with lexical or structural changes, while keeping the exact semantic meaning of the original text. Keep in mind that you should keep the exact meaning of the original text. For example, you should not use synonyms that has slightly different meaning with the original meaning or change the proper nouns in the text. Make sure that the output text is still fluent. Use the same language as the original one. Do not include any other content after the substituted text.'
QWEN_SUFFIX = 'Here is the paraphrased version of the text which preserves the exact semantic meaning:'
def gen_para_prompt(text, prompt_style='custom', tokenizer=None):
    if prompt_style == "custom":
        return "Human: Paraphrase the text below.\n%s\nAssistant: Paraphrased text:"%text
    elif prompt_style == "qwen":
        message = [
            {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
            {"role":"user", "content":QWEN_PROMPT+"\n\n"+text}
        ]
        prompt = tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True) + f"\n{QWEN_SUFFIX}\n"
        return prompt
    else:
        raise NotImplementedError()

def get_prompt_prefix_suffix(prompt_style='custom', with_bos=False):
    if prompt_style == 'custom':
        if with_bos:
            prompt_prefix = "<s> Human: Paraphrase the text below.\n"
        else:
            prompt_prefix = "Human: Paraphrase the text below.\n"
        prompt_suffix = "\nAssistant: Paraphrased text:"
    else:
        raise NotImplementedError()
    return prompt_prefix, prompt_suffix

def get_c4_dataset(args, tokenizer, max_len=128, tot_num=40000):
    workdir = '.' if args is None else args.workdir
    if not os.path.isfile('%s/data/c4-len%d-train.json'%(workdir, max_len)):
        raw_train_dataset = load_dataset("c4", "realnewslike", split="train", streaming=True, trust_remote_code=True)
        train_dataset = []
        for i, line in enumerate(raw_train_dataset):
            toks = tokenizer(line['text'])['input_ids']
            if len(toks) < 1+max_len:
                continue
            text = tokenizer.decode(toks[1:1+max_len])
            train_dataset.append({'text':text})
            if len(train_dataset) % 1000 == 0:
                print (len(train_dataset))
            if len(train_dataset) >= tot_num:
                break
        with open('%s/data/c4-len%d-train.json'%(workdir, max_len), 'w') as outf:
            json.dump(train_dataset, outf)
    else:
        with open('%s/data/c4-len%d-train.json'%(workdir, max_len)) as inf:
            train_dataset = json.load(inf)
    assert len(train_dataset) >= tot_num
    train_dataset = train_dataset[:tot_num]

    if not os.path.isfile('%s/data/c4-len%d-test.json'%(workdir, max_len)):
        raw_test_dataset = load_dataset("c4", "realnewslike", split="validation", streaming=True, trust_remote_code=True)
        test_dataset = []
        for i, line in enumerate(raw_test_dataset):
            toks = tokenizer(line['text'])['input_ids']
            if len(toks) < 1+max_len:
                continue
            text = tokenizer.decode(toks[1:1+max_len])
            test_dataset.append({'text':text})
            if len(test_dataset) % 1000 == 0:
                print (len(test_dataset))
            if len(test_dataset) >= 10000:
                break
        with open('%s/data/c4-len%d-test.json'%(workdir, max_len), 'w') as outf:
            json.dump(test_dataset, outf)
    else:
        with open('%s/data/c4-len%d-test.json'%(workdir, max_len)) as inf:
            test_dataset = json.load(inf)
    return train_dataset, test_dataset


def gen_split_point(tokenizer, cur_toks):
    is_eos_list = []
    if 'Llama' in type(tokenizer).__name__:
        for eos_tok in [29889, 29973, 29991, 29901]:
            is_eos_list.append(cur_toks==eos_tok)
    else:
        raise NotImplementedError(type(tokenizer).__name__)
    split_point = torch.stack(is_eos_list, 0).any(dim=0).nonzero().cpu().squeeze(1) + 1
    if len(split_point)>0 and split_point[-1] == len(cur_toks):
        split_point = split_point[:-1]
    return split_point

def split_toks(cur_toks, prompt_length, split_point):
    ret_list = []
    new_split_point = split_point[split_point>prompt_length]
    st = prompt_length
    for sp in new_split_point:
        ret_list.append(cur_toks[st:sp])
        st = sp
    ret_list.append(cur_toks[st:])
    return ret_list

def process_token_list(tokenizer, toks_list, device):
    max_len = max([len(toks) for toks in toks_list]) if len(toks_list) > 0 else 0
    input_ids = torch.zeros(len(toks_list), max_len, dtype=torch.int32)+tokenizer.pad_token_id
    attention_mask = torch.zeros(len(toks_list), max_len, dtype=torch.int32)
    for i, toks in enumerate(toks_list):
        input_ids[i,:len(toks)] = toks
        indices = torch.logical_and(toks!=tokenizer.pad_token_id, toks!=0).nonzero()
        end_i = indices[-1] if len(indices)>0 else 0
        attention_mask[i,:end_i+1] = 1
    return input_ids.to(device), attention_mask.to(device)


def create_prompt_loader(dataset, tokenizer, prompt_style, batch_size):
    def preprocess(examples):
        prompts = [gen_para_prompt(text,prompt_style,tokenizer) for text in examples['text']]
        return tokenizer(prompts)
    if len(dataset) == 0:
        prompt_new_dataset = Dataset.from_list(dataset)
    else:
        prompt_new_dataset = Dataset.from_list(dataset).map(preprocess, batched=True, remove_columns=['text'])
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    prompt_dataloader = torch.utils.data.DataLoader(prompt_new_dataset, batch_size=batch_size, collate_fn=data_collator)
    return prompt_dataloader

class MiniDataset:
    def __init__(self, max_size, small_batch_size):
        self.dataset = []
        self.max_size = max_size
        self.small_batch_size = small_batch_size

    def seperate(self):
        small_dataset = []
        for large_batch in self.dataset:
            if type(large_batch) == list or type(large_batch) == tuple:
                large_size = len(large_batch[0])
            elif type(large_batch) == dict:
                large_size = len(large_batch[list(large_batch.keys())[0]])
            else:
                large_size = len(large_batch)
            for i in range(0, large_size, self.small_batch_size):
                if type(large_batch) == list or type(large_batch) == tuple:
                    small_dataset.append(
                        [x[i:i + self.small_batch_size] for x in large_batch])
                elif type(large_batch) == dict:
                    small_dataset.append({
                        k: v[i:i + self.small_batch_size]
                        for k, v in large_batch.items()
                    })
                else:
                    small_dataset.append(large_batch[i:i +
                                                     self.small_batch_size])
        self.free()

        return small_dataset

    def add(self, data):
        if len(self.dataset) < self.max_size:
            self.dataset.append(data)
            if len(self.dataset) == self.max_size:
                return self.seperate()
            else:
                return None
        else:
            raise ValueError(
                "The dataset is full but we did not stop it. There is a bug in the code."
            )

    def free(self):
        self.dataset = []

def gather_log_probs(logits, labels):
    log_probs = F.log_softmax(logits, dim=-1)
    log_probs_labels = log_probs.gather(dim=-1, index=labels.unsqueeze(-1))
    return log_probs_labels.squeeze(-1)

def actor_loss_fn(logprobs, old_logprobs, advantages, mask, cliprange=-1):
    if cliprange < 0:
        raise RuntimeError("Must assign a value for cliprange!")
    ## policy gradient loss
    log_ratio = (logprobs - old_logprobs) * mask
    ratio = torch.exp(log_ratio)
    pg_loss1 = -advantages * ratio
    pg_loss2 = -advantages * torch.clamp(ratio, 1.0 - cliprange,
                                         1.0 + cliprange)
    pg_loss = torch.sum(torch.max(pg_loss1, pg_loss2) * mask) / mask.sum()
    return pg_loss

def critic_loss_fn(values, old_values, returns, mask):
    cliprange_value = 0.2
    values_clipped = torch.clamp(
        values,
        old_values - cliprange_value,
        old_values + cliprange_value,
    )
    vf_loss1 = (values - returns)**2
    vf_loss2 = (values_clipped - returns)**2
    vf_loss = 0.5 * torch.sum(
        torch.max(vf_loss1, vf_loss2) * mask) / mask.sum()
    return vf_loss

def calc_text_sim(model, tokenizer, text_list1, text_list2, device):
    def mean_pooling(model_output, attention_mask):
        token_embeddings = model_output[0] #First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    assert len(text_list1) == len(text_list2)
    encoded_input = tokenizer(text_list1+text_list2, padding=True, truncation=True, return_tensors='pt').to(device)
    with torch.no_grad():
        model_output = model(**encoded_input)
    sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
    sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
    list1_emb = sentence_embeddings[:len(text_list1)]
    list2_emb = sentence_embeddings[len(text_list1):]
    sim = F.cosine_similarity(list1_emb, list2_emb, dim=1)
    return sim


def calc_log_perplexity(model, tokenizer, text_list, device):
    toks = tokenizer(text_list, padding=True, return_tensors='pt').to(device)
    input_ids = toks['input_ids']
    att_mask = toks['attention_mask']
    target_ids = input_ids.clone()
    target_ids[att_mask==0] = -100
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=att_mask, labels=target_ids)
    logppl = outputs.loss
    return logppl


def train_ck_rlhf_DM_new(args, full_key, actor_model0, actor_model1, critic_model, reward_model, tokenizer, exp_data, max_snum=4, reward_paraphraser=None, reward_substitute_ratio=0.0, reward_sent_paraphraser=None, tokenizer_right_pad=None, cliprange=0.2):
    rew_clip_val = 5.0
    gamma = 1.0
    lam = 0.95

    prompts = exp_data['prompts']
    prompt_mask = exp_data['prompt_attention_mask']
    log_probs0 = exp_data['logprobs0']
    log_probs1 = exp_data['logprobs1']
    ref_log_probs0 = exp_data['ref_logprobs0']
    ref_log_probs1 = exp_data['ref_logprobs1']
    seq = exp_data['input_ids']
    attention_mask = exp_data['attention_mask']
    action_mask = attention_mask[:,1:]
    split_info = exp_data['split_info']

    with torch.no_grad():
        # Steps for actor loss:
        # - calculate the reward for each sub-sentence (no grad)
        # - calculate the critic score for each sub-sentence (no grad)
        # - do GAE things with reward and critic score
        # - gen two copies of GAE based on the code (reverse and masking)
        # - two actor loss based on reversed and masksed GAE
        start = prompts.size()[-1] - 1
        saved_advantages0 = torch.zeros_like(log_probs0[:,start:])
        saved_advantages1 = torch.zeros_like(log_probs1[:,start:])
        saved_rewards = torch.zeros_like(log_probs0[:,start:]) # record for debug
        saved_info_for_critic_loss = []
        prompt_length = prompts.shape[1]

        # - calculate the reward and critic score for each sub-sentence (no grad)
        for sid in range(len(seq)):
            if len(split_info[sid]) == 0:
                print ("EMPTY RESPONSE")
                continue
            new_toks_list = []
            for st, ed, key in split_info[sid]:
                new_toks_list.append(seq[sid][st:ed])
            new_input_ids, new_attention_mask = process_token_list(tokenizer, new_toks_list, actor_model0.device)
            rew_ids, rew_att_mask = new_input_ids[:max_snum], new_attention_mask[:max_snum]
            if len(new_input_ids) > max_snum:
                # Too long, only calc first several sentences
                subsent_reward_score_calc = reward_model.forward_value(rew_ids.to(reward_model.device), rew_att_mask.to(reward_model.device), prompt_length=1)['chosen_end_scores'].detach()
                subsent_reward_score = torch.cat((subsent_reward_score_calc, torch.zeros(len(new_input_ids)-max_snum).to(subsent_reward_score_calc)),0)
                subsent_value_score_calc = critic_model.forward_value(new_input_ids[:max_snum].to(critic_model.device), new_attention_mask[:max_snum].to(critic_model.device), return_value_only=True).detach()
                subsent_value_score = torch.cat((subsent_value_score_calc, torch.zeros(len(new_input_ids)-max_snum, subsent_value_score_calc.shape[1]).to(subsent_value_score_calc)),0)
            else:
                subsent_reward_score = reward_model.forward_value(rew_ids.to(reward_model.device), rew_att_mask.to(reward_model.device), prompt_length=1)['chosen_end_scores'].detach()
                subsent_value_score = critic_model.forward_value(new_input_ids.to(critic_model.device), new_attention_mask.to(critic_model.device), return_value_only=True).detach().to(actor_model0.device)

            saved_info_for_critic_loss.append((new_input_ids, new_attention_mask, subsent_reward_score.clone(), subsent_value_score.clone()))
            for i in range(len(subsent_reward_score)):
                assert split_info[sid][i][2] == full_key[i%len(full_key)]

            # - do GAE things with reward and critic score
            subsent_old_rewards = torch.zeros_like(new_input_ids).float()
            subsent_old_values = subsent_value_score
            subsent_start = 0
            ends = [len(toks) for toks in new_toks_list]
            reward_clip = torch.clamp(subsent_reward_score, -rew_clip_val, rew_clip_val)
            for i, (st,ed,_) in zip(range(len(subsent_old_rewards)), split_info[sid]):
                assert ends[i] == ed-st
                if len(subsent_old_rewards[i,subsent_start:ends[i]]) > 0:
                    subsent_old_rewards[i,subsent_start:ends[i]][-1] += reward_clip[i]
                else:
                    print ("CHECK - EMPTY!")
                subsent_old_rewards[i,ends[i]:] = 0
                subsent_old_values[i,ends[i]:] = 0
            lastgaelam = 0
            advantages_reversed = []
            length = subsent_old_rewards.size()[-1]
            for t in reversed(range(subsent_start, length)):
                nextvalues = subsent_old_values[:,t] if t < length-1 else 0.0
                prevvalues = subsent_old_values[:,t-1] if t > 0 else 0.0
                delta = subsent_old_rewards[:,t] + gamma * nextvalues - prevvalues

                lastgaelam = delta + gamma * lam * lastgaelam
                advantages_reversed.append(lastgaelam)
            advantages = torch.stack(advantages_reversed[::-1], dim=1)
            subsent_returns = advantages + subsent_old_values[:, subsent_start:]
            subsent_advantages = advantages.detach()

            # - gen two copies of GAE based on the code (reverse and masking)
            flatten_advantages0 = torch.zeros_like(log_probs0[sid])
            flatten_advantages1 = torch.zeros_like(log_probs1[sid])
            flatten_reward = torch.zeros_like(log_probs0[sid]) # for debugging
            for i, (st, ed, key) in enumerate(split_info[sid]):
                if key == 0:
                    flatten_advantages0[st-1:ed-1] = subsent_advantages[i,:ed-st]
                else:
                    flatten_advantages1[st-1:ed-1] = subsent_advantages[i,:ed-st]
                flatten_reward[st-1:ed-1] = subsent_old_rewards[i,:ed-st]
            saved_advantages0[sid, :] = -flatten_advantages0[start:] # reverse the direction for 0
            saved_advantages1[sid, :] = flatten_advantages1[start:]
            saved_rewards[sid, :] = flatten_reward[start:]
    # - two actor loss based on reversed and masksed GAE
    new_action_mask = torch.zeros_like(action_mask)
    action_mask0 = torch.zeros_like(action_mask) + 1e-8
    action_mask1 = torch.zeros_like(action_mask) + 1e-8
    for sid in range(len(new_action_mask)):
        for st,ed,key in split_info[sid]:
            if key == 0:
                action_mask0[sid, st-1:ed-1] = 1
            else:
                action_mask1[sid, st-1:ed-1] = 1
            new_action_mask[sid, st-1:ed-1] = 1
    action_mask = new_action_mask
    actor_prob0 = actor_model0(input_ids=seq, attention_mask=attention_mask, use_cache=False).logits
    actor_prob1 = actor_model1(input_ids=seq, attention_mask=attention_mask, use_cache=False).logits
    actor_log_prob0 = gather_log_probs(actor_prob0[:,:-1,:], seq[:,1:])
    actor_log_prob1 = gather_log_probs(actor_prob1[:,:-1,:], seq[:,1:])
    actor_loss0 = actor_loss_fn(actor_log_prob0[:,start:], log_probs0[:,start:], saved_advantages0, action_mask0[:,start:], cliprange=cliprange)
    actor_loss1 = actor_loss_fn(actor_log_prob1[:,start:], log_probs1[:,start:], saved_advantages1, action_mask1[:,start:], cliprange=cliprange)


    # critic loss
    critic_loss = 0.0
    for sid in range(len(seq)):
        new_input_ids, new_attention_mask, subsent_reward_score, subsent_value_score = saved_info_for_critic_loss[sid]
        subsent_old_values = subsent_value_score
        subsent_old_rewards = torch.zeros_like(subsent_old_values)

        subsent_start = 0
        ends = [att.sum().item() for att in new_attention_mask]
        reward_clip = torch.clamp(subsent_reward_score, -rew_clip_val, rew_clip_val)
        for i, (st,ed,_) in zip(range(len(subsent_old_rewards)), split_info[sid]):
            if len(subsent_old_rewards[i,subsent_start:ends[i]]) > 0:
                subsent_old_rewards[i,subsent_start:ends[i]][-1] += reward_clip[i]
            else:
                print ("CHECK - EMPTY!")
            subsent_old_rewards[i,ends[i]:] = 0
            subsent_old_values[i,ends[i]:] = 0
        lastgaelam = 0
        advantages_reversed = []
        length = subsent_old_rewards.size()[-1]
        for t in reversed(range(subsent_start, length)):
            nextvalues = subsent_old_values[:,t] if t < length-1 else 0.0
            prevvalues = subsent_old_values[:,t-1] if t > 0 else 0.0
            delta = subsent_old_rewards[:,t] + gamma * nextvalues - prevvalues

            lastgaelam = delta + gamma * lam * lastgaelam
            advantages_reversed.append(lastgaelam)
        advantages = torch.stack(advantages_reversed[::-1], dim=1)
        subsent_returns = advantages + subsent_old_values[:, subsent_start:]

        if len(new_input_ids) > max_snum:
            new_input_ids = new_input_ids[:max_snum]
            new_attention_mask = new_attention_mask[:max_snum]
            subsent_returns = subsent_returns[:max_snum]

        critic_mask = torch.zeros_like(new_attention_mask)
        for i, (st,ed,_) in zip(range(len(critic_mask)), split_info[sid]):
            critic_mask[i,st-prompt_length:ed-prompt_length] = 1
        cur_value = critic_model.forward_value(new_input_ids, new_attention_mask, return_value_only=True)[:,:-1]
        cur_critic_loss = critic_loss_fn(cur_value,cur_value.detach(),subsent_returns[:,1:], critic_mask[:,1:])
        critic_loss = critic_loss + (cur_critic_loss/len(seq))
    return actor_loss0, actor_loss1, critic_loss, actor_prob0, actor_prob1


def train_rlhf_withsim_DM(actor_model0, actor_model1, critic_model, exp_data, kl_ctl=0.0, actor_prob0=None, actor_prob1=None, cliprange=0.2):
    assert kl_ctl==0.0
    rew_clip_val = 5.0
    gamma = 1.0
    lam = 0.95

    prompts = exp_data['prompts']
    log_probs0 = exp_data['logprobs0']
    log_probs1 = exp_data['logprobs1']
    with torch.no_grad():
        values = critic_model.forward_value(exp_data['input_ids'].to(critic_model.device), exp_data['attention_mask'].to(critic_model.device), return_value_only=True).detach()[:,:-1].to(actor_model0.device)
    reward_score = exp_data['sim_reward']
    seq = exp_data['input_ids']
    attention_mask = exp_data['attention_mask']
    split_info = exp_data['split_info']

    start = prompts.size()[-1] - 1
    action_mask = attention_mask[:,1:]
    ends = start + action_mask[:,start:].sum(1)+1
    old_values = values

    with torch.no_grad():
        old_rewards = torch.zeros_like(log_probs0)
        reward_clip = torch.clamp(reward_score, -rew_clip_val, rew_clip_val)
        for i in range(len(old_rewards)):
            old_rewards[i,start:ends[i]][-1] += reward_clip[i]
            old_rewards[i,ends[i]:] = 0
            old_values[i,ends[i]:] = 0

        # calc advantage and return
        lastgaelam = 0
        advantages_reversed = []
        length = old_rewards.size()[-1]
        for t in reversed(range(start, length)):
            nextvalues = old_values[:,t+1] if t < length-1 else 0.0
            delta = old_rewards[:,t] + gamma * nextvalues - old_values[:,t]
            lastgaelam = delta + gamma * lam * lastgaelam
            advantages_reversed.append(lastgaelam)
        advantages = torch.stack(advantages_reversed[::-1], dim=1)
        returns = advantages + old_values[:, start:]
        advantages = advantages.detach()

    new_action_mask = torch.zeros_like(action_mask)
    action_mask0 = torch.zeros_like(action_mask) + 1e-8
    action_mask1 = torch.zeros_like(action_mask) + 1e-8
    for sid in range(len(new_action_mask)):
        for st,ed,key in split_info[sid]:
            if key == 0:
                action_mask0[sid, st-1:ed-1] = 1
            else:
                action_mask1[sid, st-1:ed-1] = 1
            new_action_mask[sid, st-1:ed-1] = 1
    action_mask = new_action_mask
    if actor_prob0 is None:
        actor_prob0 = actor_model0(input_ids=seq, attention_mask=attention_mask, use_cache=False).logits
    if actor_prob1 is None:
        actor_prob1 = actor_model1(input_ids=seq.to(actor_model1.device), attention_mask=attention_mask.to(actor_model1.device), use_cache=False).logits
    actor_log_prob0 = gather_log_probs(actor_prob0[:,:-1,:], seq[:,1:])
    actor_log_prob1 = gather_log_probs(actor_prob1[:,:-1,:], seq[:,1:].to(actor_model1.device))
    actor_loss0 = actor_loss_fn(actor_log_prob0[:,start:], log_probs0[:,start:], advantages, action_mask0[:,start:], cliprange=cliprange)
    actor_loss1 = actor_loss_fn(actor_log_prob1[:,start:], log_probs1[:,start:].to(actor_model1.device), advantages.to(actor_model1.device), action_mask1[:,start:].to(actor_model1.device), cliprange=cliprange).to(actor_model0.device)
    value = critic_model.forward_value(input_ids=seq.to(critic_model.device), attention_mask=attention_mask.to(critic_model.device), return_value_only=True, use_cache=False)[:, :-1]
    critic_loss = critic_loss_fn(value[:,start:], old_values[:,start:].to(critic_model.device), returns.to(critic_model.device), action_mask[:,start:].to(critic_model.device)).to(actor_model0.device)

    return actor_loss0, actor_loss1, critic_loss, actor_prob0, actor_prob1

def get_longest_common_subsequence(words1, words2, i, j):
    # dp version
    lcs_dp = np.zeros((len(words1)+1, len(words2)+1))
    for i in range(len(words1)):
        for j in range(len(words2)):
            lcs_dp[i,j] = max(lcs_dp[i,j-1], lcs_dp[i-1,j]) # if i or j is 0, then ret will be zero
            if words1[i] == words2[j]:
                lcs_dp[i,j] = max(lcs_dp[i,j], lcs_dp[i-1,j-1]+1)
    max_lcs_len = lcs_dp.max().item()
    return max_lcs_len
def calc_rogue_lcs_score(tokenizer, target_texts, predicted_texts):
    all_f1 = []
    for predicted_text, target_text in zip(predicted_texts, target_texts):
        #predicted_words, target_words = predicted_text.split(), target_text.split()
        predicted_words = tokenizer(predicted_text)['input_ids'] # be default, the first tok is <bos>, so lcs must be >= 1
        target_words = tokenizer(target_text)['input_ids']
        num_predicted_words = len(predicted_words)
        num_target_words = len(target_words)
        len_lcs = get_longest_common_subsequence(predicted_words, target_words, 0, 0)
        if len_lcs == 0:
            f1 = 0
        else:
            precision = len_lcs/num_predicted_words
            recall = len_lcs/num_target_words
            f1 = 2*precision*recall/(precision + recall)
        all_f1.append(f1)
    return torch.FloatTensor(all_f1)
