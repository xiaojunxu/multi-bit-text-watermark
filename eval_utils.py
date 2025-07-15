import argparse
import numpy as np
import json
import os
import torch
from tqdm import tqdm

from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM, get_scheduler, DataCollatorForLanguageModeling, pipeline
from transformers.pipelines.pt_utils import KeyDataset
from datasets import load_dataset, Dataset
from sklearn.metrics import roc_auc_score
from torch.utils.tensorboard import SummaryWriter

import gen_utils
import utils

def calc_fpr_at_tpr(all_labs, all_scores, percentile):
    all_labs = np.array(all_labs)
    all_scores = np.array(all_scores)
    thresh = np.percentile(all_scores[all_labs==1], 100-percentile)
    neg_scores = all_scores[all_labs==0]
    return (neg_scores>thresh).mean()

def calc_tpr_at_fpr(all_labs, all_scores, percentile):
    all_labs = np.array(all_labs)
    all_scores = np.array(all_scores)
    thresh = np.percentile(all_scores[all_labs==0], 100-percentile)
    pos_scores = all_scores[all_labs==1]
    return (pos_scores>thresh).mean()

def ck_DM_generator_with_key(model0, model1, tokenizer, dataset, do_sample, device, batch_size, max_length=128):
    assert tokenizer.padding_side == "left"
    prompt_dataset = []
    for line in dataset:
        prompt_dataset.append({'text':line['prompt']})
    def preprocess(examples):
        return tokenizer(examples['text'], max_length=2048, truncation=True)
    prompt_new_dataset = Dataset.from_list(prompt_dataset).map(preprocess, batched=True, remove_columns=['text'])
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    prompt_dataloader = torch.utils.data.DataLoader(prompt_new_dataset, batch_size=batch_size, collate_fn=data_collator)
    for batch in prompt_dataloader:
        prompt_input_ids = batch['input_ids'].to(device)
        prompt_attention_mask = batch['attention_mask'].to(device)
        prompt_length = prompt_input_ids.shape[1]
        max_min_length = prompt_length + max_length

        full_key = [1,0,1,0,1,0,1,0]*20
        with torch.no_grad():
            seq, _ = gen_utils.DM_generate_with_key(model0, model1, tokenizer, key=full_key, input_ids=prompt_input_ids, attention_mask=prompt_attention_mask, max_length=max_min_length, pad_token_id=tokenizer.pad_token_id, do_sample=do_sample)
        for one_seq in seq:
            nonzero_ids = (one_seq!=tokenizer.pad_token_id).nonzero()
            st, ed = nonzero_ids[0], nonzero_ids[-1]+1
            yield tokenizer.decode(one_seq[st:ed]), {'input_ids':one_seq[st:ed].unsqueeze(0), 'attention_mask':torch.ones_like(one_seq[st:ed]).unsqueeze(0)}, full_key



def evaluate_detection(actor_model0, actor_model1, reward_model, tokenizer, test_dataset, do_sample, save_to=None, num_tests=100, max_length=160, bsize=4):
    original_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"

    ### Test detection AUC
    tot_loss = 0.0
    tot_cscore = 0.0
    tot_rscore = 0.0
    tot_num = 0.0
    all_scores = []
    all_labs = []
    saved_info = []
    llm_pos_scores = []
    llm_neg_scores = []
    human_pos_scores = []
    human_neg_scores = []

    if hasattr(actor_model0, 'module'):
        actor_model0 = actor_model0.module
        actor_model1 = actor_model1.module

    with torch.no_grad():
        for idx, (llm_gen, llm_inp, full_key) in enumerate(tqdm(ck_DM_generator_with_key(actor_model0, actor_model1, tokenizer, test_dataset[:num_tests], do_sample, actor_model0.device, batch_size=bsize, max_length=max_length))):
            full_key = np.array(full_key)
            line = test_dataset[idx]
            prompt_inp = tokenizer(line['prompt'], return_tensors='pt', max_length=2048, truncation=True)
            human_inp = tokenizer(line['prompt']+line['cont_human'], return_tensors='pt', max_length=2048)
            prompt_length = prompt_inp['input_ids'].shape[1]

            cur_toks = human_inp['input_ids'][0][prompt_length:]
            #split_point = utils.gen_split_point(tokenizer, cur_toks)
            #new_toks_list = utils.split_toks(cur_toks, 0, split_point)
            split_sentences = gen_utils.split_sentence(tokenizer.decode(cur_toks))
            new_toks_list = []
            prev_sent = ''
            for one_sent in split_sentences:
                cur_sent = one_sent
                new_toks_list.append(tokenizer(cur_sent, return_tensors='pt')['input_ids'][0,reward_model.num_padding_at_beginning:])
            cur_input_ids, cur_attention_mask = utils.process_token_list(tokenizer, new_toks_list, reward_model.device)
            pred = reward_model.forward_value(cur_input_ids, cur_attention_mask, prompt_length=1, return_value_only=False)["chosen_end_scores"]
            cur_keys = full_key[:len(cur_input_ids)]
            human_score = (pred[cur_keys==1].sum() - pred[cur_keys==0].sum()) / len(cur_keys)
            human_pos_scores.extend(list(pred[cur_keys==1].detach().float().cpu().numpy()))
            human_neg_scores.extend(list(pred[cur_keys==0].detach().float().cpu().numpy()))
            human_id, human_pred = cur_input_ids, pred   # record, for later print

            prompt_length = prompt_inp['input_ids'].shape[1]
            cur_toks = llm_inp['input_ids'][0][prompt_length:]
            # gen split point with alg
            #split_point = utils.gen_split_point(tokenizer, cur_toks)
            #new_toks_list = utils.split_toks(cur_toks, 0, split_point)
            split_sentences = gen_utils.split_sentence(tokenizer.decode(cur_toks))
            new_toks_list = []
            prev_sent = ''
            for one_sent in split_sentences:
                cur_sent = one_sent
                new_toks_list.append(tokenizer(cur_sent, return_tensors='pt')['input_ids'][0,reward_model.num_padding_at_beginning:])
            new_toks_list = [l for l in new_toks_list if len(l) > 0]
            if len(new_toks_list) == 0:
                print ("Empty in eval! SKIPPING")
                continue
            cur_input_ids, cur_attention_mask = utils.process_token_list(tokenizer, new_toks_list, reward_model.device)
            pred = reward_model.forward_value(cur_input_ids, cur_attention_mask, prompt_length=1, return_value_only=False)["chosen_end_scores"]
            cur_keys = full_key[:len(cur_input_ids)]
            llm_score = (pred[cur_keys==1].sum() - pred[cur_keys==0].sum()) / len(cur_keys)
            llm_pos_scores.extend(list(pred[cur_keys==1].detach().float().cpu().numpy()))
            llm_neg_scores.extend(list(pred[cur_keys==0].detach().float().cpu().numpy()))
            llm_id, llm_mask, llm_pred, llm_keys = cur_input_ids, cur_attention_mask, pred, cur_keys   # record, for later print

            if idx < 10:
                print ("=======Prompt %d======="%idx)
                print (line['prompt'])
                print ("Human:", human_score)
                print (tokenizer.decode(human_inp['input_ids'][0]))
                print ("LLM:", llm_score)
                print (pred)
                print (cur_keys)
                print (pred[cur_keys==1])
                print (pred[cur_keys==0])
                print (tokenizer.decode(llm_inp['input_ids'][0]))
                print ("=========")
                print ("pos:")
                for i in range(len(llm_id)):
                    if full_key[i] == 1:
                        print (tokenizer.decode(llm_id[i]))
                        print (llm_pred[i])
                print ("neg:")
                for i in range(len(llm_id)):
                    if full_key[i] == 0:
                        print (tokenizer.decode(llm_id[i]))
                        print (llm_pred[i])
                print ("=========")

            tot_loss += (human_score+llm_score).item()
            tot_cscore += llm_score.item()
            tot_rscore += human_score.item()
            tot_num += 1
            all_scores.append(llm_score.detach().float().cpu().numpy())
            all_labs.append(1)
            all_scores.append(human_score.detach().float().cpu().numpy())
            all_labs.append(0)
            saved_info.append((line['prompt'], human_score.item(), tokenizer.decode(human_inp['input_ids'][0][prompt_length:]), llm_score.item(), tokenizer.decode(llm_inp['input_ids'][0][prompt_length:]), llm_id, llm_mask, llm_pred, llm_keys))

    fpr_at_90_tpr = calc_fpr_at_tpr(all_labs, all_scores, 90)
    fpr_at_99_tpr = calc_fpr_at_tpr(all_labs, all_scores, 99)
    tpr_at_1_fpr = calc_tpr_at_fpr(all_labs, all_scores, 1)
    tpr_at_001_fpr = calc_tpr_at_fpr(all_labs, all_scores, 0.01)
    print ("Evaluation: loss %.6f, cscore %.6f, rscore %.6f, AUC %.6f, fpr@90tpr: %.6f, fpr@99tpr: %.6f, tpr@1fpr: %.6f, tpr@0.01fpr: %.6f"%(tot_loss/tot_num, tot_cscore/tot_num, tot_rscore/tot_num, roc_auc_score(all_labs, all_scores), fpr_at_90_tpr, fpr_at_99_tpr, tpr_at_1_fpr, tpr_at_001_fpr))

    acc_array = np.concatenate([np.array(llm_pos_scores)>0, np.array(llm_neg_scores)<0]).astype(float)
    llm_bit_acc = np.mean(acc_array)
    print ("LLM bit acc: %.6f"%llm_bit_acc)
    avg_bit_num = (len(llm_pos_scores)+len(llm_neg_scores)) / num_tests
    print ("Avg bit number: %.6f"%avg_bit_num)
    print ("Human - pos score: %.6f +- %.6f, neg score: %.6f +- %.6f, AUC: %.6f"%(np.mean(human_pos_scores), np.std(human_pos_scores), np.mean(human_neg_scores), np.std(human_neg_scores), roc_auc_score( [1]*len(human_pos_scores)+[0]*len(human_neg_scores), human_pos_scores+human_neg_scores  )))
    print ("LLM - pos score: %.6f +- %.6f, neg score: %.6f +- %.6f, AUC: %.6f"%(np.mean(llm_pos_scores), np.std(llm_pos_scores), np.mean(llm_neg_scores), np.std(llm_neg_scores), roc_auc_score( [1]*len(llm_pos_scores)+[0]*len(llm_neg_scores), llm_pos_scores+llm_neg_scores  )))

    # Similarity eval
    all_sim = []
    sim_tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-mpnet-base-v2')
    sim_model = AutoModel.from_pretrained('sentence-transformers/all-mpnet-base-v2').to(actor_model0.device)
    for line in saved_info:
        sim_reward = utils.calc_text_sim(sim_model, sim_tokenizer, [line[2]], [line[4]], actor_model0.device)
        all_sim.append(sim_reward)
    all_sim = torch.cat(all_sim)
    sim_val = all_sim.mean().item()
    print ("Similarity: %.6f"%sim_val)

    if save_to is not None:
        with open(save_to, "w", encoding='utf-8') as outf:
            for line, cur_sim in zip(saved_info, all_sim):
                outf.write("**Original text**: %s\n"%line[2])
                outf.write("**Original wtm score**: %s\n"%line[1])
                outf.write("**Paraphrased text**: %s\n"%line[4])
                outf.write("**Paraphrased wtm score**: %s\n"%line[3])
                outf.write("**Similarity**: %s\n"%cur_sim)
                outf.write("**Breakdown**:\n")
                for one_id, one_mask, one_pred, one_key in zip(line[5], line[6], line[7], line[8]):
                    outf.write("  Key: %s; wtm score: %s; text: %s\n"%(one_key, one_pred.item(), tokenizer.decode(one_id[one_mask!=0])))
                outf.write("\n\n")
            outf.write("Evaluation: loss %.6f, cscore %.6f, rscore %.6f, AUC %.6f, fpr@90tpr: %.6f, fpr@99tpr: %.6f, tpr@1fpr: %.6f, tpr@0.01fpr: %.6f, similarity: %.6f\n"%(tot_loss/tot_num, tot_cscore/tot_num, tot_rscore/tot_num, roc_auc_score(all_labs, all_scores), fpr_at_90_tpr, fpr_at_99_tpr, tpr_at_1_fpr, tpr_at_001_fpr, sim_val))
        assert save_to.endswith('watermarked.txt')
        json_path = save_to[:-15] + 'results.json'
        with open(json_path,'w') as outf:
            result_dict = {
                'auc':roc_auc_score(all_labs, all_scores),
                'fpr90':fpr_at_90_tpr,
                'fpr99':fpr_at_99_tpr,
                'tpr1':tpr_at_1_fpr,
                'tpr001':tpr_at_001_fpr,
                'sim':sim_val,
                'llm_bit_acc':llm_bit_acc,
                'avg_bit_num':avg_bit_num,
            }
            json.dump(result_dict, outf)

    tokenizer.padding_side = original_padding_side
    return roc_auc_score(all_labs, all_scores), fpr_at_90_tpr, fpr_at_99_tpr, tpr_at_1_fpr, tpr_at_001_fpr, sim_val


