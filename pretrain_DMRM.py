import argparse
import numpy as np
import json
import os
import pickle
import transformers
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM, get_scheduler, DataCollatorForLanguageModeling, pipeline, default_data_collator
from transformers.pipelines.pt_utils import KeyDataset
import torch
from torch.optim import Adam, AdamW
from datasets import load_dataset, Dataset
from tqdm import tqdm

import utils
import model_utils
import gen_utils

def main(args) -> None:
    ### steps: 1) load raw dataset; 2) paraphrase the dataset; 3) add keys into the sentence; 4) sft on the added sentence

    device = torch.device("cuda:0")
    SAVE_PATH = "%s/ckpt/%s_%s_%s_RMinitDMnew"%(args.workdir, args.model, args.actor_model, args.dataset)
    SAVE_PATH = SAVE_PATH+args.suffix
    print ("\033[94mSave path:%s\033[0m"%SAVE_PATH)

    tokenizer = utils.get_tokenizer(args.actor_model)

    # Gen dataset
    TOT_NUM = args.learn_steps * args.batch_size

    model0 = utils.get_model(args.actor_model, model_class=AutoModelForCausalLM, model_path=args.model_path_0).to(device).eval()
    model1 = utils.get_model(args.actor_model, model_class=AutoModelForCausalLM, model_path=args.model_path_1).to(device).eval()
    max_length = args.max_inp_len + args.max_ans_len

    tokenizer_lp = utils.get_tokenizer(args.actor_model)
    tokenizer_lp.padding_side = "left"
    tokenizer_lp.truncation_side = "left"

    if args.dataset == 'c4':
        raw_train_dataset, raw_test_dataset = utils.get_c4_dataset(args, tokenizer, max_len=128)
        raw_train_dataset = raw_train_dataset[:TOT_NUM]
    else:
        raise NotImplementedError()
    train_prompt_loader = utils.create_prompt_loader(raw_train_dataset, tokenizer_lp, prompt_style='custom', batch_size=args.batch_size)
    dataset0, dataset1 = [], []
    for idx, batch in tqdm(enumerate(train_prompt_loader), total=len(train_prompt_loader)):
        prompt_input_ids = batch['input_ids'].to(device)
        prompt_attention_mask = batch['attention_mask'].to(device)
        prompt_length = prompt_input_ids.shape[1]
        with torch.no_grad():
            full_key = list(np.random.randint(low=0, high=2, size=(100,)))
            split_tokens = gen_utils.gen_split_tokens(tokenizer)
            seq, split_info = gen_utils.DM_generate_with_key(model0, model1, key=full_key, split_tokens=split_tokens, input_ids=prompt_input_ids, attention_mask=prompt_attention_mask, max_length=max_length, pad_token_id=tokenizer.pad_token_id, do_sample=True)
        for sid in range(len(split_info)):
            for st, ed, key in split_info[sid][:-1]:
                text = tokenizer.decode(seq[sid, st:ed], skip_special_tokens=True)
                if key == 0:
                    dataset0.append({'text':text})
                else:
                    dataset1.append({'text':text})
                if args.verbose:
                    print (tokenizer.decode(prompt_input_ids[sid]))
                    print ("----------")
    tot_len = min(len(dataset0), len(dataset1))
    def preprocess(examples):
        return tokenizer(examples['text'])
    dataset0 = Dataset.from_list(dataset0[:tot_len]).map(preprocess, batched=True, remove_columns=['text'])
    dataset1 = Dataset.from_list(dataset1[:tot_len]).map(preprocess, batched=True, remove_columns=['text'])
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    dataloader0 = torch.utils.data.DataLoader(dataset0, batch_size=args.batch_size, collate_fn=data_collator)
    dataloader1 = torch.utils.data.DataLoader(dataset1, batch_size=args.batch_size, collate_fn=data_collator)

    del model0
    del model1
    base_model = utils.get_model(args.model).to(device)
    reward_model = model_utils.RewardModel(base_model, tokenizer).to(device)
    optimizer = torch.optim.AdamW(reward_model.parameters(), lr=args.lr, betas=(0.9, 0.95))
    lr_scheduler = get_scheduler(name='cosine', optimizer=optimizer, num_warmup_steps=min(100,0.1*args.learn_steps), num_training_steps=args.learn_steps)
    reward_model.train()
    for idx, (batch0, batch1) in tqdm(enumerate(zip(dataloader0, dataloader1)), total=len(dataloader0)):
        input_ids0 = batch0['input_ids'].to(device)
        attention_mask0 = batch0['attention_mask'].to(device)
        input_ids1 = batch1['input_ids'].to(device)
        attention_mask1 = batch1['attention_mask'].to(device)
        reward_score0 = reward_model.forward_value(input_ids0, attention_mask0, prompt_length=1)['chosen_end_scores']
        reward_score1 = reward_model.forward_value(input_ids1, attention_mask1, prompt_length=1)['chosen_end_scores']
        loss0 = torch.nn.functional.binary_cross_entropy_with_logits(reward_score0, torch.zeros_like(reward_score0))
        loss1 = torch.nn.functional.binary_cross_entropy_with_logits(reward_score1, torch.ones_like(reward_score1))
        loss = loss0 + loss1
        loss.backward()
        lr_scheduler.step()
        optimizer.step()
        optimizer.zero_grad()
        if idx % 10 == 0:
            print('training, step %d/%d, loss: %f, pos score: %f, neg score: %f' % (
                idx+1, args.learn_steps, loss.item(), reward_score1.mean().item(), reward_score0.mean().item()))
        if args.verbose and idx % 100 == 0:
            for one_score, one_id, one_mask in zip(reward_score0, input_ids0, attention_mask0):
                print ("0:",one_score, tokenizer.decode(one_id[one_mask!=0]))
            for one_score, one_id, one_mask in zip(reward_score1, input_ids1, attention_mask1):
                print ("1:",one_score, tokenizer.decode(one_id[one_mask!=0]))

    # Save model.
    if not os.path.isdir(SAVE_PATH):
        os.mkdir(SAVE_PATH)
    torch.save(reward_model.state_dict(), SAVE_PATH+'/reward_model.ckpt')
    print ("\033[94mSaved to %s\033[0m"%SAVE_PATH)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--actor_model', type=str, default='llama2-1.1b-chat')
    parser.add_argument('--model', type=str, default='llama2-1.1b')
    parser.add_argument('--para_method', type=str, default='morepegasus-lengthfilter')
    parser.add_argument('--separate_gen', action='store_true')
    parser.add_argument('--dataset', type=str, default='c4')
    parser.add_argument('--max_inp_len', type=int, default=128)
    parser.add_argument('--max_ans_len', type=int, default=128)
    parser.add_argument('--verbose', action='store_true')

    parser.add_argument('--learn_steps', type=int, default=500)
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size of unlearning.')
    parser.add_argument('--lr', type=float, default=1e-5,
                        help='Unlearning LR.')
    parser.add_argument('--suffix', type=str, default='')

    parser.add_argument('--wtm_by_token', action='store_true')
    parser.add_argument('--wtm_every_K', type=int, default=20)
    parser.add_argument('--multi_gpu', action='store_true')

    parser.add_argument('--workdir', type=str, default='.')
    parser.add_argument('--with_test', action='store_true')
    parser.add_argument('--model_path_0',type=str, default=None)
    parser.add_argument('--model_path_1',type=str, default=None)


    args = parser.parse_args()
    print (args)
    if args.multi_gpu:
        assert torch.cuda.device_count() >= 2

    main(args)
