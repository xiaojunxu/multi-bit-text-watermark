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
from peft import get_peft_config, get_peft_model, LoraConfig, AdaLoraConfig, TaskType

from model_utils import get_optimizer_grouped_parameters
import utils

def main(args) -> None:
    device = torch.device("cuda:0")

    SAVE_PATH = "%s/ckpt/%s_%s_parapretrain_DM_%s_with%s"%(args.workdir, args.model, args.dataset, args.para_method, args.sim_method)
    print ("\033[94mSave path:%s\033[0m"%SAVE_PATH)

    tokenizer = utils.get_tokenizer(args.model)

    with open("%s/data/%s-paraphrase-%s.json"%(args.workdir, args.dataset, args.para_method)) as inf:
        para_dataset = json.load(inf)

    TOT_NUM = args.learn_steps * args.batch_size
    assert len(para_dataset) >= TOT_NUM, (len(para_dataset),TOT_NUM)

    new_dataset = []
    same_dataset = []
    KEY_LIST = [0,1]
    for i in range(TOT_NUM):
        text = para_dataset[i]['ori']
        para_text = para_dataset[i]['para']
        prompt = utils.gen_para_prompt(text)
        new_dataset.append({'text':prompt+para_text})
        same_dataset.append({'text':prompt+text})

    def preprocess(examples):
        return tokenizer(examples['text'], max_length=args.max_len, truncation=True)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    train_dataset_key = Dataset.from_list(new_dataset).map(preprocess, batched=True, remove_columns=['text'])
    dataloader_key = torch.utils.data.DataLoader(train_dataset_key, batch_size=args.batch_size, collate_fn=data_collator)
    train_dataset_same = Dataset.from_list(same_dataset).map(preprocess, batched=True, remove_columns=['text'])
    dataloader_same = torch.utils.data.DataLoader(train_dataset_same, batch_size=args.batch_size, collate_fn=data_collator)

    model0 = utils.get_model(args.model, model_class=AutoModelForCausalLM, bf16=False)
    model1 = utils.get_model(args.model, model_class=AutoModelForCausalLM, bf16=False)

    #assert args.use_lora
    if args.use_lora:
        peft_config = AdaLoraConfig(task_type=TaskType.CAUSAL_LM, inference_mode=False, r=128, lora_alpha=16, target_modules=["q_proj", "v_proj"])
        model0 = get_peft_model(model0, peft_config)
        model1 = get_peft_model(model1, peft_config)
        model0.print_trainable_parameters()
        model1.print_trainable_parameters()
    model0.to(device)
    model1.to(device)

    from itertools import chain
    optimizer = AdamW(chain(model0.parameters(),model1.parameters()), lr=args.lr, weight_decay=0)
    num_training_steps = args.learn_steps
    lr_scheduler = get_scheduler(
        name="cosine", optimizer=optimizer,
        num_warmup_steps=int(0.01*num_training_steps), num_training_steps=num_training_steps)
    model0.train()
    model1.train()

    for idx, (batch_key, batch_same) in enumerate(zip(dataloader_key, dataloader_same)):
        input_ids = batch_key['input_ids']
        attention_mask = batch_key['attention_mask']
        labels = batch_key['labels']
        labels[labels==-100] = tokenizer.pad_token_id  # Do not hope the collator to ignore pad tokens in SFT!
        labels[:,:128+21] = -100  # (approximately) ignore the prompt part of the paraphrase
        for sid in range(len(labels)):
            if not (labels[sid]!=-100).any():
                continue
            last_loc = max((labels[sid]!=-100).nonzero())
            if last_loc+1 < labels.shape[1]:
                labels[sid, last_loc+1] = tokenizer.pad_token_id
        if args.verbose:
            print ("==============")
            print (tokenizer.decode(input_ids[0]))
            print ("--------------")
            print (tokenizer.decode(input_ids[0][labels[0]!=-100]))
            print ("==============")
        outputs0 = model0(input_ids.to(device), attention_mask=attention_mask.to(device), labels=labels.to(device))
        outputs1 = model1(input_ids.to(device), attention_mask=attention_mask.to(device), labels=labels.to(device))
        loss_key = outputs0.loss + outputs1.loss.to(device)
        loss_key0, loss_key1 = outputs0.loss, outputs1.loss

        prob_q0 = torch.nn.functional.softmax(outputs0.logits, -1)
        prob_q1 = torch.nn.functional.softmax(outputs1.logits.to(device), -1)
        sim_att_mask = attention_mask.clone().to(device)
        sim_att_mask[:,:128+21] = 0 # mask out the prompts
        if args.sim_method == 'KL':
            loss_sim = (-(prob_q0.detach()*torch.log(prob_q1+1e-9) + prob_q1.detach()*torch.log((prob_q0+1e-9))).sum(2) * sim_att_mask).sum() / sim_att_mask.sum()
        else:
            raise NotImplementedError()

        input_ids_same = batch_same['input_ids']
        attention_mask_same = batch_same['attention_mask']
        labels_same = batch_same['labels']
        labels_same[labels_same==-100] = tokenizer.pad_token_id  # Do not hope the collator to ignore pad tokens in SFT!
        labels_same[:,:128+21] = -100  # (approximately) ignore the prompt part of the paraphrase
        for sid in range(len(labels_same)):
            if not (labels_same[sid]!=-100).any():
                continue
            last_loc = max((labels_same[sid]!=-100).nonzero())
            if last_loc+1 < labels_same.shape[1]:
                labels_same[sid, last_loc+1] = tokenizer.pad_token_id
        outputs0_same = model0(input_ids_same.to(device), attention_mask=attention_mask_same.to(device), labels=labels_same.to(device))
        outputs1_same = model1(input_ids_same.to(device), attention_mask=attention_mask_same.to(device), labels=labels_same.to(device))
        loss_same = outputs0_same.loss + outputs1_same.loss.to(device)
        loss_same0, loss_same1 = outputs0_same.loss, outputs1_same.loss
        
        if loss_sim.item() < args.threshold:
            loss = loss_key + 0.1*loss_same - args.sim_eps * loss_sim
        else:
            loss = loss_key + 0.1*loss_same

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        lr_scheduler.step()

        # Loss.
        if idx % 20 == 0:
            print('Batch %d/%d, loss: %f, loss_key: %f, loss_same: %f, loss_sim: %f' % (
                idx, args.learn_steps,
                loss.item(), loss_key.item(), loss_same, loss_sim.item()))
            print (loss_key0.item(), loss_key1.item(), loss_same0.item(), loss_same1.item())

    if args.use_lora:
        model0 = model0.merge_and_unload()
        model1 = model1.merge_and_unload()

    # Save model.
    model0.save_pretrained(SAVE_PATH+"_m0/", from_pt=True)
    model1.save_pretrained(SAVE_PATH+"_m1/", from_pt=True)
    print ("\033[94mSaved to %s\033[0m"%SAVE_PATH)
    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model', type=str, default='llama2-1.1b-chat')
    parser.add_argument('--para_method', type=str, default='morepegasus-lengthfilter')
    parser.add_argument('--dataset', type=str, default='c4')
    parser.add_argument('--max_len', type=int, default=256)
    parser.add_argument('--use_lora', action='store_true')
    parser.add_argument('--verbose', action='store_true')

    parser.add_argument('--learn_steps', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size of unlearning.')
    parser.add_argument('--lr', type=float, default=1e-6)
    parser.add_argument('--suffix', type=str, default='')
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--workdir', type=str, default='.')


    parser.add_argument('--sim_method', type=str, default='KL')
    parser.add_argument('--threshold', type=float, default=2.5, help='Threshold for the similarity score.')
    parser.add_argument('--sim_eps', type=float, default=1.0)


    args = parser.parse_args()
    print (args)

    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)

    main(args)
