import argparse
import numpy as np
import json
import os
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM, get_scheduler, DataCollatorForLanguageModeling, pipeline
from transformers.pipelines.pt_utils import KeyDataset
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from tqdm import tqdm

from datasets import load_dataset, Dataset
from torch.utils.tensorboard import SummaryWriter

import utils
import gen_utils
import model_utils
from eval_utils import evaluate_detection
from ds_utils import convert_linear_layer_to_lora

def main(args):
    device = torch.device("cuda:0")

    SAVE_PATH = '%s/ckpt/DM_%s_%s_%s_b%s_step%s'%(args.workdir, args.dataset, args.actor_model, args.reward_model, args.batch_size, args.train_steps)
    if args.save_suffix != '':
        SAVE_PATH = SAVE_PATH + '_' + args.save_suffix
    print ("Save path: %s"%SAVE_PATH)

    if not os.path.isdir('%s/ckpt'%args.workdir):
        os.mkdir('%s/ckpt'%args.workdir)
    if not os.path.isdir(SAVE_PATH):
        os.mkdir(SAVE_PATH)
    if args.with_tensorboard:
        writer = SummaryWriter(SAVE_PATH+"/logs")

    tokenizer = utils.get_tokenizer(args.actor_model)
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"
    tokenizer_right_pad = utils.get_tokenizer(args.actor_model)
    tokenizer_right_pad.padding_side = "right"
    tokenizer_right_pad.truncation_side = "right"

    raw_train_dataset, raw_test_dataset = utils.get_c4_dataset(args, tokenizer_right_pad, max_len=args.max_inp_len, tot_num=args.train_steps*args.batch_size)
    train_prompt_loader = utils.create_prompt_loader(raw_train_dataset, tokenizer, prompt_style='custom', batch_size=args.batch_size)

    base_model = utils.get_model(args.reward_model)
    critic_model = model_utils.RewardModel(base_model, tokenizer).to(device)
    critic_model.load_state_dict(torch.load('%s/ckpt/%s_%s_%s_RM%s%s/reward_model.ckpt'%(args.workdir, args.reward_model, args.actor_model, args.dataset, args.init_strategy, args.init_suffix)))
    optim_params = model_utils.get_optimizer_grouped_parameters(critic_model, weight_decay=0)
    optimizer_critic = AdamW(optim_params, lr=args.critic_lr, betas=(0.9, 0.95))
    scheduler_critic = get_scheduler(name='cosine', optimizer=optimizer_critic, num_warmup_steps=min(100,0.1*args.train_steps), num_training_steps=args.train_steps)

    base_model = utils.get_model(args.reward_model)
    reward_model = model_utils.RewardModel(base_model, tokenizer).to(device)
    reward_model.load_state_dict(torch.load('%s/ckpt/%s_%s_%s_RM%s%s/reward_model.ckpt'%(args.workdir, args.reward_model, args.actor_model, args.dataset, args.init_strategy, args.init_suffix)))
    optim_params = model_utils.get_optimizer_grouped_parameters(reward_model, weight_decay=0)
    optimizer_reward = AdamW(optim_params, lr=args.reward_lr, betas=(0.9, 0.95))
    scheduler_reward = get_scheduler(name='cosine', optimizer=optimizer_reward, num_warmup_steps=min(100,0.1*args.train_steps), num_training_steps=args.train_steps)
    critic_model.train()
    reward_model.eval()
    print ("critic & reward model loaded")

    actor_model0 = utils.get_model(args.actor_model, model_class=AutoModelForCausalLM, model_path='%s/ckpt/%s_%s_parapretrain_DM%s_m0'%(args.workdir, args.actor_model, args.dataset, args.init_suffix)).to(device)
    actor_model0 = convert_linear_layer_to_lora(actor_model0, part_module_name='model.layers.', lora_dim=128)
    optim_params0 = model_utils.get_optimizer_grouped_parameters(actor_model0, weight_decay=0, lora_lr=args.lora_lr)
    optimizer0 = AdamW(optim_params0, lr=args.lr, betas=(0.9,0.95))
    scheduler0 = get_scheduler(name='cosine', optimizer=optimizer0, num_warmup_steps=min(100,0.1*args.train_steps), num_training_steps=args.train_steps)
    actor_model1 = utils.get_model(args.actor_model, model_class=AutoModelForCausalLM, model_path='%s/ckpt/%s_%s_parapretrain_DM%s_m1'%(args.workdir, args.actor_model, args.dataset, args.init_suffix)).to(device)
    actor_model1 = convert_linear_layer_to_lora(actor_model1, part_module_name='model.layers.', lora_dim=128)
    optim_params1 = model_utils.get_optimizer_grouped_parameters(actor_model1, weight_decay=0, lora_lr=args.lora_lr)
    optimizer1 = AdamW(optim_params1, lr=args.lr, betas=(0.9,0.95))
    scheduler1 = get_scheduler(name='cosine', optimizer=optimizer1, num_warmup_steps=min(100,0.1*args.train_steps), num_training_steps=args.train_steps)
    print ("actor models loaded")

    ref_model0 = utils.get_model(args.actor_model, model_class=AutoModelForCausalLM, model_path='%s/ckpt/%s_%s_parapretrain_DM%s_m0'%(args.workdir, args.actor_model, args.dataset, args.init_suffix)).to(device)
    ref_model1 = utils.get_model(args.actor_model, model_class=AutoModelForCausalLM, model_path='%s/ckpt/%s_%s_parapretrain_DM%s_m1'%(args.workdir, args.actor_model, args.dataset, args.init_suffix)).to(device)
    print ("ref model loaded")
    ref_model0.eval()
    ref_model1.eval()
    sim_tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-mpnet-base-v2')
    sim_model = AutoModel.from_pretrained('sentence-transformers/all-mpnet-base-v2').to(device)

    LLM_do_train = [True] * args.train_steps
    RM_do_train = [False] * args.reward_fix_steps + [True] * (args.train_steps-args.reward_fix_steps)

    exp_mini_dataset = utils.MiniDataset(max_size=1, small_batch_size=args.batch_size)
    tot_step = 0
    for idx, batch in tqdm(enumerate(train_prompt_loader)):
        if args.train_steps >= 0 and tot_step >= args.train_steps:
            break
        torch.cuda.empty_cache()

        prompt_input_ids = batch['input_ids'].to(device)
        prompt_attention_mask = batch['attention_mask'].to(device)
        prompt_length = prompt_input_ids.shape[1]

        # Generate sequence
        actor_model0.eval()
        actor_model1.eval()
        max_min_length = prompt_length + args.max_ans_len
        with torch.no_grad():
            full_key = list(np.random.randint(low=0, high=2, size=(100,)))
            seq, split_info = gen_utils.DM_generate_with_key(actor_model0, actor_model1, tokenizer, key=full_key, input_ids=prompt_input_ids, attention_mask=prompt_attention_mask, max_length=max_min_length, pad_token_id=tokenizer.pad_token_id, do_sample=True)

        # Calculate sequence information
        pad_token_id = tokenizer.pad_token_id
        seq_attention_mask = seq.not_equal(pad_token_id).long()
        seq_attention_mask[:,:prompt_length] = prompt_attention_mask  # added: keep the mask of first BOS token
        actor_model0.train()
        actor_model1.train()

        # filter empty ans
        ans = seq[:, prompt_length:]
        used_ids = []
        valid_ans_len = (ans != tokenizer.pad_token_id).sum(dim=-1)
        for i in range(len(valid_ans_len)):
            if valid_ans_len[i] <= 1:
                print ("EMPTY ANSWER GENERATED!")
                print ("EMPTY ANSWER GENERATED!")
                print ("EMPTY ANSWER GENERATED!")
            else:
                used_ids.append(i)

        with torch.no_grad():
            output0 = actor_model0(seq, attention_mask=seq_attention_mask)
            output1 = actor_model1(seq, attention_mask=seq_attention_mask)
            output0_ref = ref_model0(seq, attention_mask=seq_attention_mask)
            output1_ref = ref_model1(seq, attention_mask=seq_attention_mask)
            if args.sim_eps > 0:
                all_ori_text, all_para_text = [], []
                for sid in used_ids:
                    raw_prompt_text = tokenizer.decode(prompt_input_ids[sid][prompt_attention_mask[sid]!=0])
                    prompt_prefix, prompt_suffix = utils.get_prompt_prefix_suffix(prompt_style='custom', with_bos=True)
                    assert raw_prompt_text.startswith(prompt_prefix) and raw_prompt_text.endswith(prompt_suffix), raw_prompt_text
                    cur_ori_text = raw_prompt_text[len(prompt_prefix):-len(prompt_suffix)]
                    all_ori_text.append(cur_ori_text)
                    cur_para_text = []
                    for st, ed, key in split_info[sid]:
                        cur_para_text.append(seq[sid][st:ed])
                    cur_para_text = tokenizer.decode(torch.cat(cur_para_text))
                    all_para_text.append(cur_para_text)
                sim_reward = utils.calc_text_sim(sim_model, sim_tokenizer, all_ori_text, all_para_text, device).to(device)
                sim_reward = sim_reward*2-1
                if tot_step % 10 == 0:
                    print ("SIM REWARD:", sim_reward)
                    for txt1, txt2, s in zip(all_ori_text, all_para_text, sim_reward):
                        print ("==============")
                        print ("text1:", txt1)
                        print ("text2:", txt2)
                        print ("score:", s)
            else:
                sim_reward = [0.0]*len(used_ids)
        seq_info = {
            'prompts': prompt_input_ids[used_ids].contiguous(),
            'prompt_attention_mask': prompt_attention_mask[used_ids].contiguous(),
            'logprobs0': utils.gather_log_probs(output0.logits[:,:-1,:],seq[:,1:])[used_ids].contiguous(),
            'logprobs1': utils.gather_log_probs(output1.logits.to(device)[:,:-1,:],seq[:,1:])[used_ids].contiguous(),
            'ref_logprobs0': utils.gather_log_probs(output0_ref.logits.to(device)[:,:-1,:],seq[:,1:])[used_ids].contiguous(),
            'ref_logprobs1': utils.gather_log_probs(output1_ref.logits.to(device)[:,:-1,:],seq[:,1:])[used_ids].contiguous(),
            'input_ids': seq[used_ids].contiguous(),
            'attention_mask': seq_attention_mask[used_ids].contiguous(),
            'split_info': [split_info[i] for i in used_ids],
            'sim_reward': sim_reward,
        }

        exp_dataset = exp_mini_dataset.add(seq_info)
        if exp_dataset is not None and len(exp_dataset) >= 1:
            # Calculate actor loss and critic loss
            assert len(exp_dataset) == 1
            exp_data = exp_dataset[0]

            actor_loss0, actor_loss1, critic_loss, actor_prob0, actor_prob1 = utils.train_ck_rlhf_DM_new(args, full_key, actor_model0, actor_model1, critic_model, reward_model, tokenizer, exp_data, max_snum=args.sent_batch_size, cliprange=args.ppo_clip_eps)
            if args.sim_eps > 0:
                torch.cuda.empty_cache()
                actor_loss_sim0, actor_loss_sim1, critic_loss_sim, _, _ = utils.train_rlhf_withsim_DM(actor_model0, actor_model1, critic_model, exp_data, actor_prob0=actor_prob0, actor_prob1=actor_prob1, cliprange=args.ppo_clip_eps)
            else:
                actor_loss_sim0, actor_loss_sim1, critic_loss_sim = 0.0, 0.0, 0.0
            torch.cuda.empty_cache()

            #if args.kl_eps != 0:
            if True:  # always calculate kl loss for logging
                good_input_ids = exp_data['input_ids'].to(device)
                good_mask = exp_data['attention_mask'].to(device)

                with torch.no_grad():
                    outputs0_ref = ref_model0(good_input_ids.to(device), attention_mask=good_mask.to(device))
                    outputs1_ref = ref_model1(good_input_ids.to(device), attention_mask=good_mask.to(device))
                prob_p0 = torch.nn.functional.softmax(outputs0_ref.logits, -1).to(device)
                prob_p1 = torch.nn.functional.softmax(outputs1_ref.logits, -1).to(device)
                prob_q0 = torch.nn.functional.softmax(actor_prob0, -1).to(device)
                prob_q1 = torch.nn.functional.softmax(actor_prob1, -1).to(device)
                kl_position_loss0 = -prob_p0 * torch.log(prob_q0+1e-6)
                kl_position_loss1 = -prob_p1 * torch.log(prob_q1+1e-6)
                position_weight0 = torch.zeros_like(kl_position_loss0)
                position_weight1 = torch.zeros_like(kl_position_loss1)
                # do position weight based on split info
                for sid in range(len(exp_data['split_info'])):
                    for st, ed, key in exp_data['split_info'][sid]:
                        assert prompt_length <= st
                        if key == 0:
                            position_weight0[sid][st-1:ed+1] = 1.0
                        else:
                            position_weight1[sid][st-1:ed+1] = 1.0
                position_weight0 = position_weight0 / (position_weight0.sum(dim=1,keepdim=True)+1e-8)
                position_weight1 = position_weight1 / (position_weight1.sum(dim=1,keepdim=True)+1e-8)
                kl_loss0 = (position_weight0*kl_position_loss0).sum()
                kl_loss1 = (position_weight1*kl_position_loss1).sum()
            else:
                kl_loss0 = torch.zeros_like(actor_loss0)
                kl_loss1 = torch.zeros_like(actor_loss1)
            all_loss0 = args.wtm_eps * actor_loss0 + args.kl_eps * kl_loss0 + args.sim_eps * actor_loss_sim0
            all_loss1 = args.wtm_eps * actor_loss1 + args.kl_eps * kl_loss1 + args.sim_eps * actor_loss_sim1
            critic_loss = args.wtm_eps * critic_loss + args.sim_eps * critic_loss_sim

            torch.cuda.empty_cache()
            optimizer0.zero_grad()
            optimizer1.zero_grad()
            optimizer_critic.zero_grad()
            all_loss0.backward()
            all_loss1.backward()
            critic_loss.backward()
            if LLM_do_train[tot_step]:
                optimizer0.step()
                optimizer1.step()
                optimizer_critic.step()
                scheduler0.step()
                scheduler1.step()
                scheduler_critic.step()

            if tot_step % 10 == 0:
                print ("Step %d, actor loss: %.4f/%.4f, critic loss: %.4f, benign loss: %.4f/%.4f"%(tot_step, actor_loss0.item(), actor_loss1.item(), critic_loss.item(), kl_loss0.item(), kl_loss1.item()))
            if tot_step % 10 == 0:
                print ("LLM OUTPUT:")
                print (tokenizer.decode(exp_data['input_ids'][0]))

            # Train reward model
            torch.cuda.empty_cache()
            reward_model.train()

            tok_lists, labs = [], []
            for sid in range(len(exp_data['input_ids'])):
                for st, ed, key in exp_data['split_info'][sid]:
                    if st == ed:
                        continue
                    tok_lists.append(exp_data['input_ids'][sid][st:ed])
                    labs.append(key)
            labs = torch.FloatTensor(labs).to(device)

            full_reward_score = []
            for rew_bid in range(0,min(args.max_reward_substep, (len(tok_lists)-1)//args.sent_batch_size+1)):
                rew_st = rew_bid * args.sent_batch_size
                rew_ed = min((rew_bid+1) * args.sent_batch_size, len(tok_lists))
                cur_labs = labs[rew_st:rew_ed]

                all_input_ids, all_attention_mask = utils.process_token_list(tokenizer, tok_lists[rew_st:rew_ed], device)
                reward_score = reward_model.forward_value(all_input_ids, all_attention_mask, prompt_length=1)['chosen_end_scores']
                full_reward_score.append(reward_score.detach())
                reward_loss = torch.nn.functional.binary_cross_entropy_with_logits(reward_score, cur_labs)
                if RM_do_train[tot_step]:
                    optimizer_reward.zero_grad()
                    reward_loss.backward()
                    optimizer_reward.step()
                    if rew_st == 0:
                        scheduler_reward.step()
            full_reward_score = torch.cat(full_reward_score)
            pos_scores = full_reward_score[labs==1]
            neg_scores = full_reward_score[labs==0]
            if tot_step % 10 == 0:
                print ("Reward loss: %.4f; pos_score %.4f; neg_score %.4f"%(reward_loss.item(), pos_scores.mean().item(), neg_scores.mean().item()))
            if tot_step % 10 == 0:
                print ("=========")
                print ("=========")
                print ("original:")
                for i in range(len(exp_data['input_ids'])):
                    print (tokenizer.decode(exp_data['input_ids'][i]))
                print ("pos:")
                for i in range(len(all_input_ids)):
                    if labs[i] == 1:
                        print ("<1>:", full_reward_score[i], tokenizer.decode(tok_lists[i]))
                print ("neg:")
                for i in range(len(all_input_ids)):
                    if labs[i] == 0:
                        print ("<0>:", full_reward_score[i], tokenizer.decode(tok_lists[i]))
                print ("=========")
                print ("=========")


            reward_model.eval()

            if args.with_tensorboard:
                writer.add_scalar('actor_loss0', actor_loss0.item(), global_step=tot_step)
                writer.add_scalar('actor_loss1', actor_loss1.item(), global_step=tot_step)
                writer.add_scalar('critic_loss', critic_loss.item(), global_step=tot_step)
                writer.add_scalar('kl_loss0', kl_loss0.item(), global_step=tot_step)
                writer.add_scalar('kl_loss1', kl_loss1.item(), global_step=tot_step)
                writer.add_scalar('reward_loss', reward_loss.item(), global_step=tot_step)
                writer.add_scalar('pos_score', pos_scores.mean().item(), global_step=tot_step) # TODO san check len
                writer.add_scalar('neg_score', neg_scores.mean().item(), global_step=tot_step)
                if args.sim_eps > 0:
                    writer.add_scalar('sim_reward', sim_reward.mean().item(), global_step=tot_step)

        tot_step += 1

    actor_model0.eval()
    actor_model0.save_pretrained(SAVE_PATH+'/model0', from_pt=True)
    actor_model1.eval()
    actor_model1.save_pretrained(SAVE_PATH+'/model1', from_pt=True)
    torch.save(reward_model.state_dict(), SAVE_PATH+'/reward_model.ckpt')
    print ("\033[94mSaved to %s\033[0m"%SAVE_PATH)

    print ("After RL:")
    reward_model.eval()
    # Process test dataset
    test_dataset = []
    for line in raw_test_dataset:
        text = line['text']
        prompt = utils.gen_para_prompt(text)
        cont_human = text
        test_dataset.append({'prompt':prompt, 'cont_human': cont_human})

    evaluate_detection(actor_model0, actor_model1, reward_model, tokenizer_right_pad, test_dataset, True, save_to=SAVE_PATH+"/watermarked.txt", max_length=args.max_ans_len, num_tests=args.num_tests)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataset', type=str, default='c4')
    parser.add_argument('--actor_model', type=str, default='llama2-1.1b-chat')
    parser.add_argument('--reward_model', type=str, default='llama2-1.1b')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--sent_batch_size', type=int, default=8)
    parser.add_argument('--max_inp_len', type=int, default=128)
    parser.add_argument('--max_ans_len', type=int, default=128)
    parser.add_argument('--train_steps', type=int, default=10000)
    parser.add_argument('--num_tests', type=int, default=1000)

    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--lora_lr', type=float, default=5e-4)
    parser.add_argument('--critic_lr', type=float, default=5e-6)
    parser.add_argument('--reward_fix_steps', type=int, default=1000)
    parser.add_argument('--max_reward_substep', type=int, default=999)
    parser.add_argument('--reward_lr', type=float, default=1e-5)
    parser.add_argument('--wtm_eps', type=float, default=1.0)
    parser.add_argument('--kl_eps', type=float, default=0.1)
    parser.add_argument('--ppo_clip_eps', type=float, default=0.2)
    parser.add_argument('--sim_eps', type=float, default=0.0)
    parser.add_argument('--init_strategy', type=str, default='initDMnew')
    parser.add_argument('--init_suffix', type=str, default='')

    parser.add_argument('--workdir', type=str, default='.')
    parser.add_argument('--save_suffix', type=str, default='')
    parser.add_argument('--with_tensorboard', action='store_true')


    args = parser.parse_args()
    args.dtype = 'bf16'
    print (args)

    main(args)
