import argparse
import numpy as np
import json
import os
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM, get_scheduler, DataCollatorForLanguageModeling, pipeline
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from tqdm import tqdm

import utils
import model_utils
import gen_utils

def main(args):
    device = torch.device("cuda:0")

    # Load texts to watermark
    test_dataset = []
    with open('./watermark_test_text.txt') as inf:
        cur_txt = ''
        for cur_line in inf:
            if cur_line.startswith('====='):
                test_dataset.append(cur_txt.strip())
                cur_txt = ''
            else:
                cur_txt = cur_txt + cur_line
    if cur_txt != '':
        test_dataset.append(cur_txt.strip())

    # Load encoder and decoder
    MODEL0_PATH = "xiaojunxu/WatermarkEncoder-Qwen2.5-7b-it-model0"
    MODEL1_PATH = "xiaojunxu/WatermarkEncoder-Qwen2.5-7b-it-model1"
    RM_PATH = "xiaojunxu/WatermarkDecoder-Qwen2.5-1.5b"
    RM_HEADER_PATH = "ckpt/WatermarkDecoder-v_head.pt"
    tokenizer = utils.get_tokenizer('qwen2.5-7b-it')
    tokenizer.padding_side = 'left'
    actor_model0 = utils.get_model('qwen2.5-7b-it', model_class=AutoModelForCausalLM, model_path=MODEL0_PATH).to(device).eval()
    actor_model1 = utils.get_model('qwen2.5-7b-it', model_class=AutoModelForCausalLM, model_path=MODEL1_PATH).to(device).eval()
    base_model = utils.get_model('qwen2.5-1.5b', model_path=RM_PATH)
    reward_model = model_utils.RewardModel(base_model, tokenizer).to(device)
    reward_model.v_head.load_state_dict(torch.load(RM_HEADER_PATH, map_location=device))
    reward_model.eval()

    # Load similarity comparison model
    sim_tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-mpnet-base-v2')
    sim_model = AutoModel.from_pretrained('sentence-transformers/all-mpnet-base-v2').to(device)

    # Inject watermark into texts
    KEY = [1,0,1,0,1,0,1,0]*20  # can manually set arbitrary key; only first K bits will be injected, where K is the number of sentences.
    with open(args.save_path, 'w') as outf, torch.no_grad():
        for idx, txt in enumerate(tqdm(test_dataset, desc="Text")):
            prompt = utils.gen_para_prompt(txt, prompt_style='qwen', tokenizer=tokenizer)
            toks = tokenizer(prompt, max_length=args.max_inp_len, return_tensors='pt')
            prompt_length = toks['input_ids'].shape[1]
            max_min_length = prompt_length+args.max_ans_len

            best_score, best_info = None, None
            outf.write("===========Text %d============\n"%idx)
            outf.write("===========Text %d============\n"%idx)
            outf.write("===========Text %d============\n"%idx)
            outf.write("Input Prompt: %s\n"%prompt)
            for repeat_i_st in tqdm(range(0,args.n_repeat, args.batch_size), desc="Text %d Repeat"%idx):
                input_ids = toks['input_ids'].to(device).repeat(args.batch_size,1)
                attention_mask = toks['attention_mask'].to(device).repeat(args.batch_size,1)
                seq, _ = gen_utils.DM_generate_with_key(actor_model0, actor_model1, tokenizer, key=KEY, input_ids=input_ids, attention_mask=attention_mask, max_length=max_min_length, pad_token_id=tokenizer.pad_token_id, do_sample=True)

                for bid in range(args.batch_size):
                    repeat_i = repeat_i_st + bid
                    para_txt = tokenizer.decode(seq[bid,prompt_length:], skip_special_tokens=True)
                    split_sentences = gen_utils.split_sentence(para_txt)
                    new_toks_list = [tokenizer(one_sent.strip(), return_tensors='pt')['input_ids'][0,reward_model.num_padding_at_beginning:] for one_sent in split_sentences]
                    if len(tokenizer.decode(new_toks_list[-1], skip_special_tokens=True).strip()) == 0:
                        new_toks_list = new_toks_list[:-1]
                    cur_input_ids, cur_attention_mask = utils.process_token_list(tokenizer, new_toks_list, reward_model.device)
                    pred = reward_model.forward_value(cur_input_ids, cur_attention_mask, prompt_length=1, return_value_only=False)["chosen_end_scores"]

                    sim_reward = (
                        utils.calc_text_sim(sim_model, sim_tokenizer, [txt], [para_txt], actor_model0.device)[0].item()
                        + utils.calc_rogue_lcs_score(tokenizer, [txt], [para_txt])
                        ) / 2
                    #raise NotImplementedError("change to rogue")

                    cur_keys = KEY[:len(cur_input_ids)]

                    cur_acc = ((pred>0).float().cpu().numpy() == np.array(cur_keys)).astype(float).mean()
                    #cur_len = len(cur_keys)
                    cur_avg_score = ((pred>0).float().cpu().numpy() * (np.array(cur_keys)*2-1)).mean()
                    para_len = len(split_sentences)
                    ori_len = len(gen_utils.split_sentence(txt))
                    len_penalty = ((para_len-ori_len)/max(para_len,ori_len))**2
                    #cur_score = cur_acc + 0.1*cur_len + 0.01*cur_avg_score
                    #cur_score = cur_acc + sim_reward + 0.1*cur_len + 0.01*cur_avg_score
                    cur_score = cur_acc + sim_reward + 0.01*cur_avg_score + 1.0*len_penalty
                    if best_score is None or best_score < cur_score:
                        best_score = cur_score
                        best_info = (para_txt, cur_keys, pred, new_toks_list)
            (para_txt, cur_keys, pred, new_toks_list) = best_info
            outf.write("-----------Best of %d----------\n"%args.n_repeat)
            print ("-----------Best of %d----------"%args.n_repeat)
            outf.write("Paraphrased Sentence: %s\n"%para_txt)
            print ("Paraphrased Sentence: %s"%para_txt)
            outf.write("Breakdown:\n")
            print ("Breakdown:")
            for one_key, one_score, one_tok in zip(cur_keys, pred, new_toks_list):
                outf.write("    Bit: %d; Prediction score: %s; Sentence: %s\n"%(one_key, one_score.item(), tokenizer.decode(one_tok)))
                print ("    Bit: %d; Prediction score: %s; Sentence: %s"%(one_key, one_score.item(), tokenizer.decode(one_tok)))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--save_path', type=str, default='./watermarked_output.txt')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--max_inp_len', type=int, default=512)
    parser.add_argument('--max_ans_len', type=int, default=512)
    parser.add_argument('--n_repeat', type=int, default=4)

    args = parser.parse_args()
    main(args)

