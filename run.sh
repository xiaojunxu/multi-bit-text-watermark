python pretrain_DMparaphrase.py --use_lora --lr 5e-4
python pretrain_DMRM.py --suffix _morepegasus-lengthfilter_withKL --model_path_0 ./ckpt/llama2-1.1b-chat_c4_parapretrain_DM_morepegasus-lengthfilter_withKL_m0 --model_path_1 ./ckpt/llama2-1.1b-chat_c4_parapretrain_DM_morepegasus-lengthfilter_withKL_m1
python main.py --init_suffix _morepegasus-lengthfilter_withKL --wtm_eps 0.1 --kl_eps 0.02 --sim_eps 1.0 --with_tensorboard --save_suffix main
