python train_dwie.py \
--data_dir ./dataset/dwie \
--prepro_dir ./processed_data/dwie \
--transformer_type bert \
--model_name_or_path bert-base-cased \
--save_path ./checkpoint/dwie \
--train_batch_size 4 \
--test_batch_size 4 \
--gradient_accumulation_steps 1 \
--num_labels 3 \
--learning_rate 5e-5 \
--gnn_lr 1e-4 \
--warmup_ratio 0.06 \
--num_train_epochs 30.0 \
--seed 66 \
--num_class 66 \
--model_prefix bert-TCGL \
--graph_type gat \
--max_num_match_rels 14 \
--alpha_TCG 4.0 \
--alpha_TCL 2.0 \
--topk_tcl \
--use_type \
--TCG \
--TCL \
--load_path bert-TCGL.pt