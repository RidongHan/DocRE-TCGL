python train.py \
--data_dir ./dataset/docred \
--transformer_type roberta \
--model_name_or_path roberta-large \
--train_file train_annotated.json \
--dev_file dev.json \
--test_file test.json \
--train_batch_size 4 \
--test_batch_size 4 \
--gradient_accumulation_steps 1 \
--learning_rate 5e-5 \
--max_grad_norm 1.0 \
--warmup_ratio 0.06 \
--num_train_epochs 50.0 \
--seed 66 \
--num_class 97 \
--model_prefix roberta-TCGL \
--graph_type gat \
--gnn_lr 1e-4 \
--num_labels 4 \
--use_type \
--alpha_TCG 4.0 \
--alpha_TCL 1.0 \
--max_num_match_rels 20 \
--TCG \
--TCL \
--topk_tcl \
--load_path roberta-TCGL.pt