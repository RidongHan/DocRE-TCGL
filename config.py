import argparse, os

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", default="./dataset/docred", type=str)
    parser.add_argument('--prepro_dir', type=str, default="./processed_data/docred")
    parser.add_argument("--transformer_type", default="bert", type=str)
    parser.add_argument("--model_name_or_path", default="bert-base-cased", type=str)

    parser.add_argument("--train_file", default="train_annotated.json", type=str)
    parser.add_argument("--dev_file", default="dev.json", type=str)
    parser.add_argument("--test_file", default="test.json", type=str)
    parser.add_argument('--type_pair_2_rel_file', default="type_pair_2_rel.json", type=str)
    parser.add_argument('--rel_2_type_file', default="rel_2_type.json", type=str)

    parser.add_argument("--save_path", default="./checkpoint/docred", type=str)
    parser.add_argument('--model_prefix', type=str, default="")
    parser.add_argument("--load_path", default="", type=str)

    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--max_seq_length", default=1024, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")

    parser.add_argument("--train_batch_size", default=4, type=int, help="Batch size for training.")
    parser.add_argument("--test_batch_size", default=8, type=int, help="Batch size for testing.")
    parser.add_argument("--gradient_accumulation_steps", default=1, type=int, 
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--num_labels", default=4, type=int, help="Max number of labels in prediction.")
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--gnn_lr", default=1e-4, type=float)
    
    parser.add_argument("--adam_epsilon", default=1e-6, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--warmup_ratio", default=0.06, type=float, help="Warm up ratio for Adam.")
    parser.add_argument("--num_train_epochs", default=30.0, type=float)
    parser.add_argument("--evaluation_steps", default=-1, type=int, help="Number of training steps between evaluations.")
    parser.add_argument("--seed", type=int, default=66, help="random seed for initialization")
    parser.add_argument('--random_seed', action='store_true', default=False)
    parser.add_argument("--num_class", type=int, default=97, help="Number of relation types in dataset.")

    parser.add_argument("--graph_type", type=str, default="gat", help="gcn or gat")
    parser.add_argument('--use_type', action='store_true', default=False, help="use type?")
    parser.add_argument('--TCG', action='store_true', default=False, help="use type-constrained graph (TCG)?")
    parser.add_argument('--TCL', action='store_true', default=False, help="use type-constrained loss (TCL)?")
    parser.add_argument('--topk_tcl', action='store_true', default=False, help="top-k")
    parser.add_argument("--max_num_match_rels", type=int, default=20, help="Maximum number of relations that match an Entity Type Pair")
    parser.add_argument("--alpha_balance", default=0.5, type=float)
    parser.add_argument("--alpha_TCG", default=4.0, type=float)
    parser.add_argument("--alpha_TCL", default=2.0, type=float)
    
    args = parser.parse_args()

    if args.use_type:
        if args.TCG and args.TCL:
            args.save_path = os.path.join(args.save_path, "TCG-L")
        elif args.TCG:
            args.save_path = os.path.join(args.save_path, "TCG")
        elif args.TCL:
            args.save_path = os.path.join(args.save_path, "TCL")

    if not os.path.exists(args.prepro_dir):
        os.makedirs(args.prepro_dir)
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    return args