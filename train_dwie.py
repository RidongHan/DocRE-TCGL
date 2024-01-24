import torch
import random
import os, time
import numpy as np
import ujson as json
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoModel, AutoTokenizer
from transformers.optimization import AdamW, get_linear_schedule_with_warmup
from config import get_args
from model import DocREModel
from prepro import read_docred, collate_fn, get_adjacent_matrix
from utils import set_seed, Logger
from evaluation import to_official, official_evaluate, get_output_labels, evaluate_long_tail, evaluate_multi_label


def train(args, model, train_features, dev_features, test_features, id2rel, logger):
    def finetune(args, features, optimizer, num_epoch, num_steps, id2rel, logger):
        best_score = -1
        best_score_ign = -1
        best_epoch = -1
        best_threshold = 0.0
        train_dataloader = DataLoader(features, batch_size=args.train_batch_size, shuffle=True, collate_fn=collate_fn, drop_last=True)
        train_iterator = range(int(num_epoch))
        total_steps = int(len(train_dataloader) * num_epoch // args.gradient_accumulation_steps)
        warmup_steps = int(total_steps * args.warmup_ratio)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
        logger.write("Total steps: {}\n".format(total_steps))
        logger.write("Warmup steps: {}\n".format(warmup_steps))
        for epoch in train_iterator:
            model.zero_grad()
            start_time = time.time()
            for step, batch in enumerate(train_dataloader):
                model.train()
                outputs = model(**batch)

                bce_loss = outputs["bce_loss"]
                loss = bce_loss
                TCG_align_loss = torch.tensor(0.0)
                TCL_loss = torch.tensor(0.0) 

                if args.use_type:
                    if args.TCG and args.TCL:
                        TCL_loss = outputs["tcl_loss"]
                        TCG_align_loss = outputs["align_loss"]
                        
                        loss = (1+args.alpha_TCG+args.alpha_TCL) * bce_loss * TCG_align_loss * TCL_loss / (TCG_align_loss * TCL_loss + args.alpha_TCG * bce_loss * TCL_loss + args.alpha_TCL * bce_loss * TCG_align_loss)  

                    if args.TCG and not args.TCL:
                        TCG_align_loss = outputs["align_loss"]
                        loss = (1+args.alpha_TCG) * bce_loss * TCG_align_loss / (args.alpha_TCG * bce_loss + TCG_align_loss)
                            
                    if not args.TCG and args.TCL:
                        TCL_loss = outputs["tcl_loss"]
                        loss = (1+args.alpha_TCL) * bce_loss * TCL_loss / (args.alpha_TCL * bce_loss + TCL_loss)

                loss = (loss) / args.gradient_accumulation_steps

                loss.backward()
                if step % args.gradient_accumulation_steps == 0:
                    if args.max_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    optimizer.step()
                    scheduler.step()
                    model.zero_grad()
                    num_steps += 1
                
                gap = 30 
                if step%gap == 0:
                    logger.write('{:3d} step/Epoch{:3d}, Total Loss {:8f}, DocRE_loss {:8f}, TCG_align_loss {:8f}, TCL_loss {:8f},\n'.format(step, epoch, loss.item(), bce_loss.item(), TCG_align_loss.item(), TCL_loss.item()))

                if (step + 1) == len(train_dataloader) or (args.evaluation_steps > 0 and num_steps % args.evaluation_steps == 0 and step % args.gradient_accumulation_steps == 0):
                    logger.write('| epoch {:3d} | time: {:5.2f}s \n'.format(epoch, time.time() - start_time))

                    eval_start_time = time.time()
                    dev_score_ign, best_thres, dev_output = evaluate(args, model, dev_features, id2rel, logger, tag="dev", g_threshold=0.5, eval_long_tail_macro=False, eval_multi_label_overlap=False, use_g_thres=False)
    
                    if dev_score_ign > best_score_ign:
                        best_score_ign = dev_score_ign
                        best_score = dev_output["dev_F1"]
                        best_epoch = epoch
                        best_threshold = best_thres
                        
                        if args.model_prefix != "":
                            save_path = os.path.join(args.save_path, args.model_prefix + "-" + str(args.seed)) + ".pt"
                            torch.save(model.state_dict(), save_path)
                            logger.write("best model saved!\n")
                        
                    logger.write('| epoch {:3d} | time: {:5.2f}s | best epoch:{:3d} Ign F1:{:5.3f}% F1:{:5.3f}% Threshold: {:5.3f}\n'.format(epoch, time.time() - eval_start_time, best_epoch, best_score_ign, best_score, best_threshold))
        logger.write('seed:{:3d} | best epoch:{:3d} Ign F1:{:5.3f}% F1:{:5.3f}%'.format(args.seed, best_epoch, best_score_ign,  best_score))
        logger.write(f' | {save_path.split("/")[-1]}\n')
        return num_steps

    re_layer = ["extractor", "bilinear", "TCL", ]
    graph_layer = ["TCG", "rel_embeddings"]
    if args.use_type:
        if args.TCG:
            graph_layer.append("ner_embeddings")
        else:
            re_layer.append("ner_embeddings")

    plms_parameters = []
    graph_parameters = []
    docre_parameters = []
    for n, p in model.named_parameters():
        if (not any(nd in n for nd in re_layer)) and (not any(nd in n for nd in graph_layer)):
            plms_parameters.append(p)
            # print(n)
        if any(nd in n for nd in graph_layer):
            graph_parameters.append(p)
            # print(n)
        if any(nd in n for nd in re_layer):
            docre_parameters.append(p)
            # print(n)
    optimizer_grouped_parameters = [
        {"params": plms_parameters,  "lr": args.learning_rate},  # 5e-5
        {"params": graph_parameters, "lr": args.gnn_lr},
        {"params": docre_parameters, "lr": 1e-4},
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon, weight_decay=1e-2)
    
    num_steps = 0
    set_seed(args)
    model.zero_grad()
    finetune(args, train_features, optimizer, args.num_train_epochs, num_steps, id2rel, logger)


def evaluate(args, model, features, id2rel, logger, tag="dev", g_threshold=0.5, eval_long_tail_macro=False, eval_multi_label_overlap=False, use_g_thres=False):
    dataloader = DataLoader(features, batch_size=args.test_batch_size, shuffle=False, collate_fn=collate_fn, drop_last=False)
    preds = []
    for batch in dataloader:
        model.eval()
        with torch.no_grad():
            batch.pop('labels')
            outputs = model(**batch)
            pred = outputs["sigmoid_logits"]
            preds.append(pred)
    preds = torch.cat(preds, dim=0)

    if tag == "test":
        cur_output = get_output_labels(preds, theta=g_threshold, num_labels=args.num_labels).cpu().numpy()
        cur_output[np.isnan(cur_output)] = 0
        cur_output = cur_output.astype(np.float32)

        ans = to_official(cur_output, features, id2rel)
        best_f1 = 0.0
        best_f1_ign = 0.0
        if len(ans) > 0:
            best_f1, _, best_f1_ign, _ = official_evaluate(ans, args.data_dir, eval=tag)
        output = {
            tag + "_F1": best_f1 * 100,
            tag + "_F1_ign": best_f1_ign * 100,
            tag + "_Threshold": g_threshold,
        }
        logger.write(json.dumps(output, indent=4) + "\n")
        logger.write("-"*50 + "\n")

        return best_f1, output

    elif tag == "dev":
        best_f1 = 0.0
        best_f1_ign = 0.0
        best_threshold = 0.0

        global_threshold = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        logger.write("-"*80)
        logger.write("\nEvaluation with different global thereshold ...\n")
        for threshold in global_threshold:
            cur_output = get_output_labels(preds, theta=threshold, num_labels=args.num_labels).cpu().numpy()
            cur_output[np.isnan(cur_output)] = 0
            cur_output = cur_output.astype(np.float32)

            ans = to_official(cur_output, features, id2rel)
            cur_f1 = 0.0
            cur_f1_ign = 0.0
            if len(ans) > 0:
                cur_f1, _, cur_f1_ign, _ = official_evaluate(ans, args.data_dir, eval=tag)
            cur_output = {
                tag + "_F1": cur_f1 * 100,
                tag + "_F1_Ign": cur_f1_ign * 100,
            }
            logger.write(f"[Threshold {threshold}] | ans: {len(ans)} | ")
            logger.write(json.dumps(cur_output) + "\n")
            
            if cur_f1_ign > best_f1_ign:
                best_f1 = cur_f1
                best_f1_ign = cur_f1_ign
                best_threshold = threshold
        
        output = {
            tag + "_F1": best_f1 * 100,
            tag + "_F1_Ign": best_f1_ign * 100,
            tag + "_Threshold": best_threshold,
        }
        logger.write("Results with the Best Threshold:\n")
        logger.write(json.dumps(output, indent=4) + "\n")
        logger.write("-"*50 + "\n")

        if eval_long_tail_macro:
            logger.write("Evaluation for Long-tailed Relations:\n")
            macro_output = evaluate_long_tail(args, preds, best_threshold, features, id2rel, tag=tag, logger=logger)
            logger.write("-"*50 + "\n")

        if eval_multi_label_overlap:
            logger.write("Evaluation for Multi-label Entity Pairs:\n")
            overlap_output = evaluate_multi_label(args, preds, best_threshold, features, id2rel, tag=tag, logger=logger)
            logger.write("-"*50 + "\n")

        return best_f1_ign * 100, best_threshold, output


def main():
    main_start_time = time.time()
    args = get_args()
    if args.random_seed:
        random_seed = random.randint(10, 200)
        args.seed = random_seed

    if args.load_path == "":
        log_file = "Train-" + args.data_dir.split("/")[-1] + "-" + args.model_prefix + "-" + str(args.seed) + ".log"
        logger = Logger(file_name=log_file, log=True)
    else:
        log_file = "Eval-" + args.data_dir.split("/")[-1] + "-" + args.load_path.split("/")[-1].split(".")[0] + ".log"
        logger = Logger(file_name=log_file, log=False)
        
    logger.write(json.dumps(args.__dict__, indent=4))
    logger.write("\n")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()
    args.device = device

    set_seed(args)

    config = AutoConfig.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path,
        num_labels=args.num_class,
    )
    # print(config)

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
    )

    rel2id = json.load(open(os.path.join(args.data_dir, 'rel2id.json'), 'r'))
    id2rel = {value: key for key, value in rel2id.items()}
    ner2id = json.load(open(os.path.join(args.data_dir, 'ner2id.json'), 'r'))
    id2ner = {value: key for key, value in ner2id.items()}
    
    # type_pair_2rel: Match(h_type, t_type, r)
    type_pair_2_rel_file = os.path.join(args.data_dir, args.type_pair_2_rel_file)
    type_pair_2_rels = json.load(open(type_pair_2_rel_file, "r"))
    for k in type_pair_2_rels:
        type_pair_2_rels[k] = sorted([rel2id[rel] for rel in type_pair_2_rels[k]])
        # print(k, type_pair_2_rels[k])

    # Type-constrained graphs
    adj_matrix = get_adjacent_matrix(args, rel2id, ner2id)

    train_file = os.path.join(args.data_dir, args.train_file)
    dev_file = os.path.join(args.data_dir, args.dev_file)
    test_file = os.path.join(args.data_dir, args.test_file)
    train_file_out = os.path.join(args.prepro_dir, args.transformer_type + "_" + args.train_file)
    dev_file_out = os.path.join(args.prepro_dir, args.transformer_type + "_" + args.dev_file)
    test_file_out = os.path.join(args.prepro_dir, args.transformer_type + "_" + args.test_file)

    read = read_docred
    train_features = read(args, train_file, train_file_out, tokenizer, rel2id, ner2id, type_pair_2_rel=type_pair_2_rels, max_seq_length=args.max_seq_length, logger=logger)
    dev_features = read(args, dev_file, dev_file_out, tokenizer, rel2id, ner2id, type_pair_2_rel=type_pair_2_rels, max_seq_length=args.max_seq_length, logger=logger)
    test_features = read(args, test_file, test_file_out, tokenizer, rel2id, ner2id, type_pair_2_rel=type_pair_2_rels, max_seq_length=args.max_seq_length, logger=logger)

    model = AutoModel.from_pretrained(
        args.model_name_or_path,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=config,
    )

    config.cls_token_id = tokenizer.cls_token_id
    config.sep_token_id = tokenizer.sep_token_id
    config.transformer_type = args.transformer_type
    config.max_seq_length = args.max_seq_length
    config.num_class = args.num_class
    config.device = args.device
    config.use_type = args.use_type
    config.graph_type = args.graph_type
    config.num_ner_type = len(ner2id)
    config.TCG = args.TCG
    config.TCL = args.TCL
    config.topk_tcl = args.topk_tcl
    config.max_num_labels = args.num_labels
    config.max_num_match_rels = args.max_num_match_rels
    
    model = DocREModel(config, model, adj_matrix, num_labels=args.num_labels)
    model.to(args.device)

    logger.write(str(model) + "\n")
    logger.write('total parameters:' + str(sum([np.prod(list(p.size())) for p in model.parameters() if p.requires_grad])) + "\n")

    if args.load_path == "":  # Training
        train(args, model, train_features, dev_features, test_features, id2rel, logger)
        logger.write("Total time: " + str((time.time()-main_start_time)/60.0) + " min\n")
    else:  # Testing
        args.load_path = os.path.join(args.save_path, args.load_path)
        check_point = args.load_path.split("/")[-1]
        logger.write(f"evaluation begins for checkpoint: {check_point}\n")
        start_time = time.time()
        model.load_state_dict(torch.load(args.load_path), strict=False)
        
        dev_score, best_thres, dev_output = evaluate(args, model, dev_features, id2rel, logger, tag="dev", eval_long_tail_macro=True, eval_multi_label_overlap=True)

        logger.write("Evaluation on test set with the best threshold: {}\n".format(best_thres))
        test_score, test_output = evaluate(args, model, test_features, id2rel, logger, tag="test", g_threshold=best_thres, eval_long_tail_macro=False, eval_multi_label_overlap=False)
        logger.write("Finished!!! | time: " + str(time.time()-start_time) + "s\n")


if __name__ == "__main__":
    main()
