import os
import os.path
import json
import numpy as np
import torch


def to_official(preds, features, id2rel):
    h_idx, t_idx, title = [], [], []

    for f in features:
        hts = f["hts"]
        h_idx += [ht[0] for ht in hts]
        t_idx += [ht[1] for ht in hts]
        title += [f["title"] for ht in hts]

    res = []
    for i in range(preds.shape[0]):
        pred = preds[i]
        pred = np.nonzero(pred)[0].tolist()
        for p in pred:
            if p != 0:
                res.append(
                    {
                        'title': title[i],
                        'h_idx': h_idx[i],
                        't_idx': t_idx[i],
                        'r': id2rel[p],
                    }
                )
    return res


def gen_train_facts(data_file_name, truth_dir):
    fact_file_name = data_file_name[data_file_name.find("train_"):]
    fact_file_name = os.path.join(truth_dir, fact_file_name.replace(".json", ".fact"))

    if os.path.exists(fact_file_name):
        fact_in_train = set([])
        triples = json.load(open(fact_file_name))
        for x in triples:
            fact_in_train.add(tuple(x))
        return fact_in_train

    fact_in_train = set([])
    ori_data = json.load(open(data_file_name))
    for data in ori_data:
        vertexSet = data['vertexSet']
        for label in data['labels']:
            rel = label['r']
            for n1 in vertexSet[label['h']]:
                for n2 in vertexSet[label['t']]:
                    fact_in_train.add((n1['name'], n2['name'], rel))

    json.dump(list(fact_in_train), open(fact_file_name, "w"))

    return fact_in_train


def official_evaluate(tmp, path, eval="dev"):
    '''
        Adapted from the official evaluation code
    '''
    truth_dir = os.path.join(path, 'ref')

    if not os.path.exists(truth_dir):
        os.makedirs(truth_dir)

    fact_in_train_annotated = gen_train_facts(os.path.join(path, "train_annotated.json"), truth_dir)
    fact_in_train_distant = set([])
    if "docred" in path and "dwie" not in path:
        fact_in_train_distant = gen_train_facts(os.path.join(path, "train_distant.json"), truth_dir)

    if eval=="dev":
        truth = json.load(open(os.path.join(path, "dev.json")))
    elif eval=="test":
        truth = json.load(open(os.path.join(path, "test.json")))

    std = {}
    tot_evidences = 0
    titleset = set([])

    title2vectexSet = {}

    for x in truth:
        title = x['title']
        titleset.add(title)

        vertexSet = x['vertexSet']
        title2vectexSet[title] = vertexSet

        for label in x['labels']:
            r = label['r']
            h_idx = label['h']
            t_idx = label['t']
            std[(title, r, h_idx, t_idx)] = set(label['evidence'])
            tot_evidences += len(label['evidence'])

    tot_relations = len(std)
    tmp.sort(key=lambda x: (x['title'], x['h_idx'], x['t_idx'], x['r']))
    submission_answer = [tmp[0]]
    for i in range(1, len(tmp)):
        x = tmp[i]
        y = tmp[i - 1]
        if (x['title'], x['h_idx'], x['t_idx'], x['r']) != (y['title'], y['h_idx'], y['t_idx'], y['r']):
            submission_answer.append(tmp[i])

    correct_re = 0
    correct_evidence = 0
    pred_evi = 0

    correct_in_train_annotated = 0
    correct_in_train_distant = 0
    titleset2 = set([])
    for x in submission_answer:
        title = x['title']
        h_idx = x['h_idx']
        t_idx = x['t_idx']
        r = x['r']
        titleset2.add(title)
        if title not in title2vectexSet:
            continue
        vertexSet = title2vectexSet[title]

        if 'evidence' in x:
            evi = set(x['evidence'])
        else:
            evi = set([])
        pred_evi += len(evi)

        if (title, r, h_idx, t_idx) in std:
            correct_re += 1
            stdevi = std[(title, r, h_idx, t_idx)]
            correct_evidence += len(stdevi & evi)
            in_train_annotated = in_train_distant = False
            for n1 in vertexSet[h_idx]:
                for n2 in vertexSet[t_idx]:
                    if (n1['name'], n2['name'], r) in fact_in_train_annotated:
                        in_train_annotated = True
                    if (n1['name'], n2['name'], r) in fact_in_train_distant:
                        in_train_distant = True

            if in_train_annotated:
                correct_in_train_annotated += 1
            if in_train_distant:
                correct_in_train_distant += 1

    re_p = 1.0 * correct_re / len(submission_answer)
    re_r = 1.0 * correct_re / tot_relations
    if re_p + re_r == 0:
        re_f1 = 0
    else:
        re_f1 = 2.0 * re_p * re_r / (re_p + re_r)

    evi_p = 1.0 * correct_evidence / pred_evi if pred_evi > 0 else 0
    evi_r = 1.0 * correct_evidence / tot_evidences
    if evi_p + evi_r == 0:
        evi_f1 = 0
    else:
        evi_f1 = 2.0 * evi_p * evi_r / (evi_p + evi_r)

    re_p_ignore_train_annotated = 1.0 * (correct_re - correct_in_train_annotated) / (len(submission_answer) - correct_in_train_annotated + 1e-5)
    re_p_ignore_train = 1.0 * (correct_re - correct_in_train_distant) / (len(submission_answer) - correct_in_train_distant + 1e-5)

    if re_p_ignore_train_annotated + re_r == 0:
        re_f1_ignore_train_annotated = 0
    else:
        re_f1_ignore_train_annotated = 2.0 * re_p_ignore_train_annotated * re_r / (re_p_ignore_train_annotated + re_r)

    if re_p_ignore_train + re_r == 0:
        re_f1_ignore_train = 0
    else:
        re_f1_ignore_train = 2.0 * re_p_ignore_train * re_r / (re_p_ignore_train + re_r)

    return re_f1, evi_f1, re_f1_ignore_train_annotated, re_f1_ignore_train


def long_tail_evaluate_docred(tmp, path, eval="dev"):
    rel_dict = json.load(open(os.path.join(path, "rel_500_200_100.json"))) 
    rel_all = rel_dict['rall']  # 96
    rel_500 = rel_dict['r500'].values()  # 82
    rel_200 = rel_dict['r200'].values()  # 59
    rel_100 = rel_dict['r100'].values()  # 32

    # truth ground
    num_truth_list = [0] * (len(rel_all) + 1)
    if eval=="dev":
        truth = json.load(open(os.path.join(path, "dev.json")))
    elif eval=="test":
        truth = json.load(open(os.path.join(path, "test.json")))
    std = {}
    title2vectexSet = {}
    
    for x in truth:
        title = x['title']
        vertexSet = x['vertexSet']
        title2vectexSet[title] = vertexSet
        for label in x['labels']:
            r = label['r']
            r_id = rel_all[r]
            h_idx = label['h']
            t_idx = label['t']
            std[(title, r, h_idx, t_idx)] = set(label['evidence'])
            num_truth_list[r_id] += 1
    tot_relations = len(std)
    assert tot_relations==sum(num_truth_list)

    # ans
    tmp.sort(key=lambda x: (x['title'], x['h_idx'], x['t_idx'], x['r']))
    submission_answer = [tmp[0]]
    for i in range(1, len(tmp)):
        x = tmp[i]
        y = tmp[i - 1]  # 去重
        if (x['title'], x['h_idx'], x['t_idx'], x['r']) != (y['title'], y['h_idx'], y['t_idx'], y['r']):
            submission_answer.append(tmp[i])

    # 计数
    correct_re = 0
    num_correct_list = [0] * (len(rel_all) + 1)
    num_submission_list = [0] * (len(rel_all) + 1)
    
    for x in submission_answer:
        title = x['title']
        h_idx = x['h_idx']
        t_idx = x['t_idx']
        r = x['r']
        r_id = rel_all[r]
        num_submission_list[r_id] += 1
        if title not in title2vectexSet:
            continue
        vertexSet = title2vectexSet[title]

        if (title, r, h_idx, t_idx) in std:
            correct_re += 1
            num_correct_list[r_id] += 1
            
    # micro F1
    micro_p = 1.0 * correct_re / len(submission_answer)
    micro_r = 1.0 * correct_re / tot_relations
    if micro_p + micro_r == 0:
        micro_f1 = 0
    else:
        micro_f1 = 2.0 * micro_p * micro_r / (micro_p + micro_r)

    # macro F1
    assert sum(num_truth_list)==tot_relations and sum(num_correct_list)==correct_re and sum(num_submission_list)==len(submission_answer)
    macro_f1 = [0.0] * (len(rel_all) + 1)
    for i in range(1, (len(rel_all) + 1)):
        if num_submission_list[i] != 0:
            cur_p = 1.0 * num_correct_list[i] / num_submission_list[i]
        else:
            cur_p = 0.0
        if num_truth_list[i] != 0:
            cur_r = 1.0 * num_correct_list[i] / num_truth_list[i]
        else:
            cur_r = 0.0
        if cur_p + cur_r == 0:
            macro_f1[i] = 0.0
        else:
            macro_f1[i] = 2.0 * cur_p * cur_r / (cur_p + cur_r)

    ## macro_f1 for rall
    macro_f1_all = 1.0 * sum(macro_f1[1:]) / len(rel_all)

    ## macro_f1 for r500, r200, r100
    sum_500 = 0.0
    sum_200 = 0.0
    sum_100 = 0.0
    sum_greater_500 = 0.0
    num_rich =0
    for i in range(1, (len(rel_all) + 1)):
        if i in rel_100:
            sum_500 += macro_f1[i]
            sum_200 += macro_f1[i]
            sum_100 += macro_f1[i]
        elif i in rel_200:
            sum_500 += macro_f1[i]
            sum_200 += macro_f1[i]
        elif i in rel_500:
            sum_500 += macro_f1[i]
        else:
            sum_greater_500 += macro_f1[i]
            num_rich += 1
    
    macro_f1_500 = 1.0 * sum_500 / len(rel_500)
    macro_f1_200 = 1.0 * sum_200 / len(rel_200)
    macro_f1_100 = 1.0 * sum_100 / len(rel_100)

    macro_f1_greater_500 = 1.0 * sum_greater_500 / (len(rel_all)-len(rel_500))
    assert(num_rich==(len(rel_all)-len(rel_500)))
    # print("# >500 :", num_rich, len(rel_all)-len(rel_500))

    return micro_f1, macro_f1_all, macro_f1_500, macro_f1_200, macro_f1_100, macro_f1_greater_500


def long_tail_evaluate_dwie(tmp, path, eval="dev"):
    rel_dict = json.load(open(os.path.join(path, "rel_tailed.json"))) 
    rel_all = rel_dict['rall']
    rel_100 = rel_dict['r100'].values()  
    rel_50 = rel_dict['r50'].values()  
    rel_20 = rel_dict['r20'].values() 
    rel_10 = rel_dict['r10'].values()  

    # truth ground
    num_truth_list = [0] * (len(rel_all) + 1)
    if eval=="dev":
        truth = json.load(open(os.path.join(path, "dev.json")))
    elif eval=="test":
        truth = json.load(open(os.path.join(path, "test.json")))
    std = {}
    title2vectexSet = {}
    
    for x in truth:
        title = x['title']
        vertexSet = x['vertexSet']
        title2vectexSet[title] = vertexSet
        for label in x['labels']:
            r = label['r']
            r_id = rel_all[r]
            h_idx = label['h']
            t_idx = label['t']
            if (title, r, h_idx, t_idx) not in std:
                std[(title, r, h_idx, t_idx)] = set(label['evidence'])
                num_truth_list[r_id] += 1
    tot_relations = len(std)
    # print(tot_relations, sum(num_truth_list))
    assert tot_relations==sum(num_truth_list)

    # ans
    tmp.sort(key=lambda x: (x['title'], x['h_idx'], x['t_idx'], x['r']))
    submission_answer = [tmp[0]]
    for i in range(1, len(tmp)):
        x = tmp[i]
        y = tmp[i - 1]  # 去重
        if (x['title'], x['h_idx'], x['t_idx'], x['r']) != (y['title'], y['h_idx'], y['t_idx'], y['r']):
            submission_answer.append(tmp[i])

    # 计数
    correct_re = 0
    num_correct_list = [0] * (len(rel_all) + 1)
    num_submission_list = [0] * (len(rel_all) + 1)
    
    for x in submission_answer:
        title = x['title']
        h_idx = x['h_idx']
        t_idx = x['t_idx']
        r = x['r']
        r_id = rel_all[r]
        num_submission_list[r_id] += 1
        if title not in title2vectexSet:
            continue
        vertexSet = title2vectexSet[title]

        if (title, r, h_idx, t_idx) in std:
            correct_re += 1
            num_correct_list[r_id] += 1
            
    # micro F1
    micro_p = 1.0 * correct_re / len(submission_answer)
    micro_r = 1.0 * correct_re / tot_relations
    if micro_p + micro_r == 0:
        micro_f1 = 0
    else:
        micro_f1 = 2.0 * micro_p * micro_r / (micro_p + micro_r)

    # macro F1
    assert sum(num_truth_list)==tot_relations and sum(num_correct_list)==correct_re and sum(num_submission_list)==len(submission_answer)
    macro_f1 = [0.0] * (len(rel_all) + 1)
    for i in range(1, (len(rel_all) + 1)):
        if num_submission_list[i] != 0:
            cur_p = 1.0 * num_correct_list[i] / num_submission_list[i]
        else:
            cur_p = 0.0
        if num_truth_list[i] != 0:
            cur_r = 1.0 * num_correct_list[i] / num_truth_list[i]
        else:
            cur_r = 0.0
        if cur_p + cur_r == 0:
            macro_f1[i] = 0.0
        else:
            macro_f1[i] = 2.0 * cur_p * cur_r / (cur_p + cur_r)

    ## macro_f1 for rall
    macro_f1_all = 1.0 * sum(macro_f1[1:]) / len(rel_all)

    ## macro_f1 for r500, r200, r100
    sum_100 = 0.0
    sum_50 = 0.0
    sum_20 = 0.0
    sum_10 = 0.0
    sum_greater_100 = 0.0
    num_rich = 0
    for i in range(1, (len(rel_all) + 1)):
        if i in rel_10:
            sum_100 += macro_f1[i]
            sum_50 += macro_f1[i]
            sum_20 += macro_f1[i]
            sum_10 += macro_f1[i]
        elif i in rel_20:
            sum_100 += macro_f1[i]
            sum_50 += macro_f1[i]
            sum_20 += macro_f1[i]
        elif i in rel_50:
            sum_100 += macro_f1[i]
            sum_50 += macro_f1[i]
        elif i in rel_100:
            sum_100 += macro_f1[i]
        else:
            sum_greater_100 += macro_f1[i]
            num_rich += 1


    macro_f1_greater_100 = 1.0 * sum_greater_100 / (len(rel_all)-len(rel_100))
    assert(num_rich==(len(rel_all)-len(rel_100)))
    # print("# >100 :", num_rich, len(rel_all)-len(rel_100))
    
    macro_f1_100 = 1.0 * sum_100 / len(rel_100)
    macro_f1_50 = 1.0 * sum_50 / len(rel_50)
    macro_f1_20 = 1.0 * sum_20 / len(rel_20)
    macro_f1_10 = 1.0 * sum_10 / len(rel_10)

    return micro_f1, macro_f1_all, macro_f1_100, macro_f1_50, macro_f1_20, macro_f1_10, macro_f1_greater_100


def multi_label_evaluate(tmp, path, eval="dev"):
    if eval=="dev":
        overlap_dict = json.load(open(os.path.join(path, "overlap_triplet_dev.json"))) 
    overlap_list = overlap_dict["overlap_list"]
    overlap_entity_pair_list = overlap_dict["overlap_entity_pair_list"]
    overlap_tot = len(overlap_list)
    overlap_list_2 = overlap_dict["overlap_list_2"]
    overlap_entity_pair_list_2 = overlap_dict["overlap_entity_pair_list_2"]
    overlap_tot_2 = len(overlap_list_2)
    overlap_list_3 = overlap_dict["overlap_list_3"]
    overlap_entity_pair_list_3 = overlap_dict["overlap_entity_pair_list_3"]
    overlap_tot_3 = len(overlap_list_3)
    overlap_list_4 = overlap_dict["overlap_list_4"]
    overlap_entity_pair_list_4 = overlap_dict["overlap_entity_pair_list_4"]
    overlap_tot_4 = len(overlap_list_4)

    # truth ground
    if eval=="dev":
        truth = json.load(open(os.path.join(path, "dev.json")))
    # elif eval=="test":
    #     truth = json.load(open(os.path.join(path, "test.json")))

    std = {}
    title2vectexSet = {}
    for x in truth:
        title = x['title']
        vertexSet = x['vertexSet']
        title2vectexSet[title] = vertexSet
        for label in x['labels']:
            r = label['r']
            h_idx = label['h']
            t_idx = label['t']
            std[(title, r, h_idx, t_idx)] = set(label['evidence'])
    tot_relations = len(std)

    # ans
    tmp.sort(key=lambda x: (x['title'], x['h_idx'], x['t_idx'], x['r']))
    submission_answer = [tmp[0]]
    for i in range(1, len(tmp)):
        x = tmp[i]
        y = tmp[i - 1]  # 去重
        if (x['title'], x['h_idx'], x['t_idx'], x['r']) != (y['title'], y['h_idx'], y['t_idx'], y['r']):
            submission_answer.append(tmp[i])

    # 计数
    correct_re = 0
    correct_overlap = 0
    correct_overlap_2 = 0
    correct_overlap_3 = 0
    correct_overlap_4 = 0
    correct_overlap_3and4 = 0

    overlap_submission = 0
    overlap_submission_2 = 0
    overlap_submission_3 = 0
    overlap_submission_4 = 0
    overlap_submission_3and4 = 0
    
    for x in submission_answer:
        title = x['title']
        h_idx = x['h_idx']
        t_idx = x['t_idx']
        r = x['r']
        if title not in title2vectexSet:
            continue
        vertexSet = title2vectexSet[title]

        if (title, r, h_idx, t_idx) in std:
            correct_re += 1
        
        if [title, r, h_idx, t_idx] in overlap_list:
            correct_overlap += 1
        if [title, h_idx, t_idx] in overlap_entity_pair_list:
            overlap_submission += 1

        if [title, r, h_idx, t_idx] in overlap_list_2:
            correct_overlap_2 += 1
        if [title, h_idx, t_idx] in overlap_entity_pair_list_2:
            overlap_submission_2 += 1
        
        if [title, r, h_idx, t_idx] in overlap_list_3:
            correct_overlap_3 += 1
            correct_overlap_3and4 += 1
        if [title, h_idx, t_idx] in overlap_entity_pair_list_3:
            overlap_submission_3 += 1
            overlap_submission_3and4 += 1

        if [title, r, h_idx, t_idx] in overlap_list_4:
            correct_overlap_4 += 1
            correct_overlap_3and4 += 1
        if [title, h_idx, t_idx] in overlap_entity_pair_list_4:
            overlap_submission_4 += 1
            overlap_submission_3and4 += 1
       
    # micro F1
    micro_p = 1.0 * correct_re / len(submission_answer)
    micro_r = 1.0 * correct_re / tot_relations
    if micro_p + micro_r == 0:
        micro_f1 = 0
    else:
        micro_f1 = 2.0 * micro_p * micro_r / (micro_p + micro_r)

    # micro F1 for overlap
    if overlap_submission == 0:
        micro_p_overlap = 0.0
        micro_r_overlap = 0.0
    else:
        micro_p_overlap = 1.0 * correct_overlap / overlap_submission
        micro_r_overlap = 1.0 * correct_overlap / overlap_tot
    if micro_p_overlap + micro_r_overlap == 0:
        micro_f1_overlap = 0
    else:
        micro_f1_overlap = 2.0 * micro_p_overlap * micro_r_overlap / (micro_p_overlap + micro_r_overlap)

    # micro F1 for no_overlap
    micro_p_no_overlap = 1.0 * (correct_re-correct_overlap) / (len(submission_answer)-overlap_submission)
    micro_r_no_overlap = 1.0 * (correct_re-correct_overlap) / (tot_relations - overlap_tot)
    if micro_p_no_overlap + micro_r_no_overlap == 0:
        micro_f1_no_overlap = 0
    else:
        micro_f1_no_overlap = 2.0 * micro_p_no_overlap * micro_r_no_overlap / (micro_p_no_overlap + micro_r_no_overlap)

    # micro F1 for overlap 2, 3, 4
    if overlap_submission_2 == 0:
        micro_p_overlap_2 = 0.0
        micro_r_overlap_2 = 0.0
    else:
        micro_p_overlap_2 = 1.0 * correct_overlap_2 / overlap_submission_2
        micro_r_overlap_2 = 1.0 * correct_overlap_2 / overlap_tot_2
    if micro_p_overlap_2 + micro_r_overlap_2 == 0:
        micro_f1_overlap_2 = 0
    else:
        micro_f1_overlap_2 = 2.0 * micro_p_overlap_2 * micro_r_overlap_2 / (micro_p_overlap_2 + micro_r_overlap_2)
    
    if overlap_submission_3 == 0:
        micro_p_overlap_3 = 0.0
        micro_r_overlap_3 = 0.0
    else:
        micro_p_overlap_3 = 1.0 * correct_overlap_3 / overlap_submission_3
        micro_r_overlap_3 = 1.0 * correct_overlap_3 / overlap_tot_3
    if micro_p_overlap_3 + micro_r_overlap_3 == 0:
        micro_f1_overlap_3 = 0
    else:
        micro_f1_overlap_3 = 2.0 * micro_p_overlap_3 * micro_r_overlap_3 / (micro_p_overlap_3 + micro_r_overlap_3)

    if overlap_submission_4 == 0:
        micro_p_overlap_4 = 0.0
        micro_r_overlap_4 = 0.0
    else:
        micro_p_overlap_4 = 1.0 * correct_overlap_4 / overlap_submission_4
        micro_r_overlap_4 = 1.0 * correct_overlap_4 / overlap_tot_4
    if micro_p_overlap_4 + micro_r_overlap_4 == 0:
        micro_f1_overlap_4 = 0
    else:
        micro_f1_overlap_4 = 2.0 * micro_p_overlap_4 * micro_r_overlap_4 / (micro_p_overlap_4 + micro_r_overlap_4)

    # F1 for 3and4
    if overlap_submission_3and4 == 0:
        micro_p_overlap_3and4 = 0.0
        micro_r_overlap_3and4 = 0.0
    else:
        micro_p_overlap_3and4 = 1.0 * correct_overlap_3and4 / overlap_submission_3and4
        micro_r_overlap_3and4 = 1.0 * correct_overlap_3and4 / (overlap_tot_3 + overlap_tot_4)
    if micro_p_overlap_3and4 + micro_r_overlap_3and4 == 0:
        micro_f1_overlap_3and4 = 0
    else:
        micro_f1_overlap_3and4 = 2.0 * micro_p_overlap_3and4 * micro_r_overlap_3and4 / (micro_p_overlap_3and4 + micro_r_overlap_3and4)
    
    out_dict = {
        # eval + "_micro_F1": micro_f1,
        # eval + "_micro_F1_no_overlap": micro_f1_no_overlap*100,
        # eval + "_micro_F1_overlap": micro_f1_overlap*100,
        eval + "_F1_Two": micro_f1_overlap_2*100,
        eval + "_F1_Three": micro_f1_overlap_3*100,
    }
    if len(overlap_list_4)!=0:
        out_dict.update({
            eval + "_F1_Four": micro_f1_overlap_4*100,
            # eval + "_micro_F1_overlap_3and4": micro_f1_overlap_3and4*100,
            }
        ) 
        macro_overlap_f1 = (micro_f1_overlap_2 + micro_f1_overlap_3 + micro_f1_overlap_4)/3.0
    else:
        macro_overlap_f1 = (micro_f1_overlap_2 + micro_f1_overlap_3)/2.0
    out_dict.update({
        eval + "_F1_Mean": macro_overlap_f1*100
    })
    
    return out_dict


def get_output_labels(sigmoid_logits, theta=0., num_labels=-1):
    output = torch.zeros_like(sigmoid_logits).to(sigmoid_logits.device)
    mask = sigmoid_logits > theta

    if num_labels > 0:
        top_v, _ = torch.topk(sigmoid_logits, num_labels, dim=-1)
        top_v = top_v[:, -1]
        mask = (sigmoid_logits >= top_v.unsqueeze(1)) & mask

    output[mask] = 1.0
    output[:, 0] = (output.sum(dim=1) == 0.).to(sigmoid_logits.device)
    return output


# def get_output_labels_topk(sigmoid_logits, theta=0., num_labels=4):  # 最多返回4个labels
#     output = torch.zeros_like(sigmoid_logits).to(sigmoid_logits.device)
#     mask = sigmoid_logits > theta

#     if num_labels > 0:
#         top_v, _ = torch.topk(sigmoid_logits, num_labels, dim=-1)
#         top_v = top_v[:, -1]
#         mask = (sigmoid_logits >= top_v.unsqueeze(1)) & mask

#     output[mask] = 1.0
#     output[:, 0] = (output.sum(dim=1) == 0.).to(sigmoid_logits.device)
#     return output


def evaluate_long_tail(args, preds, best_threshold, features, id2rel, tag="dev", logger=None):
    cur_output = get_output_labels(preds, theta=best_threshold, num_labels=args.num_labels).cpu().numpy()
    cur_output[np.isnan(cur_output)] = 0
    cur_output = cur_output.astype(np.float32)

    ans = to_official(cur_output, features, id2rel)

    if "docred" in args.data_dir.lower():
        micro_f1, macro_f1_all, macro_f1_500, macro_f1_200, macro_f1_100 = 0, 0, 0, 0, 0
        if len(ans) > 0:
            micro_f1, macro_f1_all, macro_f1_500, macro_f1_200, macro_f1_100, macro_f1_greater_500 = \
                long_tail_evaluate_docred(ans, args.data_dir, eval=tag)
        cur_output = {
            # tag + "_F1": micro_f1 * 100,
            tag + "_Macro@all": macro_f1_all * 100,
            tag + "_Macro@500": macro_f1_500 * 100,
            tag + "_Macro@200": macro_f1_200 * 100,
            tag + "_Macro@100": macro_f1_100 * 100,
            tag + "_Macro@over_500": macro_f1_greater_500 * 100,
        }
        logger.write(json.dumps(cur_output, indent=4) + "\n")

    elif "dwie" in args.data_dir.lower():
        micro_f1, macro_f1_all, macro_f1_100, macro_f1_50, macro_f1_20, macro_f1_10, macro_f1_greater_100 = 0, 0, 0, 0, 0, 0, 0
        if len(ans) > 0:
            micro_f1, macro_f1_all, macro_f1_100, macro_f1_50, macro_f1_20, macro_f1_10, macro_f1_greater_100 = \
                long_tail_evaluate_dwie(ans, args.data_dir, eval=tag)
        cur_output = {
            # tag + "_F1": micro_f1 * 100,
            tag + "_Macro@all": macro_f1_all * 100,
            tag + "_Macro@100": macro_f1_100 * 100,
            tag + "_Macro@50": macro_f1_50 * 100,
            # tag + "_Macro@20": macro_f1_20 * 100,
            # tag + "_Macro@10": macro_f1_10 * 100,
            tag + "_Macro@over_100": macro_f1_greater_100 * 100,
        }
        logger.write(json.dumps(cur_output, indent=4) + "\n")
    return cur_output


def evaluate_multi_label(args, preds, best_threshold, features, id2rel, tag="dev", logger=None):
    cur_output = get_output_labels(preds, theta=best_threshold, num_labels=args.num_labels).cpu().numpy()
    cur_output[np.isnan(cur_output)] = 0
    cur_output = cur_output.astype(np.float32)

    ans = to_official(cur_output, features, id2rel)
    if len(ans) > 0:
        overlap_output = multi_label_evaluate(ans, args.data_dir, eval=tag)
        logger.write(json.dumps(overlap_output, indent=4) + "\n")
    return cur_output

