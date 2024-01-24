import os
import torch
import numpy as np
import ujson as json
from tqdm import tqdm


def read_docred(args, file_in, file_out, tokenizer, docred_rel2id, docred_ner2id, type_pair_2_rel=None, max_seq_length=1024, logger=None):
    if os.path.exists(file_out):
        with open(file_out, "r") as fh:
            features = json.load(fh)
            logger.write(file_out + " has been loaded! | # of the document: " + str(len(features)) + "\n")
        return features

    i_line = 0
    pos_samples = 0
    neg_samples = 0
    self_relation_num = 0
    single_or_no_entity_num = 0
    features = []
    if file_in == "":
        return None
    with open(file_in, "r") as fh:
        data = json.load(fh)

    token_length = max_seq_length

    for sample in tqdm(data, desc="Example"):
        mentions_info = []  # for graph  # start | end | entityid | sentid
        entities_info = []  # for graph
        sentences_info = []  # for graph

        ###### for sentences_info
        Ls = [0]  
        for sent in sample['sents']:
            Ls.append(len(sent))  
        ###### for sentences_info

        sents = []
        sent_map = []

        entities = sample['vertexSet']
        if len(entities) < 2:  # For dwie dataset
            single_or_no_entity_num += 1
            # print(sample)
            continue
        entity_start, entity_end = [], []
        for entity in entities:
            for mention in entity:
                sent_id = mention["sent_id"]
                pos = mention["pos"]
                entity_start.append((sent_id, pos[0],))
                entity_end.append((sent_id, pos[1] - 1,))
        for i_s, sent in enumerate(sample['sents']):
            new_map = {}
            for i_t, token in enumerate(sent):
                tokens_wordpiece = tokenizer.tokenize(token)
                if (i_s, i_t) in entity_start:
                    tokens_wordpiece = ["*"] + tokens_wordpiece
                if (i_s, i_t) in entity_end:
                    tokens_wordpiece = tokens_wordpiece + ["*"]
                new_map[i_t] = len(sents)
                sents.extend(tokens_wordpiece)
            new_map[i_t + 1] = len(sents)
            sent_map.append(new_map)

        ###### for sentences_info
        new_Ls = [0]  
        for idx in range(1, len(Ls)):
            new_Ls.append(sent_map[idx-1][Ls[idx]])
            if new_Ls[idx-1] < token_length:
                sentences_info.append([new_Ls[idx-1], new_Ls[idx], -1, idx-1])  
        ###### for sentences_info

        train_triple = {}
        relation_set = []
        if "labels" in sample:
            for label in sample['labels']:
                if label['h'] == label['t']:  # For dwie dataset
                    self_relation_num += 1
                    continue
                evidence = label['evidence']
                r = int(docred_rel2id[label['r']])
                if r not in relation_set:
                    relation_set.append(r)
                if (label['h'], label['t']) not in train_triple:
                    train_triple[(label['h'], label['t'])] = [
                        {'relation': r, 'evidence': evidence}]
                else:
                    train_triple[(label['h'], label['t'])].append(
                        {'relation': r, 'evidence': evidence})

        entity_pos = []
        for e_id, e in enumerate(entities):
            entities_info.append([-1, -1, e_id, -1])
            entity_pos.append([])
            for m in e:
                start = sent_map[m["sent_id"]][m["pos"][0]]
                end = sent_map[m["sent_id"]][m["pos"][1]]
                entity_pos[-1].append((start, end,))
                if start < token_length:
                    mentions_info.append([start, end, e_id, m["sent_id"]])

        relations, hts = [], []
        ht_types = []
        Match_htr = []
        match_rels = []
        match_rels_mask = []

        for h, t in train_triple.keys():
            relation = [0] * len(docred_rel2id)
            for mention in train_triple[h, t]:
                relation[mention["relation"]] = 1
                evidence = mention["evidence"]
            relations.append(relation)
            hts.append([h, t])
            pos_samples += 1

            h_type = entities[h][0]["type"]
            t_type = entities[t][0]["type"]
            ht_types.append([docred_ner2id[h_type], docred_ner2id[t_type]])
            if type_pair_2_rel is not None:
                cur_key = "('{}', '{}')".format(h_type, t_type)
                if cur_key in type_pair_2_rel:
                    rel_list = type_pair_2_rel[cur_key] + [0]
                else:
                    rel_list = [0]
                tmp_match_htr = np.array([0.0] * args.num_class)
                tmp_match_htr[rel_list] = 1.0
                tmp_match_htr = tmp_match_htr.tolist()
                # print(rel_list)
                assert len(rel_list) == int(sum(tmp_match_htr))
                Match_htr.append(tmp_match_htr)

                sorted_rel_list = sorted(rel_list) + [0] * (args.max_num_match_rels - len(rel_list))
                sorted_rel_list_mask = [1.0] * len(rel_list) + [0.] * (args.max_num_match_rels - len(rel_list))
                match_rels.append(sorted_rel_list)
                match_rels_mask.append(sorted_rel_list_mask)

        for h in range(len(entities)):
            for t in range(len(entities)):
                if h != t and [h, t] not in hts:
                    relation = [1] + [0] * (len(docred_rel2id) - 1)
                    relations.append(relation)
                    hts.append([h, t])
                    neg_samples += 1

                    h_type = entities[h][0]["type"]
                    t_type = entities[t][0]["type"]
                    ht_types.append([docred_ner2id[h_type], docred_ner2id[t_type]])
                    if type_pair_2_rel is not None:
                        cur_key = "('{}', '{}')".format(h_type, t_type)
                        if cur_key in type_pair_2_rel:
                            rel_list = type_pair_2_rel[cur_key] + [0]
                        else:
                            rel_list = [0]
                        tmp_match_htr = np.array([0.0] * args.num_class)
                        tmp_match_htr[rel_list] = 1.0
                        tmp_match_htr = tmp_match_htr.tolist()
                        # print(rel_list)
                        assert len(rel_list) == int(sum(tmp_match_htr))
                        Match_htr.append(tmp_match_htr)

                        sorted_rel_list = sorted(rel_list) + [0] * (args.max_num_match_rels - len(rel_list))
                        sorted_rel_list_mask = [1.0] * len(rel_list) + [0.] * (args.max_num_match_rels - len(rel_list))
                        match_rels.append(sorted_rel_list)
                        match_rels_mask.append(sorted_rel_list_mask)

        assert len(relations) == len(entities) * (len(entities) - 1)

        sents = sents[:token_length - 2]
        input_ids = tokenizer.convert_tokens_to_ids(sents)
        input_ids = tokenizer.build_inputs_with_special_tokens(input_ids)

        labels_mask = [1.0] * len(train_triple) + [0.0] * (len(entities) * (len(entities)-1) - len(train_triple))
        assert len(labels_mask) == len(entities) * (len(entities)-1), "Error labels_mask"
        i_line += 1

        feature = {'input_ids': input_ids,
                   'labels': relations,
                   'labels_mask': labels_mask,
                   'entity_pos': entity_pos,
                   'hts': hts,
                   'ht_types': ht_types,
                   'Match_htr': Match_htr,
                   'match_rels': match_rels,
                   'match_rels_mask': match_rels_mask,
                   'title': sample['title'], 
                   'mentions_info': mentions_info,
                   'entities_info': entities_info,
                   'sentences_info': sentences_info,
                   }
        features.append(feature)
        
    json.dump(features, open(file_out, "w"))
    if "dwie" in file_in:
        logger.write("# of self-relation {}.\n".format(self_relation_num))
        logger.write("# of 1/0 entities {}.\n".format(single_or_no_entity_num))
    logger.write("# of documents {}.\n".format(i_line))
    logger.write("# of positive examples {}.\n".format(pos_samples))
    logger.write("# of negative examples {}.\n".format(neg_samples))

    return features


def collate_fn(batch):
    max_len = max([len(f["input_ids"]) for f in batch])
    input_ids = [f["input_ids"] + [0] * (max_len - len(f["input_ids"])) for f in batch]
    input_mask = [[1.0] * len(f["input_ids"]) + [0.0] * (max_len - len(f["input_ids"])) for f in batch]
    labels = [f["labels"] for f in batch]
    entity_pos = [f["entity_pos"] for f in batch]
    hts = [f["hts"] for f in batch]
    ht_types = []
    Match_htr = []
    labels_mask = []
    match_rels = []
    match_rels_mask = []
    for f in batch:
        ht_types += f["ht_types"]
        Match_htr += f["Match_htr"]
        labels_mask += f["labels_mask"]
        match_rels += f["match_rels"]
        match_rels_mask += f["match_rels_mask"]

    input_ids = torch.tensor(input_ids, dtype=torch.long)  # [bs, max_len]
    input_mask = torch.tensor(input_mask, dtype=torch.float)
    ht_types = torch.tensor(ht_types, dtype=torch.long)
    Match_htr = torch.tensor(Match_htr, dtype=torch.float)
    labels_mask = torch.tensor(labels_mask, dtype=torch.float)
    match_rels = torch.tensor(match_rels, dtype=torch.int64)
    match_rels_mask = torch.tensor(match_rels_mask, dtype=torch.float)

    output = {
        "input_ids": input_ids,
        "input_mask": input_mask,
        "labels": labels,
        "labels_mask": labels_mask,
        "entity_pos": entity_pos,
        "hts": hts,
        "ht_types": ht_types,
        "Match_htr": Match_htr,
        "match_rels": match_rels,
        "match_rels_mask": match_rels_mask,
    }
    return output


def get_adjacent_matrix(args, rel2id, ner2id):
    rel_2_type_file = os.path.join(args.data_dir, args.rel_2_type_file)
    rel_2_type = json.load(open(rel_2_type_file, "r"))
    rel_2_head_types = rel_2_type["r_2_head_types"]
    rel_2_tail_types = rel_2_type["r_2_tail_types"]

    num_ner_type = len(ner2id)
    num_nodes = num_ner_type + args.num_class
    adj_matrix_head_type = torch.eye(num_nodes, dtype=torch.float)
    adj_matrix_tail_type = torch.eye(num_nodes, dtype=torch.float)
    # adj_matrix_head_type[:num_ner_type, num_ner_type] = 1.0
    # adj_matrix_head_type[num_ner_type, :num_ner_type] = 1.0
    # adj_matrix_tail_type[:num_ner_type, num_ner_type] = 1.0
    # adj_matrix_tail_type[num_ner_type, :num_ner_type] = 1.0

    for k in rel_2_head_types:
        rel_id = rel2id[k]
        for ner in rel_2_head_types[k].keys():
            ner_id = ner2id[ner]
            adj_matrix_head_type[ner_id][rel_id + num_ner_type] = 1.0
            adj_matrix_head_type[rel_id + num_ner_type][ner_id] = 1.0

    for k in rel_2_tail_types:
        rel_id = rel2id[k]
        for ner in rel_2_tail_types[k].keys():
            ner_id = ner2id[ner]
            adj_matrix_tail_type[ner_id][rel_id + num_ner_type] = 1.0
            adj_matrix_tail_type[rel_id + num_ner_type][ner_id] = 1.0

    adj_matrix = [adj_matrix_head_type.to(args.device), adj_matrix_tail_type.to(args.device)]
    
    return adj_matrix