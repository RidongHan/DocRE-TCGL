import os
import ujson as json

dataset = "./dwie"
train_set = os.path.join(dataset, "train_annotated.json")
dev_set = os.path.join(dataset, "dev.json")
file_out = os.path.join(dataset, "rel_2_type.json")
file_out_1 = os.path.join(dataset, "type_pair_2_rel.json")

train_set = json.load(open(train_set))
dev_set = json.load(open(dev_set))

prob_threshold = 0.05

##### train set
r2type_pair = {}
r2head_type = {}
r2tail_type = {}

type_pair_2_rel = {}

for sample in train_set:
    title = sample['title']
    entities = sample["vertexSet"]
    labels = sample["labels"]

    for label in labels:
        if label['h'] == label['t']:  # For dwie dataset
            continue
        r = label['r']
        h = label['h']
        h_type = entities[h][0]["type"]
        t = label['t']
        t_type = entities[t][0]["type"]

        if r in r2head_type:
            if h_type in r2head_type[r]:
                r2head_type[r][h_type] += 1
            else:
                r2head_type[r][h_type] = 1
        else:
            r2head_type[r] = {}
            r2head_type[r][h_type] = 1

        if r in r2tail_type:
            if t_type in r2tail_type[r]:
                r2tail_type[r][t_type] += 1
            else:
                r2tail_type[r][t_type] = 1
        else:
            r2tail_type[r] = {}
            r2tail_type[r][t_type] = 1


        if r in r2type_pair:
            if [h_type, t_type] not in r2type_pair[r]:
                r2type_pair[r].append([h_type, t_type])
        else:
            r2type_pair[r] = []
            r2type_pair[r].append([h_type, t_type])

        if (h_type, t_type) in type_pair_2_rel:
            if r not in type_pair_2_rel[(h_type, t_type)]:
                type_pair_2_rel[(h_type, t_type)][r] = 1
            else:
                type_pair_2_rel[(h_type, t_type)][r] += 1
        else:
            type_pair_2_rel[(h_type, t_type)] = {}
            type_pair_2_rel[(h_type, t_type)][r] = 1

total_edge = 0
remove_total_edge = 0
for r in r2head_type:
    total_edge += len(r2head_type[r])
    total_edge += len(r2tail_type[r])
    
    r_head_total = 0
    for k, v in r2head_type[r].items():
        r_head_total += v
    for k in r2head_type[r]:
        r2head_type[r][k] = round(r2head_type[r][k] / (1.0*r_head_total), 4)

    r_tail_total = 0
    for k, v in r2tail_type[r].items():
        r_tail_total += v
    for k in r2tail_type[r]:
        r2tail_type[r][k] = round(r2tail_type[r][k] / (1.0*r_tail_total), 4)
    
    ## all
    print(r, r2head_type[r], r2tail_type[r])

    ## 去除 小于阈值的 0.01 (1%)的 type
    new_dict = {}
    for k, v in r2head_type[r].items():
        if v >= prob_threshold:
            new_dict[k] = v
    r2head_type[r] = new_dict
    remove_total_edge += len(r2head_type[r])

    new_dict = {}
    for k, v in r2tail_type[r].items():
        if v >= prob_threshold:
            new_dict[k] = v
    r2tail_type[r] = new_dict
    remove_total_edge += len(r2tail_type[r])
    print(r, r2head_type[r], r2tail_type[r])
    print()

print(total_edge, remove_total_edge)

result = {
    "r_2_type_pairs": r2type_pair,
    "r_2_head_types": r2head_type,
    "r_2_tail_types": r2tail_type
}

json.dump(result, open(file_out, "w"))

total_r = 0
total_r_remove = 0
for k in type_pair_2_rel:
    total_r += len(type_pair_2_rel[k])
    total_num=0
    for kk, vv in type_pair_2_rel[k].items():
        total_num += vv
    for kk in type_pair_2_rel[k]:
        type_pair_2_rel[k][kk] = round(type_pair_2_rel[k][kk]/(1.0*total_num), 4)
    print(k, len(type_pair_2_rel[k]), type_pair_2_rel[k])

    new_dict = {}
    for kk, vv in type_pair_2_rel[k].items():
        if vv >= 0.001:  # 0.1%
            new_dict[kk] = vv
    
    type_pair_2_rel[k] = new_dict
    total_r_remove += len(new_dict)
    print(k, len(type_pair_2_rel[k]), type_pair_2_rel[k])
    print()

print(total_r, total_r_remove)

print(len(type_pair_2_rel))
json.dump(type_pair_2_rel, open(file_out_1, "w"))

