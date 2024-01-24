import math
import torch
import torch.nn as nn
from random import choice
from torch.nn import functional as F
from opt_einsum import contract
from long_seq import process_long_input
from graph import GraphReasonLayer
import pickle, os


class TCL_Loss_R(nn.Module):
    def __init__(self, max_num_match_rels=20, max_num_labels=1):  ## docred: 20 , DWIE: 14
        super().__init__()
        self.max_num_match_rels = max_num_match_rels
        self.max_num_labels = max_num_labels  

    def forward(self, logits, Match_htr, match_rels, match_rels_mask, labels=None, labels_mask=None,):
        N, num_class = logits.shape
        expand_logits = logits.unsqueeze(1).repeat(1, self.max_num_match_rels, 1)

        minus_item = torch.gather(logits, dim=1, index=match_rels)

        sigmoid_minus_item = torch.sigmoid(minus_item)
        _, topk_index = torch.topk(sigmoid_minus_item * match_rels_mask, self.max_num_labels) 
        topk_mask = torch.zeros(minus_item.shape, dtype=torch.float).to(logits.device)
        topk_mask = torch.scatter(topk_mask, 1, topk_index, 1.0)

        diff = expand_logits - minus_item.unsqueeze(2)
        rank_tmp = (diff >= 0).float()
        rank = torch.sum(rank_tmp, dim=-1)
        
        unmatch_htr = 1.0 - Match_htr.unsqueeze(1).repeat(1, self.max_num_match_rels, 1)
        neg_rank = torch.sum(rank_tmp * unmatch_htr, dim=-1).clamp(min=0.1)
        loss = neg_rank / rank
        
        final_mask = topk_mask * match_rels_mask
        # match_rels_mask[:, 0] = 0.0  # 去除 NA 影响
        loss = (loss * final_mask).sum() / final_mask.sum()
        return loss
    

class TCL_Loss(nn.Module):
    def __init__(self, max_num_match_rels=20):  ## docred: 20 , DWIE: 14
        super().__init__()
        self.max_num_match_rels = max_num_match_rels

    def forward(self, logits, Match_htr, match_rels, match_rels_mask, labels=None, labels_mask=None,):
        N, num_class = logits.shape
        expand_logits = logits.unsqueeze(1).repeat(1, self.max_num_match_rels, 1)

        minus_item = torch.gather(logits, dim=1, index=match_rels)
        minus_item = minus_item.unsqueeze(2)

        diff = expand_logits - minus_item
        rank_tmp = (diff >= 0).float()
        rank = torch.sum(rank_tmp, dim=-1)
        
        unmatch_htr = 1.0 - Match_htr.unsqueeze(1).repeat(1, self.max_num_match_rels, 1)
        neg_rank = torch.sum(rank_tmp * unmatch_htr, dim=-1).clamp(min=0.1)
        loss = neg_rank / rank

        loss = (loss * match_rels_mask).sum() / match_rels_mask.sum()
        return loss
    

class Align_Loss(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, logits, origin_labels, Match_htr, labels_mask=None):
        align_labels = []
        for labels in origin_labels:  # 多标签 保留 一个: 默认保留第一个
            for label in labels:
                idx_list = [i for i in range(len(label)) if label[i] == 1]
                # idx = choice(idx_list)
                idx = idx_list[0]
                relation = [0] * len(label)
                relation[idx] = 1
                align_labels.append(torch.tensor(relation))
        align_labels = torch.stack(align_labels, dim=0).to(logits).float()  # 转 float

        # cross-entropy loss
        neg_log_prob = -1.0 * F.log_softmax(logits, dim=-1)
        loss = (align_labels * neg_log_prob).sum(dim=-1).mean()

        return loss


class DocREModel(nn.Module):
    def __init__(self, config, model, adj_matrix=None, block_size=64, num_labels=-1):
        super().__init__()
        self.config = config
        self.model = model
        self.emb_size = config.hidden_size
        self.block_size = block_size
        self.hidden_size = config.hidden_size
        self.adj_matrix = adj_matrix

        fea_dim = self.hidden_size * 2  ## add context features

        if self.config.use_type:
            self.gnn_hidden_size = self.hidden_size
            self.ner_embeddings = nn.Embedding(self.config.num_ner_type, self.hidden_size)
            nn.init.orthogonal_(self.ner_embeddings.weight)
            fea_dim += self.hidden_size  ## add type features

            if self.config.TCG:
                self.rel_embeddings = nn.Embedding(self.config.num_class, self.hidden_size)
                # nn.init.orthogonal_(self.rel_embeddings.weight, gain=nn.init.calculate_gain('leaky_relu', 0.2))
                nn.init.orthogonal_(self.rel_embeddings.weight)  # defaults: gain=1
                fea_dim += self.hidden_size  ## add relation features

                self.TCG_graph_reason = GraphReasonLayer(
                    ['head_type', 'tail_type'],
                    self.gnn_hidden_size,
                    self.gnn_hidden_size,
                    2,
                    graph_type=self.config.graph_type,
                    graph_drop=0.2,
                    xavier_init=True,
                )
                self.TCG_feature_fusion = nn.Linear(2 * self.hidden_size, self.hidden_size)
                self.TCG_align_loss_fnt = Align_Loss()
            
            if self.config.TCL:
                if self.config.topk_tcl:
                    self.TCL_loss_fnt = TCL_Loss_R(max_num_match_rels=self.config.max_num_match_rels, max_num_labels=self.config.max_num_labels)
                else:
                    self.TCL_loss_fnt = TCL_Loss(max_num_match_rels=self.config.max_num_match_rels)

        self.head_extractor = nn.Linear(fea_dim, self.hidden_size)
        self.tail_extractor = nn.Linear(fea_dim, self.hidden_size)
        self.bilinear = nn.Linear(self.hidden_size * block_size, self.config.num_class)    
        self.bce_loss_fnt = nn.BCEWithLogitsLoss(reduction='none')   
        

    def encode(self, input_ids, attention_mask):
        start_tokens = [self.config.cls_token_id]
        end_tokens = [self.config.sep_token_id]

        sequence_output, attention, pooler_output = process_long_input(self.model, input_ids, attention_mask, start_tokens, end_tokens)

        return sequence_output, attention, pooler_output


    def get_hrt(self, sequence_output, attention, entity_pos, hts):
        offset = 1 if self.config.transformer_type in ["bert", "roberta", "deberta"] else 0  # [CLS] or <s> 
        n, h, _, c = attention.size()
        hss, tss, rss = [], [], []
        for i in range(len(entity_pos)):  # bs
            entity_embs, entity_atts = [], []
            for e in entity_pos[i]:
                if len(e) > 1:
                    e_emb, e_att = [], []
                    for start, end in e:
                        if start + offset < c:
                            # In case the entity mention is truncated due to limited max seq length.
                            e_emb.append(sequence_output[i, start + offset])
                            e_att.append(attention[i, :, start + offset])
                            
                    if len(e_emb) > 0:
                        e_emb = torch.logsumexp(torch.stack(e_emb, dim=0), dim=0)
                        e_att = torch.stack(e_att, dim=0).mean(0)
                    else:
                        e_emb = torch.zeros(self.config.hidden_size).to(sequence_output)
                        e_att = torch.zeros(h, c).to(attention)
                else:
                    start, end = e[0]
                    if start + offset < c:
                        e_emb = sequence_output[i, start + offset]
                        e_att = attention[i, :, start + offset]
                    else:
                        e_emb = torch.zeros(self.config.hidden_size).to(sequence_output)
                        e_att = torch.zeros(h, c).to(attention)
            
                entity_embs.append(e_emb)
                entity_atts.append(e_att)

            entity_embs = torch.stack(entity_embs, dim=0)  # [n_e, d]
            entity_atts = torch.stack(entity_atts, dim=0)  # [n_e, h, seq_len]

            ht_i = torch.LongTensor(hts[i]).to(sequence_output.device)
            hs = torch.index_select(entity_embs, 0, ht_i[:, 0])
            ts = torch.index_select(entity_embs, 0, ht_i[:, 1])
        
            h_att = torch.index_select(entity_atts, 0, ht_i[:, 0])
            t_att = torch.index_select(entity_atts, 0, ht_i[:, 1])
            ht_att = (h_att * t_att).mean(1)
            ht_att = ht_att / (ht_att.sum(1, keepdim=True) + 1e-5)
            rs = contract("ld,rl->rd", sequence_output[i], ht_att)
            hss.append(hs)
            tss.append(ts)
            rss.append(rs)

        hss = torch.cat(hss, dim=0)
        tss = torch.cat(tss, dim=0)
        rss = torch.cat(rss, dim=0)

        return hss, rss, tss


    def get_nodes_repres(self, ):
        ner_nodes = self.ner_embeddings(torch.tensor(range(self.config.num_ner_type)).to(self.config.device))
        rel_nodes = self.rel_embeddings(torch.tensor(range(self.config.num_class)).to(self.config.device))
        all_nodes = torch.cat([ner_nodes, rel_nodes], dim=0)
        return all_nodes
    

    def forward(self, input_ids=None, input_mask=None, labels=None, labels_mask=None, entity_pos=None, hts=None, ht_types=None, Match_htr=None, match_rels=None, match_rels_mask=None):

        sequence_output, attention, _ = self.encode(input_ids.to(self.config.device), input_mask.to(self.config.device))
        hs, contexts, ts = self.get_hrt(sequence_output, attention, entity_pos, hts)
        hs_fea = [hs, contexts]
        ts_fea = [ts, contexts]

        if self.config.use_type:
            ht_types = ht_types.to(self.config.device)

            if self.config.TCG:
                all_nodes = self.get_nodes_repres()
                res_nodes = self.TCG_graph_reason(all_nodes, self.adj_matrix)
                res_ner_nodes = res_nodes[:self.config.num_ner_type, :]
                res_rel_nodes = res_nodes[self.config.num_ner_type:, :]

                h_types = self.ner_embeddings(ht_types[:, 0])
                t_types = self.ner_embeddings(ht_types[:, 1])
                hs_fea.append(h_types)
                ts_fea.append(t_types)

                fusion_feature = torch.tanh(self.TCG_feature_fusion(torch.cat([hs, ts], dim=-1)))
                align_logits = torch.matmul(fusion_feature, res_rel_nodes.t())
                align_probs = F.softmax(align_logits, dim=-1)
                extra_rel_features = torch.matmul(align_probs, res_rel_nodes)

                hs_fea.append(extra_rel_features)
                ts_fea.append(extra_rel_features)
            else:
                h_types = self.ner_embeddings(ht_types[:, 0])
                t_types = self.ner_embeddings(ht_types[:, 1])
                hs_fea.append(h_types)
                ts_fea.append(t_types)

        hs_fea = torch.cat(hs_fea, dim=1)
        ts_fea = torch.cat(ts_fea, dim=1) 
        hs = torch.tanh(self.head_extractor(hs_fea))
        ts = torch.tanh(self.tail_extractor(ts_fea))

        b1 = hs.view(-1, self.emb_size // self.block_size, self.block_size)
        b2 = ts.view(-1, self.emb_size // self.block_size, self.block_size)
        bl = (b1.unsqueeze(3) * b2.unsqueeze(2)).view(-1, self.emb_size * self.block_size)
        logits = self.bilinear(bl)  # [sum_all_hts, num_class]

        sigmoid_logits = torch.sigmoid(logits)
        output = {"sigmoid_logits": sigmoid_logits}

        if labels is not None:
            origin_labels = labels
            labels = [torch.tensor(label) for label in labels]
            labels = torch.cat(labels, dim=0).to(logits)  # [sum_hts, 97]
            re_loss = self.bce_loss_fnt(logits.float(), labels.float())
            bce_loss = re_loss.mean()
            assert not math.isnan(bce_loss)
            output["bce_loss"] = bce_loss

            if self.config.use_type:
                if self.config.TCG:
                    align_loss = self.TCG_align_loss_fnt(align_logits.float(), origin_labels, Match_htr.to(self.config.device))
                    output["align_loss"] = align_loss
            
                if self.config.TCL:
                    tcl_loss = self.TCL_loss_fnt(logits.float(), Match_htr.to(self.config.device), match_rels.to(self.config.device), match_rels_mask.to(self.config.device))
                    output["tcl_loss"] = tcl_loss

        return output

    