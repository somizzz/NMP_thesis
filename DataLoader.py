import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import random
import numpy as np
import os
import pickle
import ipdb
from utils import read_roidb, box_id, get_box_feats


class VrdPredDataset(Dataset):
    """docstring for VrdPred"""
    def __init__(self, mode = 'train', feat_mode = 'full', prior = False, ori_vgg=False, use_loc=False):
        super(VrdPredDataset, self).__init__()
        self.num_nodes = 21
        self.num_node_types = 101
        self.num_edge_types = 71
        self.num_edges = 41 #41 #30 #91
        if mode == 'train':
            self.mode = 'train'
        else:
            self.mode = 'test'
        self.feat_mode = feat_mode
        self.prior = prior
        # ----------- senmantic feature ------------- #
        #self.predicates_vec = np.load('/home/p_zhuzy/p_zhu/NMP/data/vrd_predicates_vec.npy')
        #self.objects_vec = np.load('/home/p_zhuzy/p_zhu/NMP/data/vrd_objects_vec.npy')
        self.predicates_vec = np.load('/home/p_zhuzy/p_zhu/NMP/data/vrd_predicates_vec_bert.npy')
        self.objects_vec = np.load('/home/p_zhuzy/p_zhu/NMP/data/vrd_objects_vec_bert.npy')
        # ------------ original roidb feature --------#
        self.roidb_read = read_roidb('/home/p_zhuzy/p_zhu/NMP/data/vrd_pred_graph_roidb.npz')
        self.roidb = self.roidb_read[self.mode]

        # Exclude self edges
        self.off_diag_idx = np.ravel_multi_index(
            np.where(np.ones((self.num_nodes, self.num_nodes)) - np.eye(self.num_nodes)),
            [self.num_nodes, self.num_nodes])

        # ------------ prior probability ------------- #
        # shape: [100, 100, 70] sum of the last dimension is 1
        f = open('./data/vrd_so_prior.pkl', 'rb')
        f.seek(0)
        self.rel_so_prior = pickle.load(f, encoding='bytes')    #[100, 100, 70]

        # ------------- prior of the existance of current [sub, obj] pair ---#
        # shape: [100, 100] sum=1
        self.prior_probs = np.load('./data/vrd_prior_prob.npy', encoding='bytes')

        self.use_loc = use_loc

    def get_adj(self, roidb_use):
        bbox_coordinates = np.zeros([self.num_edges, 20])
        matrix = np.eye(self.num_nodes)
        rel_rec = np.zeros([self.num_edges, self.num_nodes])
        rel_send = np.zeros([self.num_edges, self.num_nodes])
        sub_idx = box_id(roidb_use['sub_box_gt'], roidb_use['uni_box_gt'])
        obj_idx = box_id(roidb_use['obj_box_gt'], roidb_use['uni_box_gt'])

        for i in range(len(sub_idx)):
            sub_id = int(sub_idx[i])
            obj_id = int(obj_idx[i])
            rel_rec[i] = matrix[obj_id]
            rel_send[i] = matrix[sub_id]
            bbox_coordinates[i] = get_box_feats(roidb_use['uni_box_gt'][sub_id], roidb_use['uni_box_gt'][obj_id])
        # --------- cross entropy loss ---------#
        edges = np.zeros(self.num_edges) + self.num_edge_types - 1
        edges[:len(roidb_use['rela_gt'])] = roidb_use['rela_gt']
        edges = np.array(edges, dtype=np.int64)

        node_cls = np.zeros(self.num_nodes) + self.num_node_types - 1
        node_cls[:len(roidb_use['uni_gt'])] = roidb_use['uni_gt']
        node_cls = np.array(node_cls, dtype=np.int64)

        return edges, node_cls, rel_rec, rel_send, bbox_coordinates

    def train_item(self, roidb_use):
        if self.feat_mode == 'full':
            # --------- node feature ------------#
            #feats = np.load(roidb_use['uni_fc7'])
            old_path = roidb_use['uni_fc7']
            #print(f"原始路径: {old_path}")  # 调试输出
            
            new_path = old_path.replace(
                "/DATA5_DB8/data/yhu/VTransE/dsr_vrd_vgg_feats",
                "/home/p_zhuzy/p_zhu/NMP/data/feat/vg_rela_vgg_feats/vrd_rela_vgg_feats"
            )
            #print(f"修正后路径: {new_path}")  # 调试输出
            

            feats = np.load(new_path)  # 形状 (16, 4096)
            w2vec = np.array([self.objects_vec[int(x)] for x in roidb_use['uni_gt']])  # 形状 (7, 300)
            # 统一维度
            min_len = min(feats.shape[0], len(roidb_use['uni_gt']))
            nodes = np.zeros([self.num_nodes, 4096+768])
            nodes[:min_len, :4096] = feats[:min_len]  # 只填充前min_len行
            nodes[:min_len, 4096:] = w2vec[:min_len]  # 只填充前min_len行
            # feats = np.load(new_path)
            # w2vec = list(map(lambda x: self.objects_vec[int(x)], roidb_use['uni_gt']))
            # w2vec = np.reshape(np.array(w2vec),[-1, 300])
            # nodes = np.zeros([self.num_nodes, 4396])
            # nodes[:feats.shape[0], :4096] = feats
            # nodes[:feats.shape[0], 4096:] = w2vec   # [self.num_nodes, 4096+300]
        elif self.feat_mode == 'vis':
            old_path = roidb_use['uni_fc7']
            new_uni_path = old_path.replace(
                "/DATA5_DB8/data/yhu/VTransE/dsr_vrd_vgg_feats",
                "/home/p_zhuzy/p_zhu/NMP/data/feat/vg_rela_vgg_feats/vrd_rela_vgg_feats"
            )
            feats = np.load(new_uni_path)
            nodes = np.zeros([self.num_nodes, 4096])
            nodes[:feats.shape[0]] = feats
        elif self.feat_mode == 'sem':
            w2vec = list(map(lambda x: self.objects_vec[int(x)], roidb_use['uni_gt']))
            w2vec = np.reshape(np.array(w2vec),[-1, 768])
            nodes = np.zeros([self.num_nodes, 300])
            nodes[:w2vec.shape[0]] = w2vec

        prior_matrix = np.zeros([self.num_edges, self.num_edge_types])-0.5/self.num_edge_types
        for i in range(len(roidb_use['rela_gt'])):
            sub_cls = int(roidb_use['sub_gt'][i])
            obj_cls = int(roidb_use['obj_gt'][i])
            current_prior = self.rel_so_prior[sub_cls, obj_cls]
            # current_prior = -0.5*(current_prior+1.0/self.num_edge_types)
            current_prior = -0.5*(1.0/self.num_edge_types)
            prior_matrix[i, :(self.num_edge_types-1)] = current_prior

        # ------ region vgg feature --- initialize edge feature ---------#
        # sub_idx = box_id(roidb_use['sub_box_gt'], roidb_use['uni_box_gt'])
        # obj_idx = box_id(roidb_use['obj_box_gt'], roidb_use['uni_box_gt'])
        edge_feats = np.zeros([self.num_edges, 512])
        old_pred_path = roidb_use['pred_fc7']
        new_pred_path = old_pred_path.replace(
            "/DATA5_DB8/data/yhu/VTransE/dsr_vrd_vgg_feats",
            "/home/p_zhuzy/p_zhu/NMP/data/feat/vg_rela_vgg_feats/vrd_rela_vgg_feats"
        )
        pred_fc7 = np.load(new_pred_path)  # 替换后的路径
        rela_gt_len = len(roidb_use['rela_gt'])

        # 检查 pred_fc7 的行数是否足够
        if pred_fc7.shape[0] < rela_gt_len:
            # 如果 pred_fc7 的行数不足，填充零
            padded_pred_fc7 = np.zeros((rela_gt_len, 512))
            padded_pred_fc7[:pred_fc7.shape[0]] = pred_fc7
            pred_fc7 = padded_pred_fc7
        else:
            # 如果 pred_fc7 的行数足够，截取前 rela_gt_len 行
            pred_fc7 = pred_fc7[:rela_gt_len]

        edge_feats[:rela_gt_len] = pred_fc7
        #edge_feats[:len(roidb_use['rela_gt'])] = pred_fc7
        
        return nodes, edge_feats, prior_matrix

    def __getitem__(self, index):
        roidb_use = self.roidb[index]
        nodes, edge_feats, prior_matrix = self.train_item(roidb_use)
        edges, node_cls, rel_rec, rel_send, bbox_coordinates = self.get_adj(roidb_use)
        bbox_coordinates = torch.FloatTensor(bbox_coordinates)
        nodes = torch.FloatTensor(nodes)
        edges = torch.LongTensor(edges)
        node_cls = torch.LongTensor(node_cls)
        edge_feats = torch.FloatTensor(edge_feats)
        rel_rec = torch.FloatTensor(rel_rec)
        rel_send = torch.FloatTensor(rel_send)
        prior_matrix = torch.FloatTensor(prior_matrix)
        if self.prior:
            return nodes, edges, node_cls, edge_feats, rel_rec, rel_send, bbox_coordinates, prior_matrix
        else:
            return nodes, edges, node_cls, edge_feats, rel_rec, rel_send

    def __len__(self):
        return len(self.roidb)

class VrdRelaDataset(Dataset):
    """docstring for VrdRela"""
    def __init__(self, mode = 'train', feat_mode = 'full', prior=False, ori_vgg=False, use_loc=False):
        super(VrdRelaDataset, self).__init__()
        self.num_nodes = 21 #44 #21
        self.num_edges = 41 #30 #170
        self.num_node_types = 101
        self.num_edge_types = 71
        self.feat_mode = feat_mode
        self.prior = prior
        if mode == 'train':
            self.mode = 'train'
        else:
            self.mode = 'test'
            # if mode == 'test':
            self.num_nodes = 96 #63
            self.num_edges = self.num_nodes * (self.num_nodes-1)

        # ----------- senmantic feature ------------- #
        # self.predicates_vec = np.load('/home/p_zhuzy/p_zhu/NMP/data/vrd_predicates_vec.npy')
        # self.objects_vec = np.load('/home/p_zhuzy/p_zhu/NMP/data/vrd_objects_vec.npy')

        self.predicates_vec = np.load('/home/p_zhuzy/p_zhu/NMP/data/vrd_predicates_vec_bert.npy')
        self.objects_vec = np.load('/home/p_zhuzy/p_zhu/NMP/data/vrd_objects_vec_bert.npy')

        # ------------ original roidb feature --------#
        self.roidb_read = read_roidb('/home/p_zhuzy/p_zhu/NMP/data/vrd_rela_graph_roidb_iou_dis_{}_{}.npz'.format(0.5*10, 0.45*10))
        
        self.roidb = self.roidb_read[self.mode]

        # Exclude self edges
        self.off_diag_idx = np.ravel_multi_index(
            np.where(np.ones((self.num_nodes, self.num_nodes)) - np.eye(self.num_nodes)),
            [self.num_nodes, self.num_nodes])

        # ------------ prior probability ------------- #
        self.prior = prior
        f = open('/home/p_zhuzy/p_zhu/NMP/data/vrd_so_prior.pkl', 'rb')
        f.seek(0)
        self.rel_so_prior = pickle.load(f, encoding='bytes')    #[100, 100, 70]

        self.use_loc = use_loc

    def get_adj(self, roidb_use):
        bbox_coordinates = np.zeros([self.num_edges, 20])
        matrix = np.eye(self.num_nodes)
        rel_rec = np.zeros([self.num_edges, self.num_nodes])
        rel_send = np.zeros([self.num_edges, self.num_nodes])
        sub_idx = box_id(roidb_use['sub_box_dete'], roidb_use['uni_box_gt'])
        obj_idx = box_id(roidb_use['obj_box_dete'], roidb_use['uni_box_gt'])

        for i in range(len(sub_idx)):
            sub_id = int(sub_idx[i])
            obj_id = int(obj_idx[i])
            rel_rec[i] = matrix[obj_id]
            rel_send[i] = matrix[sub_id]
            bbox_coordinates[i] = get_box_feats(roidb_use['uni_box_gt'][sub_id], roidb_use['uni_box_gt'][obj_id])

        edges = np.zeros(self.num_edges) + self.num_edge_types-1
        edges[:len(roidb_use['rela_dete'])] = roidb_use['rela_dete']
        edges = np.array(edges, dtype=np.int64)

        node_cls = np.zeros(self.num_nodes) + self.num_node_types-1
        node_cls[:len(roidb_use['uni_gt'])] = roidb_use['uni_gt']
        node_cls = np.array(node_cls, dtype=np.int64)
        return edges, node_cls, rel_rec, rel_send, bbox_coordinates

    def train_item(self, roidb_use):
        # --------- node feature ------------#
        #feats = np.load(roidb_use['uni_fc7'])
        old_uni_path = roidb_use['uni_fc7']
        new_uni_path = old_uni_path.replace(
            "/DATA5_DB8/data/yhu/VTransE/vrd_rela_vgg_feats",
            "/home/p_zhuzy/p_zhu/NMP/data/feat/vg_rela_vgg_feats/vrd_rela_vgg_feats"
        )
        feats = np.load(new_uni_path)
        
        w2vec = list(map(lambda x: self.objects_vec[int(x)], roidb_use['uni_gt']))
        w2vec = np.reshape(np.array(w2vec),[-1, 768])

        if feats.shape[0] > self.num_nodes:
            index_box = np.sort(random.sample(range(feats.shape[0]), self.num_nodes))
            feats = feats[index_box, :]
            w2vec = w2vec[index_box, :]
            if self.feat_mode == 'full':
                nodes = np.concatenate([feats, w2vec], 1)   # [self.num_nodes, 4096+300]
            elif self.feat_mode == 'vis':
                nodes = feats
            elif self.feat_mode == 'sem':
                nodes = w2vec

            # --------- edge feature ------------#
            # edge_idx = roidb_use['edge_matrix'][index_box, :]
            # edge_idx = edge_idx[:, index_box]             # [self.num_nodes, self.num_nodes]
        else:
            if self.feat_mode == 'full':
                nodes = np.zeros([self.num_nodes, 4096 + 768])
                nodes[:feats.shape[0], :4096] = feats
                nodes[:feats.shape[0], 4096:] = w2vec   # [self.num_nodes, 4096+300]
            elif self.feat_mode == 'vis':
                nodes = np.zeros([self.num_nodes, 4096])
                nodes[:feats.shape[0]] = feats
            elif self.feat_mode == 'sem':
                nodes = np.zeros([self.num_nodes, 300])
                nodes[:w2vec.shape[0]] = w2vec
            
        prior_matrix = np.zeros([self.num_edges, self.num_edge_types])-0.5/self.num_edge_types
        for i in range(len(roidb_use['rela_dete'])):
            sub_cls = int(roidb_use['sub_dete'][i])
            obj_cls = int(roidb_use['obj_dete'][i])
            current_prior = self.rel_so_prior[sub_cls, obj_cls]
            # current_prior = -0.5*(current_prior+1.0/self.num_edge_types)
            current_prior = -0.5*(1.0/self.num_edge_types)
            prior_matrix[i, :(self.num_edge_types-1)] = current_prior

        # ------ region vgg feature --- initialize edge feature ---------#
        sub_idx = box_id(roidb_use['sub_box_dete'], roidb_use['uni_box_gt'])
        obj_idx = box_id(roidb_use['obj_box_dete'], roidb_use['uni_box_gt'])
        edge_feats = np.zeros([self.num_edges, 512])

        # pred_fc7 = np.load(roidb_use['pred_fc7'])

        # edge_feats[:len(roidb_use['rela_dete'])] = pred_fc7

        # for i in range(len(sub_idx)):
        #     edge_feats[int(sub_idx[i]),int(obj_idx[i])] = pred_fc7[i]
        # edge_feats = np.reshape(edge_feats, [self.num_nodes ** 2, -1])
        # edge_feats = edge_feats[self.off_diag_idx]

        return nodes, edge_feats, prior_matrix

    def __getitem__(self, index):
        roidb_use = self.roidb[index]
        nodes, edge_feats, prior_matrix = self.train_item(roidb_use)
        edges, node_cls, rel_rec, rel_send, bbox_coordinates = self.get_adj(roidb_use)
        bbox_coordinates = torch.FloatTensor(bbox_coordinates)
        nodes = torch.FloatTensor(nodes)
        edges = torch.LongTensor(edges)
        node_cls = torch.LongTensor(node_cls)
        edge_feats = torch.FloatTensor(edge_feats)
        rel_rec = torch.FloatTensor(rel_rec)
        rel_send = torch.FloatTensor(rel_send)

        prior_matrix = torch.FloatTensor(prior_matrix)
        if self.prior:
            return nodes, edges, node_cls, edge_feats, rel_rec, rel_send, bbox_coordinates, prior_matrix
        else:
            return nodes, edges, node_cls, edge_feats, rel_rec, rel_send

    def __len__(self):
        return len(self.roidb)

class VgPredDataset(Dataset):
    """docstring for VgPred"""
    def __init__(self, mode = 'train', feat_mode = 'full', prior = False, ori_vgg=False, use_loc=False):
        super(VgPredDataset, self).__init__()
        self.num_nodes = 110 #98
        self.num_edge_types = 101
        self.num_node_types = 201
        self.num_edges = 490 #352
        if mode == 'train':
            self.mode = 'train'
        else:
            self.mode = 'test'
        self.feat_mode = feat_mode
        self.prior = prior
        # ----------- senmantic feature ------------- #
        self.predicates_vec = np.load('./data/vg_predicates_vec.npy')
        self.objects_vec = np.load('./data/vg_objects_vec.npy')

        # ------------ original roidb feature --------#
        self.roidb_read = read_roidb('./data/vg_pred_graph_roidb.npz')
        self.roidb = self.roidb_read[self.mode]

        self.rel_so_prior = np.load('./data/vg_so_prior.npy')    #[201, 201, 100]

        self.use_loc = use_loc

    def get_adj(self, roidb_use):
        bbox_coordinates = np.zeros([self.num_edges, 20])
        matrix = np.eye(self.num_nodes)
        rel_rec = np.zeros([self.num_edges, self.num_nodes])
        rel_send = np.zeros([self.num_edges, self.num_nodes])
        sub_idx = box_id(roidb_use['sub_box_gt'], roidb_use['uni_box_gt'])
        obj_idx = box_id(roidb_use['obj_box_gt'], roidb_use['uni_box_gt'])

        for i in range(len(sub_idx)):
            sub_id = int(sub_idx[i])
            obj_id = int(obj_idx[i])
            rel_rec[i] = matrix[obj_id]
            rel_send[i] = matrix[sub_id]
            bbox_coordinates[i] = get_box_feats(roidb_use['uni_box_gt'][sub_id], roidb_use['uni_box_gt'][obj_id])

        edges = np.zeros(self.num_edges) + self.num_edge_types - 1
        edges[:len(roidb_use['rela_gt'])] = roidb_use['rela_gt']
        edges = np.array(edges, dtype=np.int64)

        node_cls = np.zeros(self.num_nodes) + self.num_node_types-1
        node_cls[:len(roidb_use['uni_gt'])] = roidb_use['uni_gt']
        node_cls = np.array(node_cls, dtype=np.int64)
        return edges, node_cls, rel_rec, rel_send, bbox_coordinates

    def train_item(self, roidb_use):

        if self.feat_mode == 'full':
            # --------- node feature ------------#
            feats = np.load(roidb_use['uni_fc7'])
            w2vec = list(map(lambda x: self.objects_vec[int(x)], roidb_use['uni_gt']))
            w2vec = np.reshape(np.array(w2vec),[-1, 300])
            nodes = np.zeros([self.num_nodes, 4396])
            nodes[:feats.shape[0], :4096] = feats
            nodes[:feats.shape[0], 4096:] = w2vec   # [self.num_nodes, 4096+300]
        elif self.feat_mode == 'vis':
            feats = np.load(roidb_use['uni_fc7'])
            nodes = np.zeros([self.num_nodes, 4096])
            nodes[:feats.shape[0]] = feats
        elif self.feat_mode == 'sem':
            w2vec = list(map(lambda x: self.objects_vec[int(x)], roidb_use['uni_gt']))
            w2vec = np.reshape(np.array(w2vec),[-1, 300])
            nodes = np.zeros([self.num_nodes, 300])
            nodes[:w2vec.shape[0]] = w2vec

        
        # prior_matrix = np.zeros([self.num_edges, self.num_edge_types])
        prior_matrix = np.zeros([self.num_edges, self.num_edge_types])-0.5/self.num_edge_types
        for i in range(len(roidb_use['rela_gt'])):
            sub_cls = int(roidb_use['sub_gt'][i])
            obj_cls = int(roidb_use['obj_gt'][i])
            current_prior = self.rel_so_prior[sub_cls, obj_cls]
            current_prior = -0.5*(current_prior+1.0/self.num_edge_types)
            # current_prior = -1.0*(current_prior+1.0/self.num_edge_types)
            prior_matrix[i, :(self.num_edge_types-1)] = current_prior

        # ------ region vgg feature --- initialize edge feature ---------#
        # sub_idx = box_id(roidb_use['sub_box_gt'], roidb_use['uni_box_gt'])
        # obj_idx = box_id(roidb_use['obj_box_gt'], roidb_use['uni_box_gt'])
        edge_feats = np.zeros([self.num_edges, 512])
        pred_fc7 = np.load(roidb_use['pred_fc7'])
        edge_feats[:len(roidb_use['rela_gt'])] = pred_fc7
        return nodes, edge_feats, prior_matrix

    def __getitem__(self, index):
        roidb_use = self.roidb[index]
        nodes, edge_feats, prior_matrix = self.train_item(roidb_use)
        edges, node_cls, rel_rec, rel_send, bbox_coordinates = self.get_adj(roidb_use)
        nodes = torch.FloatTensor(nodes)
        edges = torch.LongTensor(edges)
        node_cls = torch.LongTensor(node_cls)
        edge_feats = torch.FloatTensor(edge_feats)
        rel_rec = torch.FloatTensor(rel_rec)
        rel_send = torch.FloatTensor(rel_send)
        prior_matrix = torch.FloatTensor(prior_matrix)
        bbox_coordinates = torch.FloatTensor(bbox_coordinates)
        if self.prior:
            return nodes, edges, node_cls, edge_feats, rel_rec, rel_send, bbox_coordinates, prior_matrix
        else:
            return nodes, edges, node_cls, edge_feats, rel_rec, rel_send

    def __len__(self):
        return len(self.roidb)

def load_dataset(data_set='vrd', ori_vgg=False, dataset='pred', level='image', batch_size=32, eval_batch_size=32, shuffle=False,  feat_mode='full', prior=False):
    if data_set == 'vrd':
        if dataset=='pred' and level=='image':
            load_func_name = VrdPredDataset
        elif dataset=='rela' and level=='image':
            load_func_name = VrdRelaDataset
    else:
        load_func_name = VgPredDataset

    train_data = load_func_name(mode='train', feat_mode = feat_mode, prior=True, ori_vgg=ori_vgg)
    val_data = load_func_name(mode='test', feat_mode = feat_mode, prior=True, ori_vgg=ori_vgg)
    test_data = load_func_name(mode='test', feat_mode = feat_mode, prior=True, ori_vgg=ori_vgg)

    train_loader = DataLoader(train_data, shuffle=shuffle, batch_size=batch_size)
    val_loader = DataLoader(val_data, shuffle=False, batch_size=eval_batch_size)
    test_loader = DataLoader(test_data, shuffle=False, batch_size=eval_batch_size)

    return train_loader, val_loader, test_loader