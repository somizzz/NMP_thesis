'''
	Feed the path of vgg features into roidb file
'''

import numpy as np
import os
import ipdb
from ass_fun import *
from tqdm import tqdm
import h5py
import json
from sklearn.decomposition import PCA

import torch
import torch.nn as nn

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='vrd',
                    help='dataset: vrd or vg')
parser.add_argument('--data_type', type=str, default='pred',
                    help='data_type: pred or rela')
parser.add_argument('--ori_vgg', action='store_true', default=False,
                    help='original vgg')
parser.add_argument('--random_vgg', action='store_true', default=False,
                    help='random initialize vgg')

args = parser.parse_args()
data_type = args.data_type
dataset = args.dataset
use_ori_vgg = args.ori_vgg
use_random_vgg = args.random_vgg
print(args)
def compute_iou_each(box1, box2):
    '''
    function: calculate the iou based on the box ordinates
    box1: [x_min, y_min, x_max, y_max]
    '''
    xA = max(box1[0], box2[0])
    yA = max(box1[1], box2[1])
    xB = min(box1[2], box2[2])
    yB = min(box1[3], box2[3])

    if xB<xA or yB<yA:
        IoU = 0
    else:
        area_I = (xB - xA + 1) * (yB - yA + 1)
        area1 = (box1[2] - box1[0] + 1)*(box1[3] - box1[1] + 1)
        area2 = (box2[2] - box2[0] + 1)*(box2[3] - box2[1] + 1)
        IoU = area_I/float(area1 + area2 - area_I)
    return IoU

def compute_distance(box1, box2):
    cx1 = (box1[0] + box1[2])/2.0
    cy1 = (box1[1] + box1[3])/2.0
    cx2 = (box2[0] + box2[2])/2.0
    cy2 = (box2[1] + box2[3])/2.0

    x_min = min(box1[0], box2[0])
    y_min = min(box1[1], box2[1])
    x_max = max(box1[2], box2[2])
    y_max = max(box1[3], box2[3])

    I = (cx1 - cx2)**2 + (cy1 - cy2)**2
    U = (x_min - x_max)**2 + (y_min - y_max)**2
    dis = np.sqrt(I/float(U))
    return dis
#============= the max rela of one image ========================#
def count_max_rela(train_roidb, test_roidb):
	rela_total = []
	for name, roidb in zip(['train', 'test'], [train_roidb, test_roidb]):
		rela = np.zeros(len(roidb), dtype=np.int64)
		for i in range(len(roidb)):
			rela[i] = int(len(roidb[i]['rela_gt']))
		r_max = np.max(rela)
		r_min = np.min(rela)
		r_mean = np.mean(rela)
		print("{0} | max: {1} | mean: {2} | min: {3}".format(name, r_max, r_mean, r_min))
		# VRD
		# train | max: 34 | mean: 8.03042328042 | min: 1
		# test | max: 41 | mean: 8.0 | min: 1
		# VG
		# train | max: 490 | mean: 10.8853836355 | min: 1
		# test | max: 352 | mean: 11.0894500735 | min: 1
		rela_total.append(rela)
	return
#============== pred the max objects in one image ======================#
def unique_gt(box_gt, cls_gt, fc7):
	_, idx = np.unique(box_gt, axis=0, return_index=True)
	idx = np.sort(idx)
	uni_box_gt = box_gt[idx]
	uni_cls_gt = cls_gt[idx]
	uni_fc7 = fc7[idx]
	return uni_box_gt, uni_cls_gt, uni_fc7

def new_id(uni_box_gt, ori_box_gt, ori_cls_gt):
	new_idx = np.zeros_like(ori_cls_gt)
	for i in range(len(ori_box_gt)):
		for j in range(len(uni_box_gt)):
			if np.array_equal(ori_box_gt[i], uni_box_gt[j]):
				new_idx[i] = j
	return new_idx
#/home/p_zhuzy/p_zhu/NMP/data/feat/vg_rela_vgg_feats/vrd_rela_vgg_feats/train/uni_fc7
def pred_write_feat_into_roidb(save_path, train_roidb, test_roidb, dataset='vrd', edges_types = 70):
	edge_total = []
	node_total = []
	new_roidb = {}
	pbar = tqdm(total=len(train_roidb)+len(test_roidb))
	for name, roidb in zip(['train', 'test'], [train_roidb, test_roidb]):
		full_path = save_path + '/' + name
		feat_name = ['pred_pool5', 'pred_fc7', 'pool5', 'fc7', 'sub_fc7', 'ob_fc7']
		check_path_exists(full_path+'/uni_fc7')
		rela = np.zeros(len(roidb), dtype=np.int64)
		edges = []
		nodes = []
		for i in range(len(roidb)):
			#====== feats ============#
			sub_fc7 = np.load(os.path.join(full_path, 'sub_fc7', os.path.basename(roidb[i]['image'])+'.npy'))
			ob_fc7 = np.load(os.path.join(full_path, 'ob_fc7', os.path.basename(roidb[i]['image'])+'.npy'))
			fc7 = np.concatenate([sub_fc7, ob_fc7], 0)

			# sub_pool5 = np.load(os.path.join(full_path, 'sub_pool5', os.path.basename(roidb[i]['image'])+'.npy'))
			# ob_pool5 = np.load(os.path.join(full_path, 'ob_pool5', os.path.basename(roidb[i]['image'])+'.npy'))
			# pool5 = np.concatenate([sub_pool5, ob_pool5], 0)

			box_gt = np.concatenate([roidb[i]['sub_box_gt'], roidb[i]['obj_box_gt']], 0)
			cls_gt = np.concatenate([roidb[i]['sub_gt'], roidb[i]['obj_gt']], 0)

			uni_box_gt, uni_cls_gt, uni_fc7 = unique_gt(box_gt, cls_gt, fc7)
			sub_idx = new_id(uni_box_gt, roidb[i]['sub_box_gt'], roidb[i]['sub_gt'])
			obj_idx = new_id(uni_box_gt, roidb[i]['obj_box_gt'], roidb[i]['obj_gt'])
			# ipdb.set_trace()
			edge_matrix = np.zeros([len(uni_cls_gt), len(uni_cls_gt)]) + edges_types
			for j, x, y in zip(np.array(range(len(sub_idx))), sub_idx, obj_idx):
				edge_matrix[int(x)][int(y)] = roidb[i]['rela_gt'][j]

			nodes.append(len(uni_cls_gt))
			edges.append(len(roidb[i]['rela_gt']))

			roidb[i]['uni_box_gt'] = uni_box_gt
			roidb[i]['uni_gt'] = uni_cls_gt
			roidb[i]['edge_matrix'] = edge_matrix
			roidb[i]['sub_idx'] = sub_idx
			roidb[i]['obj_idx'] = obj_idx

			pred_pool5_path = os.path.join(full_path, 'pred_pool5', os.path.basename(roidb[i]['image'])+'.npy')
			pred_fc7_path = os.path.join(full_path, 'pred_fc7', os.path.basename(roidb[i]['image'])+'.npy')
			uni_fc7_path = os.path.join(full_path, 'uni_fc7', os.path.basename(roidb[i]['image'])+'.npy')
			img_fc7_path = os.path.join(full_path, 'fc7', os.path.basename(roidb[i]['image'])+'.npy')
			img_pool5_path = os.path.join(full_path, 'pool5', os.path.basename(roidb[i]['image'])+'.npy')

			if os.path.exists(uni_fc7_path):
				pass
			else:
				np.save(uni_fc7_path, uni_fc7)

			roidb[i]['pred_pool5'] = pred_pool5_path
			roidb[i]['pred_fc7'] = pred_fc7_path
			roidb[i]['uni_fc7'] = uni_fc7_path
			roidb[i]['img_fc7'] = img_fc7_path
			roidb[i]['img_pool5'] = img_pool5_path
			pbar.update(1)
		new_roidb[name] = roidb

		print("nodes: {0} | max: {1} | mean: {2} | min: {3}".format(name, np.max(nodes), np.mean(nodes), np.min(nodes)))
		print("edges: {0} | max: {1} | mean: {2} | min: {3}".format(name, np.max(edges), np.mean(edges), np.min(edges)))

		edge_total.append(edges)
		node_total.append(nodes)
	pbar.close()
	np.savez('/home/p_zhuzy/p_zhu/NMP/data/{}_pred_graph_roidb.npz'.format(dataset), new_roidb)
	return

def rela_write_feat_into_roidb(save_path, train_roidb, test_roidb, dataset='vrd', edges_types = 70):
	edge_total = []
	node_total = []
	new_roidb = {}
	pbar = tqdm(total=len(test_roidb)+len(train_roidb))
	# ------- test rela -------------#
	for name, roidb in zip(['train', 'test'], [train_roidb, test_roidb]):
		full_path = save_path + '/' + name
		# feat_name = ['pool5', 'fc7', 'sub_fc7', 'ob_fc7']
		feat_name = ['pred_pool5', 'pred_fc7', 'pool5', 'fc7', 'sub_fc7', 'ob_fc7']
		check_path_exists(full_path+'/uni_fc7')
		rela = np.zeros(len(roidb), dtype=np.int64)
		edges = []
		nodes = []
		for i in range(len(roidb)):
			#====== feats ============#
			sub_fc7 = np.load(os.path.join(full_path, 'sub_fc7', os.path.basename(roidb[i]['image'])+'.npy'))
			ob_fc7 = np.load(os.path.join(full_path, 'ob_fc7', os.path.basename(roidb[i]['image'])+'.npy'))
			fc7 = np.concatenate([sub_fc7, ob_fc7], 0)

			box_gt = np.concatenate([roidb[i]['sub_box_dete'], roidb[i]['obj_box_dete']], 0)
			cls_gt = np.concatenate([roidb[i]['sub_dete'], roidb[i]['obj_dete']], 0)

			uni_box_gt, uni_cls_gt, uni_fc7 = unique_gt(box_gt, cls_gt, fc7)
			sub_idx = new_id(uni_box_gt, roidb[i]['sub_box_dete'], roidb[i]['sub_dete'])
			obj_idx = new_id(uni_box_gt, roidb[i]['obj_box_dete'], roidb[i]['obj_dete'])
			
			edge_matrix = np.zeros([len(uni_cls_gt), len(uni_cls_gt)]) + edges_types
			for j, x, y in zip(np.array(range(len(sub_idx))), sub_idx, obj_idx):
				edge_matrix[int(x)][int(y)] = roidb[i]['rela_dete'][j]

			nodes.append(len(uni_cls_gt))
			edges.append(len(roidb[i]['rela_gt']))

			
			roidb[i]['uni_box_gt'] = uni_box_gt
			roidb[i]['uni_gt'] = uni_cls_gt
			roidb[i]['edge_matrix'] = edge_matrix
			roidb[i]['sub_idx'] = sub_idx
			roidb[i]['obj_idx'] = obj_idx

			pred_pool5_path = os.path.join(full_path, 'pred_pool5', os.path.basename(roidb[i]['image'])+'.npy')
			pred_fc7_path = os.path.join(full_path, 'pred_fc7', os.path.basename(roidb[i]['image'])+'.npy')
			uni_fc7_path = os.path.join(full_path, 'uni_fc7', os.path.basename(roidb[i]['image'])+'.npy')
			img_fc7_path = os.path.join(full_path, 'fc7', os.path.basename(roidb[i]['image'])+'.npy')
			img_pool5_path = os.path.join(full_path, 'pool5', os.path.basename(roidb[i]['image'])+'.npy')

			np.save(uni_fc7_path, uni_fc7)

			roidb[i]['pred_pool5'] = pred_pool5_path
			roidb[i]['pred_fc7'] = pred_fc7_path
			roidb[i]['uni_fc7'] = uni_fc7_path
			roidb[i]['img_fc7'] = img_fc7_path
			roidb[i]['img_pool5'] = img_pool5_path
			pbar.update(1)
		new_roidb[name] = roidb
		print("nodes: {0} | max: {1} | mean: {2} | min: {3}".format(name, np.max(nodes), np.mean(nodes), np.min(nodes)))
		print("edges: {0} | max: {1} | mean: {2} | min: {3}".format(name, np.max(edges), np.mean(edges), np.min(edges)))
		# train | max: 21 | mean: 6.95423280423 | min: 1
		# test | max: 20 | mean: 7.00838574423 | min: 2
		edge_total.append(edges)
		node_total.append(nodes)
	np.savez('/home/p_zhuzy/p_zhu/NMP/data/{}_rela_graph_roidb.npz'.format(dataset), new_roidb)
	pbar.close()
	return roidb
# AttentionBasedThreshold 网络
class AttentionBasedThreshold(nn.Module):
    def __init__(self, feature_dim):
        super(AttentionBasedThreshold, self).__init__()
        self.attn_fc = nn.Linear(feature_dim * 2, 1)  # 输入是两个特征的拼接
        self.sigmoid = nn.Sigmoid()

    def forward(self, sub_fc7, ob_fc7):
        pair_features = torch.cat([sub_fc7, ob_fc7], dim=-1)
        attn_weights = self.attn_fc(pair_features)
        attn_weights = self.sigmoid(attn_weights)
        
        # 动态计算阈值
        dist_thresh = attn_weights * 0.5  
        iou_thresh = attn_weights * 0.5 
        return dist_thresh, iou_thresh

    
def rela_write_feat_into_roidb_ious(save_path, train_roidb, test_roidb, dataset='vrd', edges_types=70, dist_thresh=5.0, iou_thresh=0.45):
    edge_total = []
    node_total = []
    new_roidb = {}
    pbar = tqdm(total=len(test_roidb)+len(train_roidb))
    
    
    for name, roidb in zip(['train', 'test'], [train_roidb, test_roidb]):
        full_path = save_path + '/' + name
        feat_name = ['pred_pool5', 'pred_fc7', 'pool5', 'fc7', 'sub_fc7', 'ob_fc7']
        check_path_exists(full_path+'/uni_fc7')
        rela = np.zeros(len(roidb), dtype=np.int64)
        edges = []
        nodes = []
        
        for i in range(len(roidb)):
            #====== feats ============#
            sub_fc7 = np.load(os.path.join(full_path, 'sub_fc7', os.path.basename(roidb[i]['image'])+'.npy'))
            ob_fc7 = np.load(os.path.join(full_path, 'ob_fc7', os.path.basename(roidb[i]['image'])+'.npy'))
            fc7 = np.concatenate([sub_fc7, ob_fc7], 0)

            box_gt = np.concatenate([roidb[i]['sub_box_dete'], roidb[i]['obj_box_dete']], 0)
            cls_gt = np.concatenate([roidb[i]['sub_dete'], roidb[i]['obj_dete']], 0)

            uni_box_gt, uni_cls_gt, uni_fc7 = unique_gt(box_gt, cls_gt, fc7)
            sub_idx = new_id(uni_box_gt, roidb[i]['sub_box_dete'], roidb[i]['sub_dete'])
            obj_idx = new_id(uni_box_gt, roidb[i]['obj_box_dete'], roidb[i]['obj_dete'])
            
            # 初始化边矩阵
            edge_matrix = np.zeros([len(uni_cls_gt), len(uni_cls_gt)]) + edges_types
            
            # 计算所有box之间的距离和IoU
            dist_matrix = np.zeros([len(uni_cls_gt), len(uni_cls_gt)])
            iou_matrix = np.zeros([len(uni_cls_gt), len(uni_cls_gt)])
            
            for x in range(len(uni_cls_gt)):
                for y in range(len(uni_cls_gt)):
                    dist_matrix[x][y] = compute_distance(uni_box_gt[x], uni_box_gt[y])
                    iou_matrix[x][y] = compute_iou_each(uni_box_gt[x], uni_box_gt[y])
            # 使用 AttentionBasedThreshold 动态调整 dist_thresh,iou_thresh
            sub_fc7_tensor = torch.tensor(sub_fc7).float()
            ob_fc7_tensor = torch.tensor(ob_fc7).float()

            dist_thresh, iou_thresh = attention_model(sub_fc7_tensor, ob_fc7_tensor)
            dist_thresh=dist_thresh[0].item()
            iou_thresh=iou_thresh[0].item()
            
   
            # 通过动态阈值过滤边
            for j, x, y in zip(np.array(range(len(sub_idx))), sub_idx, obj_idx):
                x_int = int(x)
                y_int = int(y)
                # 使用计算的动态阈值
                if dist_matrix[x_int][y_int] < dist_thresh or iou_matrix[x_int][y_int] > iou_thresh:
                    edge_matrix[x_int][y_int] = roidb[i]['rela_dete'][j]
                else:
                    edge_matrix[x_int][y_int] = edges_types
                    
            # # 应用过滤条件
            # for j, x, y in zip(np.array(range(len(sub_idx))), sub_idx, obj_idx):
            #     x_int = int(x)
            #     y_int = int(y)
            #     # 只有当距离 < t1 或 IoU > t2 时才保留边
            #     if dist_matrix[x_int][y_int] < dist_thresh or iou_matrix[x_int][y_int] > iou_thresh:
            #         edge_matrix[x_int][y_int] = roidb[i]['rela_dete'][j]
            #     else:
            #         edge_matrix[x_int][y_int] = edges_types  # 不满足条件则设为无效边

            nodes.append(len(uni_cls_gt))
            edges.append(np.sum(edge_matrix < edges_types))  # 只计算有效的边
            
            roidb[i]['uni_box_gt'] = uni_box_gt
            roidb[i]['uni_gt'] = uni_cls_gt
            roidb[i]['edge_matrix'] = edge_matrix
            roidb[i]['sub_idx'] = sub_idx
            roidb[i]['obj_idx'] = obj_idx

            pred_pool5_path = os.path.join(full_path, 'pred_pool5', os.path.basename(roidb[i]['image'])+'.npy')
            pred_fc7_path = os.path.join(full_path, 'pred_fc7', os.path.basename(roidb[i]['image'])+'.npy')
            uni_fc7_path = os.path.join(full_path, 'uni_fc7', os.path.basename(roidb[i]['image'])+'.npy')
            img_fc7_path = os.path.join(full_path, 'fc7', os.path.basename(roidb[i]['image'])+'.npy')
            img_pool5_path = os.path.join(full_path, 'pool5', os.path.basename(roidb[i]['image'])+'.npy')

            np.save(uni_fc7_path, uni_fc7)

            roidb[i]['pred_pool5'] = pred_pool5_path
            roidb[i]['pred_fc7'] = pred_fc7_path
            roidb[i]['uni_fc7'] = uni_fc7_path
            roidb[i]['img_fc7'] = img_fc7_path
            roidb[i]['img_pool5'] = img_pool5_path
            pbar.update(1)
        
        new_roidb[name] = roidb
        print("nodes: {0} | max: {1} | mean: {2} | min: {3}".format(name, np.max(nodes), np.mean(nodes), np.min(nodes)))
        print("edges: {0} | max: {1} | mean: {2} | min: {3}".format(name, np.max(edges), np.mean(edges), np.min(edges)))
        
        edge_total.append(edges)
        node_total.append(nodes)
    
    # 生成带有阈值参数的文件名
    output_filename = '/home/p_zhuzy/p_zhu/NMP/data/{}_rela_graph_roidb_iou_dis_{}_{}.npz'.format(
        dataset, str(dist_thresh).replace('.', '_'), str(iou_thresh).replace('.', '_'))
    np.savez(output_filename, new_roidb)
    pbar.close()
    return roidb

def process_vrd_pred_instance_data(save_path):
	'''
	function: Build the source data for instance-level training
	node feature :[num_instance, 4096+300]
	edge label: [num_instance, 4096+300]
	'''
	data_dir = '../data'
	save_dir = save_path

	predicates_vec = np.load(os.path.join(data_dir, 'predicates_vec.npy'))
	objects_vec = np.load(os.path.join(data_dir, 'objects_vec.npy'))
	roidb_read = read_roidb(os.path.join(save_dir, 'graph_roidb.npz'))
	train_roidb = roidb_read['train']
	test_roidb = roidb_read['test']
	N_train = len(train_roidb)
	N_test = len(test_roidb)
	pbar = tqdm(total=N_train+N_test+N_test)

	def initial(N, roidb):
		sub_nodes = []
		obj_nodes = []
		edges = []
		for i in range(N):
			roidb_use = roidb[i]
			uni_box = roidb_use['uni_box_gt']
			sub_idx = box_id(roidb_use['sub_box_gt'], uni_box)
			obj_idx = box_id(roidb_use['obj_box_gt'], uni_box)

			nodes_feat = np.load(roidb_use['uni_fc7'])
			sub_feat = list(map(lambda x: nodes_feat[int(x)], sub_idx))
			sub_feat = np.reshape(np.array(sub_feat), [-1, 4096])
			obj_feat = list(map(lambda x: nodes_feat[int(x)], obj_idx))
			obj_feat = np.reshape(np.array(obj_feat), [-1, 4096])

			sub_sem = list(map(lambda x: objects_vec[int(x)], roidb_use['sub_gt']))
			sub_sem = np.reshape(np.array(sub_sem),[-1, 300])

			obj_sem = list(map(lambda x: objects_vec[int(x)], roidb_use['obj_gt']))
			obj_sem = np.reshape(np.array(obj_sem),[-1, 300])

			edge = roidb_use['rela_gt']
			
			sub_node = np.concatenate([sub_feat, sub_sem], 1)
			obj_node = np.concatenate([obj_feat, obj_sem], 1)

			sub_nodes.append(sub_node)
			obj_nodes.append(obj_node)
			edges.append(edge)
			pbar.update(1)
		sub_nodes = np.concatenate(sub_nodes, 0)
		obj_nodes = np.concatenate(obj_nodes, 0)
		edges = np.concatenate(edges, 0)
		assert sub_nodes.shape[0] == edges.shape[0]
		return sub_nodes, obj_nodes, edges

	sub_nodes_train, obj_nodes_train, edges_train = initial(N_train, train_roidb)
	sub_nodes_val, obj_nodes_val, edges_val = initial(N_test, test_roidb)
	sub_nodes_test, obj_nodes_test, edges_test = initial(N_test, test_roidb)
	pbar.close()

	np.save(os.path.join(save_dir, 'instance_sub_nodes_train'), sub_nodes_train)
	np.save(os.path.join(save_dir, 'instance_obj_nodes_train'), obj_nodes_train)
	np.save(os.path.join(save_dir, 'instance_edges_train'), edges_train)
	np.save(os.path.join(save_dir, 'instance_sub_nodes_val'), sub_nodes_val)
	np.save(os.path.join(save_dir, 'instance_obj_nodes_val'), obj_nodes_val)
	np.save(os.path.join(save_dir, 'instance_edges_val'), edges_val)
	np.save(os.path.join(save_dir, 'instance_sub_nodes_test'), sub_nodes_test)
	np.save(os.path.join(save_dir, 'instance_obj_nodes_test'), obj_nodes_test)
	np.save(os.path.join(save_dir, 'instance_edges_test'), edges_test)
	return

def process_vrd_rela_instance_data(save_path):
	'''
	function: Build the source data for instance-level training
	node feature :[num_instance, 4096+300]
	edge label: [num_instance, 4096+300]
	'''
	data_dir = '../data'
	save_dir = save_path

	predicates_vec = np.load(os.path.join(data_dir, 'predicates_vec.npy'))
	objects_vec = np.load(os.path.join(data_dir, 'objects_vec.npy'))
	roidb_read = read_roidb(os.path.join(save_dir, 'graph_roidb.npz'))
	train_roidb = roidb_read['train']
	test_roidb = roidb_read['test']
	# ipdb.set_trace()
	N_train = len(train_roidb)
	N_test = len(test_roidb)
	pbar = tqdm(total=N_train+N_test+N_test)
	def initial(N, roidb):
		sub_nodes = []
		obj_nodes = []
		edges = []
		for i in range(N):
			roidb_use = roidb[i]
			uni_box = roidb_use['uni_box_gt']
			sub_idx = box_id(roidb_use['sub_box_dete'], uni_box)
			obj_idx = box_id(roidb_use['obj_box_dete'], uni_box)

			nodes_feat = np.load(roidb_use['uni_fc7'])
			sub_feat = list(map(lambda x: nodes_feat[int(x)], sub_idx))
			sub_feat = np.reshape(np.array(sub_feat), [-1, 4096])
			obj_feat = list(map(lambda x: nodes_feat[int(x)], obj_idx))
			obj_feat = np.reshape(np.array(obj_feat), [-1, 4096])

			# sub_sem = list(map(lambda x: objects_vec[int(x)-1], roidb_use['sub_dete']))
			sub_sem = list(map(lambda x: objects_vec[int(x)], roidb_use['sub_dete']))
			sub_sem = np.reshape(np.array(sub_sem),[-1, 300])

			# obj_sem = list(map(lambda x: objects_vec[int(x)-1], roidb_use['obj_dete']))
			obj_sem = list(map(lambda x: objects_vec[int(x)], roidb_use['obj_dete']))
			obj_sem = np.reshape(np.array(obj_sem),[-1, 300])

			edge = roidb_use['rela_dete']
			
			sub_node = np.concatenate([sub_feat, sub_sem], 1)
			obj_node = np.concatenate([obj_feat, obj_sem], 1)

			sub_nodes.append(sub_node)
			obj_nodes.append(obj_node)
			edges.append(edge)
			pbar.update(1)
		sub_nodes = np.concatenate(sub_nodes, 0)
		obj_nodes = np.concatenate(obj_nodes, 0)
		edges = np.concatenate(edges, 0)
		assert sub_nodes.shape[0] == edges.shape[0]
		return sub_nodes, obj_nodes, edges

	# sub_nodes_train, obj_nodes_train, edges_train = initial(N_train, train_roidb)
	sub_nodes_val, obj_nodes_val, edges_val = initial(N_test, test_roidb)
	sub_nodes_test, obj_nodes_test, edges_test = initial(N_test, test_roidb)
	pbar.close()

	# np.save(os.path.join(save_dir, 'instance_sub_nodes_train'), sub_nodes_train)
	# np.save(os.path.join(save_dir, 'instance_obj_nodes_train'), obj_nodes_train)
	# np.save(os.path.join(save_dir, 'instance_edges_train'), edges_train)
	np.save(os.path.join(save_dir, 'instance_sub_nodes_val'), sub_nodes_val)
	np.save(os.path.join(save_dir, 'instance_obj_nodes_val'), obj_nodes_val)
	np.save(os.path.join(save_dir, 'instance_edges_val'), edges_val)
	np.save(os.path.join(save_dir, 'instance_sub_nodes_test'), sub_nodes_test)
	np.save(os.path.join(save_dir, 'instance_obj_nodes_test'), obj_nodes_test)
	np.save(os.path.join(save_dir, 'instance_edges_test'), edges_test)
	return

def get_path(dataset = 'vg', data_type = 'rela', use_ori_vgg=False):
	base_path = '/home/p_zhuzy/p_zhu/NMP/data/feat/vg_rela_vgg_feats/'
	if dataset == 'vrd' and data_type == 'pred':
		# ---------- vrd pred dataset ---------------#
		if use_ori_vgg:
			save_path = base_path + 'ori_vrd_vgg_feats'
		elif use_random_vgg:
			save_path = base_path + 'random_vrd_vgg_feats'
		else:
			save_path = base_path + 'vrd_vgg_feats'
		roidb_path = '/home/p_zhuzy/p_zhu/NMP/data/vrd_roidb.npz'
	elif dataset == 'vrd' and data_type == 'rela':
		if use_ori_vgg:
			save_path = base_path + 'ori_vrd_rela_vgg_feats'
		elif use_random_vgg:
			save_path = base_path + 'random_vrd_rela_vgg_feats'
		else:
			save_path = base_path + 'vrd_rela_vgg_feats'
		roidb_path = '/home/p_zhuzy/p_zhu/NMP/data/vrd_rela_roidb.npz'
	elif dataset == 'vg' and data_type == 'pred':
		# ----------- vg dataset ---------------#
		save_path = base_path + 'vg_vgg_feats'
		roidb_path = '../data/vg_roidb.npz'
	elif dataset == 'vg' and data_type == 'rela':
		# ----------- vg rela dataset ---------------#
		save_path = base_path + 'vg_rela_vgg_feats'
		roidb_path = '../data/vg_rela_roidb.npz'
	return save_path, roidb_path

save_path, roidb_path = get_path(dataset, data_type, use_ori_vgg)
# ============== vrd pred ==============#
# # -------- read data --------------#
roidb_read = read_roidb(roidb_path)
train_roidb = roidb_read['train_roidb']
test_roidb = roidb_read['test_roidb']

# nodes: train | max: 21 | mean: 6.95423280423 | min: 1
# edges: train | max: 34 | mean: 8.03042328042 | min: 1
# nodes: test | max: 20 | mean: 7.00838574423 | min: 2
# edges: test | max: 41 | mean: 8.0 | min: 1

# ----- dsr --------#
# nodes: train | max: 21 | mean: 6.95423280423 | min: 1
# edges: train | max: 30 | mean: 7.89867724868 | min: 1
# nodes: test | max: 20 | mean: 7.00838574423 | min: 2
# edges: test | max: 23 | mean: 7.82809224319 | min: 1

if dataset == 'vrd' and data_type == 'pred':
	pred_write_feat_into_roidb(save_path, train_roidb, test_roidb, dataset='vrd', edges_types=70)
	process_vrd_pred_instance_data(save_path)

# ============== vrd rela ==============#
# # -------- read data --------------#
# roidb_read = read_roidb(roidb_path)
# train_roidb = roidb_read['train_roidb']
# test_roidb = roidb_read['test_roidb']
# # # ipdb.set_trace()
# # # nodes: train | max: 44 | mean: 14 | min: 1
# # # edges: train | max: 34 | mean: 8 | min: 1
# # # nodes: test | max: 96 | mean: 39.9381551363 | min: 9
# # # edges: test | max: 41 | mean: 8.0 | min: 1
# # dsr test rela
# # nodes: test | max: 63 | mean: 8.071278826 | min: 2
# # edges: test | max: 23 | mean: 8.0 | min: 1
if dataset == 'vrd' and data_type == 'rela':
	#rela_write_feat_into_roidb(save_path, train_roidb, test_roidb, dataset='vrd', edges_types=70)
    # 新版本调用（带过滤条件）
    # rela_write_feat_into_roidb_ious(
    #     save_path=save_path,
    #     train_roidb=train_roidb,
    #     test_roidb=test_roidb,
    #     dataset='vrd',
    #     edges_types=70,
    #     dist_thresh=5.0,   # 距离阈值（可根据需要调整）
    #     iou_thresh=0.45    # IoU阈值（0-1之间，可根据需要调整）
    # )	
    rela_write_feat_into_roidb_ious(
    save_path=save_path,
    train_roidb=train_roidb,
    test_roidb=test_roidb,
    dataset='vrd',
    edges_types=70
    # 不需要传递 dist_thresh 和 iou_thresh，这些会在函数内部动态计算
	)
 # process_vrd_rela_instance_data(save_path)

# ============== vg pred ==============#
# save_path, roidb_path = get_path('vg', 'pred')
# # -------- read data --------------#
# roidb_read = read_roidb(roidb_path)
# train_roidb = roidb_read['train_roidb']
# test_roidb = roidb_read['test_roidb']

# # nodes: train | max: 98 | mean: 12.9205761986 | min: 1
# # edges: train | max: 490 | mean: 10.8853836355 | min: 1
# # nodes: test | max: 110 | mean: 13.1718230335 | min: 1
# # edges: test | max: 352 | mean: 11.0894500735 | min: 1

if dataset == 'vg' and data_type == 'pred':
	pred_write_feat_into_roidb(save_path, train_roidb, test_roidb, dataset='vg', edges_types=100)
# pred_save_vgg_feat(save_path, train_roidb, test_roidb)
# pred_write_feat_into_roidb(save_path, train_roidb, test_roidb, edges_types=100)

# # ============== vg rela ==============#
# save_path, roidb_path = get_path('vg', 'rela')
# # -------- read data --------------#
# roidb_read = read_roidb(roidb_path)
# train_roidb = roidb_read['train_roidb']
# test_roidb = roidb_read['test_roidb']

# # nodes: train | max: 72 | mean: 16.3680922568 | min: 1
# # edges: train | max: 490 | mean: 10.8853836355 | min: 1
# # nodes: test | max: 90 | mean: 28.3761698507 | min: 2
# # edges: test | max: 352 | mean: 11.0894500735 | min: 1


# rela_save_vgg_feat(save_path, train_roidb, test_roidb)
# rela_write_feat_into_roidb(save_path, train_roidb, test_roidb, edges_types=100)
