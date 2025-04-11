import os
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim
import psutil
import tempfile

def print_memory_usage():
    """打印当前内存使用情况"""
    process = psutil.Process(os.getpid())
    print(f"Memory usage: {process.memory_info().rss / 1024 / 1024:.2f} MB")

# === Attention 模型 ===
class AttentionBasedThreshold(nn.Module):
    def __init__(self, feature_dim):
        super(AttentionBasedThreshold, self).__init__()
        self.attn_fc = nn.Linear(feature_dim * 2, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, sub_fc7, ob_fc7):
        pair_features = torch.cat([sub_fc7, ob_fc7], dim=-1)
        attn_weights = self.attn_fc(pair_features)
        attn_weights = self.sigmoid(attn_weights)
        dist_thresh = attn_weights * 0.5
        iou_thresh = attn_weights * 0.5
        return dist_thresh, iou_thresh

# === 数据提取函数 ===
def load_attention_training_data(npz_path, feature_base_path, edges_types=70):
    """使用memmap加载大数据，避免内存不足"""
    print("Loading data...")
    print_memory_usage()
    
    data = np.load(npz_path, allow_pickle=True, encoding='latin1')
    roidb_dict = data['arr_0'].item()
    
    # 先统计总样本数
    total_samples = 0
    for split in ['train', 'test']:
        for entry in roidb_dict[split]:
            total_samples += len(entry['rela_dete'])
    print(f"Total samples: {total_samples}")
    
    # 使用临时文件避免磁盘残留
    with tempfile.NamedTemporaryFile() as sub_file, \
         tempfile.NamedTemporaryFile() as ob_file, \
         tempfile.NamedTemporaryFile() as label_file:
        
        # 创建memmap文件
        sub_feats = np.memmap(sub_file.name, dtype='float32', mode='w+', shape=(total_samples, 4096))
        ob_feats = np.memmap(ob_file.name, dtype='float32', mode='w+', shape=(total_samples, 4096))
        labels = np.memmap(label_file.name, dtype='float32', mode='w+', shape=(total_samples,))
        
        idx = 0
        for split in ['train', 'test']:
            roidb = roidb_dict[split]
            for entry in tqdm(roidb, desc=f'Processing {split}'):
                img_id = os.path.basename(entry['image']) + '.npy'
                sub_feat_path = os.path.join(feature_base_path, split, 'sub_fc7', img_id)
                ob_feat_path = os.path.join(feature_base_path, split, 'ob_fc7', img_id)

                if not (os.path.exists(sub_feat_path) and os.path.exists(ob_feat_path)):
                    continue

                sub_fc7 = np.load(sub_feat_path)
                ob_fc7 = np.load(ob_feat_path)

                for i in range(len(entry['rela_dete'])):
                    sub_feats[idx] = sub_fc7[i]
                    ob_feats[idx] = ob_fc7[i]
                    labels[idx] = 1 if entry['rela_dete'][i] != edges_types else 0
                    idx += 1
        
        print_memory_usage()
        # 转换为普通numpy数组返回（只返回实际填充的部分）
        return np.array(sub_feats[:idx]), np.array(ob_feats[:idx]), np.array(labels[:idx])

# === 训练函数 ===
def train_attention_model(sub_feats, ob_feats, labels, feature_dim, 
                         save_path='attention_model.pth', epochs=10, batch_size=32, lr=1e-4):
    """训练函数，添加了GPU内存监控"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if torch.cuda.is_available():
        print(f"GPU Memory allocated: {torch.cuda.memory_allocated()/1024/1024:.2f} MB")
        print(f"GPU Memory cached: {torch.cuda.memory_reserved()/1024/1024:.2f} MB")

    model = AttentionBasedThreshold(feature_dim=feature_dim).to(device)
    
    # 转换为torch tensor
    dataset = TensorDataset(
        torch.tensor(sub_feats, dtype=torch.float32),
        torch.tensor(ob_feats, dtype=torch.float32),
        torch.tensor(labels, dtype=torch.float32)
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for sub_f, ob_f, label in tqdm(dataloader, desc=f'Epoch {epoch+1}/{epochs}'):
            sub_f, ob_f, label = sub_f.to(device), ob_f.to(device), label.to(device).unsqueeze(1)

            optimizer.zero_grad()
            attn_score = model.attn_fc(torch.cat([sub_f, ob_f], dim=-1))
            pred = model.sigmoid(attn_score)

            loss = criterion(pred, label)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * sub_f.size(0)
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        print(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss / len(dataset):.4f}")
        print_memory_usage()

    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

# === 主执行逻辑 ===
if __name__ == "__main__":
    npz_path = '/home/p_zhuzy/p_zhu/NMP/data/vrd_rela_graph_roidb.npz'
    feature_base_path = '/home/p_zhuzy/p_zhu/NMP/data/feat/vg_rela_vgg_feats/vrd_rela_vgg_feats'

    print("Loading training data...")
    try:
        sub_feats, ob_feats, labels = load_attention_training_data(npz_path, feature_base_path)
        print(f"Loaded data shapes - sub: {sub_feats.shape}, ob: {ob_feats.shape}, labels: {labels.shape}")
        
        print("Training attention model...")
        train_attention_model(
            sub_feats=sub_feats,
            ob_feats=ob_feats,
            labels=labels,
            feature_dim=4096,
            save_path='attention_model.pth',
            epochs=10,
            batch_size=32  # 减小batch size防止内存不足
        )
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        import traceback
        traceback.print_exc()