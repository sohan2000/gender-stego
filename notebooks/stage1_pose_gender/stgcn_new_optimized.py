"""GaitSTGCN model, dataset and collate code, copied verbatim from the 30k-sample
training run in stgcn_final_code.ipynb (the run that produced 72.65% val accuracy).
Imported by Model_Arch.ipynb and Evaluate_GaitSTGCN.ipynb. Do not edit: it must stay
identical to the code the checkpoint was trained with.
"""
"""
OPTIMIZED ST-GCN Pipeline for Fast Results
Key optimizations:
1. Uses annotations_present.csv (only existing files)
2. Subset training for quick results
3. Mixed precision training (2x speedup)
4. Efficient data loading
5. Progress tracking
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split, Subset
from tqdm import tqdm
import time

# ============================================================================
# GRAPH & MODEL (Your existing code - unchanged)
# ============================================================================

class Graph:
    """Skeleton graph structure for human body (18 joints OpenPose format)"""
    def __init__(self, num_node=18, layout='openpose'):
        self.num_node = num_node
        self.layout = layout
        self.get_edge()
        self.get_adjacency()

    def get_edge(self):
        """Define skeleton connectivity (bone connections)"""
        self.self_link = [(i, i) for i in range(self.num_node)]

        if self.num_node == 18:
            self.neighbor_link = [
                (0, 1), (1, 2), (2, 3), (3, 4), (1, 5), (5, 6), (6, 7),
                (1, 8), (8, 9), (9, 10), (10, 11), (8, 12), (12, 13),
                (13, 14), (0, 15), (15, 17), (0, 16), (16, 18),
            ]
        else:
            self.neighbor_link = [(i, i+1) for i in range(self.num_node-1)]

        self.edge = self.self_link + self.neighbor_link
        self.center = 1

    def get_adjacency(self):
        """Build normalized adjacency matrix"""
        A = np.zeros((self.num_node, self.num_node))
        for i, j in self.edge:
            if i < self.num_node and j < self.num_node:
                A[i, j] = 1
                A[j, i] = 1

        A_with_self = A + np.eye(self.num_node)
        D = np.sum(A_with_self, axis=1)
        D_inv_sqrt = np.power(D, -0.5)
        D_inv_sqrt[np.isinf(D_inv_sqrt)] = 0
        D_mat = np.diag(D_inv_sqrt)
        A_normalized = D_mat @ A_with_self @ D_mat

        self.A = np.stack([np.eye(self.num_node), A_normalized, A_normalized])


class ConvTemporalGraphical(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv2d(in_channels, out_channels * kernel_size, kernel_size=1)

    def forward(self, x, A):
        x = self.conv(x)
        n, kc, t, v = x.size()
        x = x.view(n, self.kernel_size, kc // self.kernel_size, t, v)
        x = torch.einsum('nkctv,kvw->nctw', (x, A))
        return x.contiguous(), A


class STGCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 dropout=0.2, residual=True):
        super().__init__()

        assert len(kernel_size) == 2
        assert kernel_size[0] % 2 == 1
        padding = ((kernel_size[0] - 1) // 2, 0)

        self.gcn = ConvTemporalGraphical(in_channels, out_channels, kernel_size[1])

        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, (kernel_size[0], 1),
                     (stride, 1), padding),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout, inplace=True),
        )

        if not residual:
            self.residual = lambda x: 0
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=(stride, 1)),
                nn.BatchNorm2d(out_channels),
            )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, A):
        res = self.residual(x)
        x, A = self.gcn(x, A)
        x = self.tcn(x) + res
        return self.relu(x), A


class GaitSTGCN(nn.Module):
    def __init__(self, in_channels=3, num_class=1, num_joints=18,
                 edge_importance_weighting=True, dropout=0.3):
        super().__init__()

        self.graph = Graph(num_node=num_joints)
        A = torch.tensor(self.graph.A, dtype=torch.float32, requires_grad=False)
        self.register_buffer('A', A)

        spatial_kernel_size = A.size(0)
        temporal_kernel_size = 9
        kernel_size = (temporal_kernel_size, spatial_kernel_size)

        self.data_bn = nn.BatchNorm1d(in_channels * A.size(1))

        self.st_gcn_networks = nn.ModuleList([
            STGCNBlock(in_channels, 64, kernel_size, 1, residual=False, dropout=0),
            STGCNBlock(64, 64, kernel_size, 1, dropout=dropout),
            STGCNBlock(64, 64, kernel_size, 1, dropout=dropout),
            STGCNBlock(64, 128, kernel_size, 2, dropout=dropout),
            STGCNBlock(128, 128, kernel_size, 1, dropout=dropout),
            STGCNBlock(128, 256, kernel_size, 2, dropout=dropout),
            STGCNBlock(256, 256, kernel_size, 1, dropout=dropout),
        ])

        if edge_importance_weighting:
            self.edge_importance = nn.Parameter(
                torch.ones(len(self.st_gcn_networks), *A.size())
            )
        else:
            self.edge_importance = [1] * len(self.st_gcn_networks)

        self.fc = nn.Linear(256, num_class)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        N, C, T, V, M = x.size()

        x = x.permute(0, 4, 3, 1, 2).contiguous()
        x = x.view(N * M, V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T)
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        x = x.view(N * M, C, T, V)

        for gcn, importance in zip(self.st_gcn_networks, self.edge_importance):
            x, _ = gcn(x, self.A * importance)

        x = F.avg_pool2d(x, x.size()[2:])
        x = x.view(N, M, -1, 1, 1).mean(dim=1)
        x = x.view(N, -1)

        # Return raw logits (no sigmoid)
        # BCEWithLogitsLoss will apply sigmoid internally
        x = self.fc(x)
        return x

# ============================================================================
# OPTIMIZED PREPROCESSING
# ============================================================================

def detect_and_convert_format(data, target_V=18, target_C=3):
    shape = data.shape

    if len(shape) == 3:
        T, V, C = shape
        if V == target_V and C == target_C:
            return data
        elif V == target_V and C == 2 and target_C == 3:
            confidence = np.ones((T, V, 1), dtype=data.dtype)
            return np.concatenate([data, confidence], axis=2)
        flat = data.reshape(T, -1)
        return flat.reshape(T, target_V, target_C)

    elif len(shape) == 2:
        T, VC = shape
        if VC == target_V * target_C:
            return data.reshape(T, target_V, target_C)
        elif VC == target_V * 2 and target_C == 3:
            data_2d = data.reshape(T, target_V, 2)
            confidence = np.ones((T, target_V, 1), dtype=data.dtype)
            return np.concatenate([data_2d, confidence], axis=2)
        else:
            raise ValueError(f"Cannot convert shape {shape} to ({T}, {target_V}, {target_C})")
    else:
        raise ValueError(f"Unexpected data shape: {shape}")


def normalize_pose(data, method='center'):
    if method == 'center':
        center_idx = 1 if data.shape[1] >= 18 else 0
        center = data[:, center_idx:center_idx+1, :2]
        data_norm = data.copy()
        data_norm[:, :, :2] = data[:, :, :2] - center
        return data_norm
    return data


def handle_missing_values(data):
    for v in range(data.shape[1]):
        for c in range(data.shape[2]):
            series = data[:, v, c]
            nonzero = np.where(series != 0)[0]

            if len(nonzero) > 1:
                zero_indices = np.where(series == 0)[0]
                if len(zero_indices) > 0:
                    data[zero_indices, v, c] = np.interp(
                        zero_indices, nonzero, series[nonzero]
                    )
    return data

# ============================================================================
# OPTIMIZED DATASET
# ============================================================================

class GaitDatasetSTGCN(Dataset):
    def __init__(self, annotations_path, poses_dir, num_joints=18, num_coords=3,
                 normalize=True, handle_missing=True, skip_validation=True):
        self.annotations = pd.read_csv(annotations_path)
        self.poses_dir = poses_dir
        self.num_joints = num_joints
        self.num_coords = num_coords
        self.normalize = normalize
        self.handle_missing = handle_missing
        self.corrupted_indices = set()
        self.skip_validation = skip_validation

        self.file_names = self.annotations['file_name']
        self.labels = (self.annotations['Female'] > 0.5).astype(int).values

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        # Skip if we know this index is corrupted
        if idx in self.corrupted_indices:
            return self.__getitem__((idx + 1) % len(self))

        try:
            pose_path = os.path.join(self.poses_dir, self.file_names.iloc[idx])
            data = np.load(pose_path)

            # Validate data is not empty
            if data.size == 0:
                raise ValueError("Empty file")

            data = detect_and_convert_format(data, self.num_joints, self.num_coords)

            if self.handle_missing:
                data = handle_missing_values(data)
            if self.normalize:
                data = normalize_pose(data, method='center')

            pose_sequence = torch.from_numpy(data).float()
            pose_sequence = pose_sequence.permute(2, 0, 1).unsqueeze(-1)
            label = torch.tensor(self.labels[idx]).float()

            return pose_sequence, label

        except Exception as e:
            # Mark as corrupted and skip
            self.corrupted_indices.add(idx)
            # Return next valid sample
            return self.__getitem__((idx + 1) % len(self))


def collate_fn_stgcn(batch, max_seq_length=300):
    """Collate function that pads temporal dimension and truncates long sequences"""
    sequences, labels = zip(*batch)

    # Truncate sequences that are too long (memory safety)
    sequences = [seq[:, :max_seq_length, :, :] if seq.shape[1] > max_seq_length else seq
                 for seq in sequences]

    max_T = max(seq.shape[1] for seq in sequences)

    padded_sequences = []
    lengths = []

    for seq in sequences:
        C, T, V, M = seq.shape
        lengths.append(T)

        if T < max_T:
            padding = torch.zeros(C, max_T - T, V, M)
            seq = torch.cat([seq, padding], dim=1)

        padded_sequences.append(seq)

    padded_sequences = torch.stack(padded_sequences, dim=0)
    labels = torch.stack(labels, dim=0)

    return padded_sequences, labels, torch.tensor(lengths)

# ============================================================================
# OPTIMIZED TRAINING FUNCTIONS
# ============================================================================

def train_epoch(model, dataloader, criterion, optimizer, device, scaler, use_amp=True, clear_cache_interval=25):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(dataloader, desc="Training", leave=False)
    for batch_idx, (poses, labels, lengths) in enumerate(pbar):
        poses = poses.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()

        if use_amp:
            with torch.amp.autocast('cuda'):
                outputs = model(poses)
                loss = criterion(outputs.squeeze(), labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(poses)
            loss = criterion(outputs.squeeze(), labels)
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        predicted = (torch.sigmoid(outputs.squeeze()) > 0.5).float()
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

        # Clear cache periodically to prevent fragmentation
        if (batch_idx + 1) % clear_cache_interval == 0:
            torch.cuda.empty_cache()

        pbar.set_postfix({
            'loss': f'{total_loss/(batch_idx+1):.4f}',
            'acc': f'{100*correct/total:.2f}%'
        })

    return total_loss / len(dataloader), 100 * correct / total


def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for poses, labels, lengths in tqdm(dataloader, desc="Validating", leave=False):
            poses = poses.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(poses)
            loss = criterion(outputs.squeeze(), labels)

            total_loss += loss.item()
            # Apply sigmoid for predictions since model returns logits
            predicted = (torch.sigmoid(outputs.squeeze()) > 0.5).float()
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    return total_loss / len(dataloader), 100 * correct / total
