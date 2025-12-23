# -*- coding: UTF-8 -*-

import torch
import numpy as np
from torch import nn
import torch.nn.functional as F

from models.BaseModel import GeneralModel
from models.layers import ResidualVectorQuantizer, ProductVectorQuantizer, LightGCNConv
from utils.graph_utils import build_adj_matrix, gcn_norm


class CoGCL(GeneralModel):
    reader = 'BaseReader'
    runner = 'BaseRunner'

    @staticmethod
    def parse_model_args(parser):
        parser.add_argument('--embedding_size', type=int, default=64,
                            help='Size of embedding vectors.')
        parser.add_argument('--n_layers', type=int, default=2,
                            help='Number of LightGCN layers.')
        parser.add_argument('--user_code_num', type=int, default=4)
        parser.add_argument('--item_code_num', type=int, default=4)
        parser.add_argument('--user_code_size', type=int, default=256)
        parser.add_argument('--item_code_size', type=int, default=256)
        parser.add_argument('--code_dist', type=str, default='cos')
        parser.add_argument('--code_dist_tau', type=float, default=0.2)
        parser.add_argument('--code_batch_size', type=int, default=2048)
        parser.add_argument('--vq_loss_weight', type=float, default=0.1)
        parser.add_argument('--vq_type', type=str, default='rq')
        parser.add_argument('--vq_ema', type=int, default=0)
        parser.add_argument('--cl_weight', type=float, default=0.1)
        parser.add_argument('--sim_cl_weight', type=float, default=0.1)
        parser.add_argument('--cl_tau', type=float, default=0.2)
        parser.add_argument('--graph_replace_p', type=float, default=0.1)
        parser.add_argument('--graph_add_p', type=float, default=0.1)
        parser.add_argument('--drop_p', type=float, default=0.1)
        parser.add_argument('--drop_fwd', type=int, default=1)
        parser.add_argument('--data_aug_delay', type=int, default=0)
        parser.add_argument('--code_mix_alpha', type=float, default=0.0,
                            help='[QER] 量化增强表示融合系数。0=禁用，>0 时将量化嵌入融合到 GCN 输出。')
        return GeneralModel.parse_model_args(parser)

    def __init__(self, args, corpus):
        super().__init__(args, corpus)

        self.embedding_size = args.embedding_size
        self.n_layers = args.n_layers
        self.user_code_num = args.user_code_num
        self.item_code_num = args.item_code_num
        self.user_code_size = args.user_code_size
        self.item_code_size = args.item_code_size
        self.n_user_codes = self.user_code_num * self.user_code_size
        self.n_item_codes = self.item_code_num * self.item_code_size
        self.code_dist = args.code_dist
        self.code_dist_tau = args.code_dist_tau
        self.code_batch_size = args.code_batch_size
        self.vq_loss_weight = args.vq_loss_weight
        self.vq_type = args.vq_type
        self.vq_ema = args.vq_ema
        self.cl_weight = args.cl_weight
        self.sim_cl_weight = args.sim_cl_weight
        self.cl_tau = args.cl_tau
        self.graph_replace_p = args.graph_replace_p
        self.graph_add_p = args.graph_add_p
        self.drop_p = args.drop_p
        self.drop_fwd = args.drop_fwd
        self.data_aug_delay = args.data_aug_delay
        self.code_mix_alpha = args.code_mix_alpha  # QER 融合系数
        self.epoch_num = 0
        self.reg_weight = args.l2  # L2 regularization weight

        self._init_embeddings()
        self._init_graph(corpus)
        
        # 初始化相似度信息存储
        self.code_similar_users = None  # 基于码本的相似用户
        self.code_similar_items = None  # 基于码本的相似物品
        self.same_target_users = None   # 交互过相同物品的用户
        self.same_target_items = None   # 被相同用户交互的物品
        
        # 如果需要sim_cl_loss，预计算same_target信息
        if self.sim_cl_weight > 0:
            self._compute_same_target_info(corpus)

        if self.vq_type.lower() == 'rq':
            self.user_vq = ResidualVectorQuantizer(codebook_num=self.user_code_num, codebook_size=self.user_code_size, 
                                                   codebook_dim=self.embedding_size, dist=self.code_dist, tau=self.code_dist_tau, vq_ema=self.vq_ema)
            self.item_vq = ResidualVectorQuantizer(codebook_num=self.item_code_num, codebook_size=self.item_code_size,
                                                     codebook_dim=self.embedding_size, dist=self.code_dist, tau=self.code_dist_tau, vq_ema=self.vq_ema)
        elif self.vq_type.lower() == 'pq':
            assert self.embedding_size % self.user_code_num == 0
            assert self.embedding_size % self.item_code_num == 0
            self.user_vq = ProductVectorQuantizer(codebook_num=self.user_code_num, codebook_size=self.user_code_size,
                                                  codebook_dim=self.embedding_size//self.user_code_num, dist=self.code_dist, tau=self.code_dist_tau, vq_ema=self.vq_ema)
            self.item_vq = ProductVectorQuantizer(codebook_num=self.item_code_num, codebook_size=self.item_code_size,
                                                  codebook_dim=self.embedding_size//self.item_code_num, dist=self.code_dist, tau=self.code_dist_tau, vq_ema=self.vq_ema)
        else:
            raise NotImplementedError

        self.gcn_conv = LightGCNConv(dim=self.embedding_size)
        self.dropout = nn.Dropout(self.drop_p)

    def _init_embeddings(self):
        self.user_embedding = nn.Embedding(self.user_num, self.embedding_size)
        self.item_embedding = nn.Embedding(self.item_num, self.embedding_size)
        self.user_code_embedding = nn.Embedding(self.n_user_codes, self.embedding_size)
        self.item_code_embedding = nn.Embedding(self.n_item_codes, self.embedding_size)
        
    def _init_graph(self, corpus):
        user_ids = corpus.data_df['train']['user_id'].values
        item_ids = corpus.data_df['train']['item_id'].values
        
        # 保存交互对，用于图增强
        self._user = torch.from_numpy(user_ids).long()
        self._item = torch.from_numpy(item_ids).long()
        
        adj_mat = build_adj_matrix(user_ids, item_ids, self.user_num, self.item_num)
        self.edge_index, self.edge_weight = gcn_norm(adj_mat, num_nodes=self.user_num + self.item_num, self_loop=True)
        self.edge_index = self.edge_index.to(self.device)
        self.edge_weight = self.edge_weight.to(self.device)
    
    def _compute_same_target_info(self, corpus):
        """
        预计算same_target信息：
        - same_target_users[item_i]: 所有交互过物品i的用户
        - same_target_items[user_j]: 用户j交互过的所有物品
        """
        print("Computing same target users and items for sim_cl_loss...")
        user_ids = corpus.data_df['train']['user_id'].values
        item_ids = corpus.data_df['train']['item_id'].values
        
        # 为每个物品找到交互过它的所有用户
        same_target_users = [[] for _ in range(self.item_num)]
        for user, item in zip(user_ids, item_ids):
            same_target_users[item].append(user)
        # 去重并转为numpy数组
        same_target_users = [np.unique(users) for users in same_target_users]
        
        # 为每个用户找到交互过的所有物品
        same_target_items = [[] for _ in range(self.user_num)]
        for user, item in zip(user_ids, item_ids):
            same_target_items[user].append(item)
        # 去重并转为numpy数组
        same_target_items = [np.unique(items) for items in same_target_items]
        
        self.same_target_users = same_target_users
        self.same_target_items = same_target_items
        
        mean_user = np.mean([len(users) for users in same_target_users if len(users) > 0])
        mean_item = np.mean([len(items) for items in same_target_items if len(items) > 0])
        print(f"Mean same target users per item: {mean_user:.2f}")
        print(f"Mean same target items per user: {mean_item:.2f}")

    def get_ego_embeddings(self):
        user_embeddings = self.user_embedding.weight
        item_embeddings = self.item_embedding.weight
        ego_embeddings = torch.cat([user_embeddings, item_embeddings], dim=0)
        return ego_embeddings

    def get_w_codes_embeddings(self):
        user_embeddings = self.user_embedding.weight
        item_embeddings = self.item_embedding.weight
        user_code_embeddings = self.user_code_embedding.weight
        item_code_embeddings = self.item_code_embedding.weight
        ego_embeddings = torch.cat([user_embeddings, user_code_embeddings, item_embeddings, item_code_embeddings], dim=0)
        return ego_embeddings

    def forward(self, feed_dict):
        user_all_embeddings, item_all_embeddings = self._forward_gcn()
        u_embeddings = user_all_embeddings[feed_dict['user_id']]
        i_embeddings = item_all_embeddings[feed_dict['item_id']]
        prediction = (u_embeddings[:, None, :] * i_embeddings).sum(-1)
        return {'prediction': prediction}

    def _forward_gcn(self, drop=False):
        embeddings = self.get_ego_embeddings()
        embeddings_list = []  # CoGCL 原始设计：不包含初始嵌入

        for layer_idx in range(self.n_layers):
            if drop:
                embeddings = self.dropout(embeddings)
            embeddings = self.gcn_conv(embeddings, self.edge_index, self.edge_weight)
            embeddings_list.append(embeddings)

        all_embeddings = torch.stack(embeddings_list, dim=1)
        all_embeddings = torch.mean(all_embeddings, dim=1)

        user_all_embeddings, item_all_embeddings = torch.split(
            all_embeddings, [self.user_num, self.item_num]
        )
        
        # QER: 量化增强表示
        if self.code_mix_alpha > 0 and not drop:  # 仅在推理和主 forward 中应用
            user_all_embeddings, item_all_embeddings = self._apply_qer(
                user_all_embeddings, item_all_embeddings
            )
        
        return user_all_embeddings, item_all_embeddings
    
    def _apply_qer(self, user_embeddings, item_embeddings):
        """
        QER (Quantization-Enhanced Representation): 量化增强表示
        将量化后的码本嵌入融合到 GCN 输出中，利用全局语义信息优化局部表示。
        
        Args:
            user_embeddings: [n_users, emb_dim] 用户 GCN 嵌入
            item_embeddings: [n_items, emb_dim] 物品 GCN 嵌入
        
        Returns:
            增强后的嵌入: E_final = E_gcn + alpha * E_vq
        """
        with torch.no_grad():
            # 获取量化嵌入
            user_vq_emb, _, _ = self.user_vq(user_embeddings.detach())
            item_vq_emb, _, _ = self.item_vq(item_embeddings.detach())
        
        # 融合：E_final = E_gcn + alpha * E_vq
        user_enhanced = user_embeddings + self.code_mix_alpha * user_vq_emb
        item_enhanced = item_embeddings + self.code_mix_alpha * item_vq_emb
        
        return user_enhanced, item_enhanced

    def train(self, mode=True):
        if mode and self.epoch_num >= self.data_aug_delay:
            if self.epoch_num == self.data_aug_delay:
                print("Start Data Augmentation")
            self.graph_augment()

        if mode:
            self.epoch_num += 1
        return super().train(mode=mode)

    def graph_augment(self):
        all_user_codes, all_item_codes = self.get_all_codes()
        aug_types = np.random.choice([0, 1], size=2, replace=True)
        self.aug_edge_index_1, self.aug_edge_weight_1 = self.inter_graph_aug(all_user_codes, all_item_codes, aug_types[0])
        self.aug_edge_index_2, self.aug_edge_weight_2 = self.inter_graph_aug(all_user_codes, all_item_codes, aug_types[1])
        
        # 如果需要sim_cl_loss，计算码本相似度
        if self.sim_cl_weight > 0:
            self.code_similar_users, self.code_similar_items = self.get_share_codes_info(all_user_codes, all_item_codes)

    @torch.no_grad()
    def get_all_codes(self):
        self.user_vq.eval()
        self.item_vq.eval()
        user_all_embeddings, item_all_embeddings = self._forward_gcn()

        start, batch_size = 0, self.code_batch_size
        all_user_codes = []
        while start < self.user_num:
            batch_user_embs = user_all_embeddings[start: start + batch_size]
            _, codes = self.emb_quantize(batch_user_embs, self.user_vq)
            all_user_codes.append(codes)
            start += batch_size
        all_user_codes = torch.cat(all_user_codes, dim=0)
        user_offset = torch.arange(self.user_code_num, dtype=torch.long, device=self.device) * self.user_code_size
        all_user_codes = all_user_codes + user_offset.unsqueeze(0)

        start, batch_size = 0, self.code_batch_size
        all_item_codes = []
        while start < self.item_num:
            batch_item_embs = item_all_embeddings[start: start + batch_size]
            _, codes = self.emb_quantize(batch_item_embs, self.item_vq)
            all_item_codes.append(codes)
            start += batch_size
        all_item_codes = torch.cat(all_item_codes, dim=0)
        item_offset = torch.arange(self.item_code_num, dtype=torch.long, device=self.device) * self.item_code_size
        all_item_codes = all_item_codes + item_offset.unsqueeze(0)

        return all_user_codes, all_item_codes

    def inter_graph_aug(self, all_user_codes, all_item_codes, aug_type):
        """
        基于码本的图增强
        Args:
            all_user_codes: [n_users, user_code_num] 所有用户的码本索引
            all_item_codes: [n_items, item_code_num] 所有物品的码本索引
            aug_type: 0=replace(替换), 1=add(添加), 2=非图增强
        Returns:
            aug_edge_index, aug_edge_weight: 增强后的图
        """
        row = self._user  # 所有训练交互的用户ID
        col = self._item  # 所有训练交互的物品ID
        
        all_idx = np.arange(len(row))
        
        if aug_type == 0:  # replace: 用码本替换原始边
            aug_num = int(len(all_idx) * self.graph_replace_p)
            user_aug_idx = np.random.choice(len(row), aug_num, replace=False)
            item_aug_idx = np.random.choice(len(col), aug_num, replace=False)
            keep_idx = list(set(all_idx) - set(user_aug_idx) - set(item_aug_idx))
            
        elif aug_type == 1:  # add: 保留所有边，添加码本边
            keep_idx = all_idx
            aug_num = int(len(all_idx) * self.graph_add_p)
            user_aug_idx = np.random.choice(len(row), aug_num, replace=False)
            item_aug_idx = np.random.choice(len(col), aug_num, replace=False)
            
        else:  # aug_type == 2: 非图增强，只保留原始边
            keep_idx = all_idx
            user_aug_idx = []
            item_aug_idx = []
        
        # 保留的原始边（调整物品索引：item -> user_num + n_user_codes + item）
        keep_row = row[keep_idx]
        keep_col = col[keep_idx] + self.user_num + self.n_user_codes
        
        # 构建增强边
        aug_inter_row = []
        aug_inter_col = []
        
        if aug_type != 2:
            # 用户增强：用户的码本 -> 原始物品
            aug_row = row[user_aug_idx]
            aug_col = col[user_aug_idx]
            for user, item in zip(aug_row, aug_col):
                # 获取该物品的所有码本（调整索引）
                item_codes = all_item_codes[item] + self.user_num + self.n_user_codes + self.item_num
                item_codes = item_codes.tolist()
                # 连接：user -> 所有 item_codes
                aug_inter_row.extend([user.item()] * len(item_codes))
                aug_inter_col.extend(item_codes)
            
            # 物品增强：原始用户 -> 物品的码本
            aug_row = row[item_aug_idx]
            aug_col = col[item_aug_idx]
            for user, item in zip(aug_row, aug_col):
                # 获取该用户的所有码本（调整索引）
                user_codes = all_user_codes[user] + self.user_num
                item = item + self.user_num + self.n_user_codes
                user_codes = user_codes.tolist()
                # 连接：所有 user_codes -> item
                aug_inter_row.extend(user_codes)
                aug_inter_col.extend([item.item()] * len(user_codes))
        
        # 合并保留边和增强边
        all_row = torch.tensor(keep_row.numpy().tolist() + aug_inter_row, dtype=keep_row.dtype)
        all_col = torch.tensor(keep_col.numpy().tolist() + aug_inter_col, dtype=keep_col.dtype)
        
        # 构建双向边
        edge_index1 = torch.stack([all_row, all_col])
        edge_index2 = torch.stack([all_col, all_row])
        aug_edge_index = torch.cat([edge_index1, edge_index2], dim=1)
        aug_edge_weight = torch.ones(aug_edge_index.size(1))
        
        # 归一化（包含所有节点：users + user_codes + items + item_codes）
        num_nodes = self.user_num + self.n_user_codes + self.item_num + self.n_item_codes
        
        # 构建稀疏邻接矩阵
        from scipy.sparse import coo_matrix
        aug_adj = coo_matrix(
            (aug_edge_weight.cpu().numpy(), 
             (aug_edge_index[0].cpu().numpy(), aug_edge_index[1].cpu().numpy())),
            shape=(num_nodes, num_nodes)
        )
        
        # GCN 归一化
        aug_edge_index, aug_edge_weight = gcn_norm(aug_adj, num_nodes=num_nodes, self_loop=False)
        
        # 确保张量在正确的设备上
        aug_edge_index = aug_edge_index.to(self.device)
        aug_edge_weight = aug_edge_weight.to(self.device)
        
        return aug_edge_index, aug_edge_weight

    def aug_forward(self, aug_edge_index, aug_edge_weight):
        embeddings = self.get_w_codes_embeddings()
        embeddings_list = []  # CoGCL 原始设计：不包含初始嵌入

        for layer_idx in range(self.n_layers):
            embeddings = self.dropout(embeddings)
            embeddings = self.gcn_conv(embeddings, aug_edge_index, aug_edge_weight)
            embeddings_list.append(embeddings)

        all_embeddings = torch.stack(embeddings_list, dim=1)
        all_embeddings = torch.mean(all_embeddings, dim=1)

        user_all_embeddings, item_all_embeddings = torch.split(
            all_embeddings, [self.user_num + self.n_user_codes, self.item_num + self.n_item_codes]
        )
        return user_all_embeddings, item_all_embeddings

    def quantize(self, users, items, user_embeddings, item_embeddings):
        user_embs = user_embeddings[users]
        item_embs = item_embeddings[items]
        user_vq_loss, _ = self.emb_quantize(user_embs, self.user_vq)
        item_vq_loss, _ = self.emb_quantize(item_embs, self.item_vq)
        return user_vq_loss + item_vq_loss

    def emb_quantize(self, x, vq_layer):
        x = x.detach()
        x_q, mean_com_loss, all_codes = vq_layer(x)
        if self.code_dist.lower() == 'l2':
            recon_loss = F.mse_loss(x_q, x)
        elif self.code_dist.lower() == 'cos':
            recon_scores = torch.matmul(F.normalize(x_q, dim=-1), F.normalize(x, dim=-1).t()) / self.code_dist_tau
            recon_labels = torch.arange(x.size(0), dtype=torch.long).to(x.device)
            recon_loss = F.cross_entropy(recon_scores, recon_labels)
        else:
            raise NotImplementedError
        loss = recon_loss + mean_com_loss
        return loss, all_codes
    
    def get_share_codes_info(self, all_user_codes, all_item_codes):
        """
        计算码本相似度：找到拥有相似码本的用户和物品
        
        原理：如果两个用户/物品的码本有 >= code_num-1 个相同，则认为它们相似
        例如：user_code_num=4，如果两个用户至少有3个码本相同，则相似
        
        Args:
            all_user_codes: [n_users, user_code_num] 所有用户的码本索引
            all_item_codes: [n_items, item_code_num] 所有物品的码本索引
        
        Returns:
            similar_users: List[np.ndarray] - similar_users[i]是与用户i相似的用户ID列表
            similar_items: List[np.ndarray] - similar_items[i]是与物品i相似的物品ID列表
        """
        # print("Computing code-based similarity...")
        
        # 计算用户相似度
        start, batch_size = 0, self.code_batch_size
        similar_users = [[]] * self.user_num  # 与官方实现保持一致
        
        while start < self.user_num:
            batch_ucodes = all_user_codes[start: start + batch_size]  # [B, code_num]
            batch_sim = []
            
            # 分块计算相似度矩阵以节省内存
            k = 0
            while k < self.user_num:
                block = all_user_codes[k: k + batch_size]  # [B', code_num]
                # 计算码本匹配数: batch_ucodes[i] 与 block[j] 有多少个码本相同
                # [B, code_num, 1] == [1, code_num, B'] -> [B, code_num, B'] -> [B, B']
                sim = (batch_ucodes.unsqueeze(-1) == block.T.unsqueeze(0)).to(torch.int).sum(dim=1)
                batch_sim.append(sim)
                k += batch_size
            
            batch_sim = torch.cat(batch_sim, dim=1)  # [B, n_users]
            batch_sim[:, 0] = 0  # 排除ID=0（官方实现的做法）
            
            # 对于每个用户，找到相似度 >= user_code_num-1 的用户
            for i, sim in enumerate(batch_sim):
                uid = start + i
                sim_index = torch.where(sim >= self.user_code_num - 1)[0]
                similar_users[uid] = sim_index.cpu().numpy()
            
            start += batch_size
        
        # 计算物品相似度
        start, batch_size = 0, self.code_batch_size
        similar_items = [[]] * self.item_num  # 与官方实现保持一致
        
        while start < self.item_num:
            batch_icodes = all_item_codes[start: start + batch_size]
            batch_sim = []
            
            k = 0
            while k < self.item_num:
                block = all_item_codes[k: k + batch_size]
                sim = (batch_icodes.unsqueeze(-1) == block.T.unsqueeze(0)).to(torch.int).sum(dim=1)
                batch_sim.append(sim)
                k += batch_size
            
            batch_sim = torch.cat(batch_sim, dim=1)
            batch_sim[:, 0] = 0  # 排除ID=0（官方实现的做法）
            
            for i, sim in enumerate(batch_sim):
                iid = start + i
                sim_index = torch.where(sim >= self.item_code_num - 1)[0]
                similar_items[iid] = sim_index.cpu().numpy()
            
            start += batch_size
        
        mean_sim_users = np.mean([len(users) for users in similar_users if len(users) > 0])
        mean_sim_items = np.mean([len(items) for items in similar_items if len(items) > 0])
        print(f"Mean code-similar users: {mean_sim_users:.2f}")
        print(f"Mean code-similar items: {mean_sim_items:.2f}")
        
        return similar_users, similar_items
    
    def sample_augmented_pairs(self, users, items):
        """
        为sim_cl_loss采样增强的用户-物品对
        
        策略：结合码本相似度和协同过滤相似度
        - 对于用户u: 从 (交互过相同物品的用户 ∪ 码本相似用户) 中采样
        - 对于物品i: 从 (被相同用户交互的物品 ∪ 码本相似物品) 中采样
        
        与官方实现保持一致：
        1. 对batch中所有样本采样（包括重复的）
        2. 用unique索引切片，保证返回的维度与unique后的users/items一致
        
        Args:
            users: [batch_size] 用户ID（batch中所有样本，可能有重复）
            items: [batch_size] 物品ID（batch中所有样本，可能有重复）
        
        Returns:
            aug_users: [n_unique_users] 增强用户ID
            aug_items: [n_unique_items] 增强物品ID
        """
        users_np = users.cpu().numpy()
        items_np = items.cpu().numpy()
        device = users.device
        
        aug_users = []
        aug_items = []
        
        # 步骤1: 对batch中所有样本采样（包括重复的）
        for user_id, item_id in zip(users_np, items_np):
            # 采样增强用户
            same_item_users = self.same_target_users[item_id]  # 交互过item_id的用户
            similar_users = self.code_similar_users[user_id]   # 与user_id码本相似的用户
            
            # 混合策略：优先选择same_item_users，补充similar_users
            sim_num = len(same_item_users) // 2 + 1
            if len(similar_users) > sim_num:
                similar_users = np.random.choice(similar_users, size=sim_num, replace=False)
            
            cand_users = list(set(same_item_users) | set(similar_users))
            
            if len(cand_users) == 0:
                aug_users.append(user_id)  # 没有候选，用自己
            elif len(cand_users) == 1:
                aug_users.append(cand_users[0])
            else:
                sample_user = np.random.choice(cand_users)
                # 避免采样到自己
                while sample_user == user_id and len(cand_users) > 1:
                    sample_user = np.random.choice(cand_users)
                aug_users.append(sample_user)
            
            # 采样增强物品
            same_user_items = self.same_target_items[user_id]  # user_id交互过的物品
            similar_items = self.code_similar_items[item_id]   # 与item_id码本相似的物品
            
            sim_num = len(same_user_items) // 2 + 1
            if len(similar_items) > sim_num:
                similar_items = np.random.choice(similar_items, size=sim_num, replace=False)
            
            cand_items = list(set(same_user_items) | set(similar_items))
            
            if len(cand_items) == 0:
                aug_items.append(item_id)
            elif len(cand_items) == 1:
                aug_items.append(cand_items[0])
            else:
                sample_item = np.random.choice(cand_items)
                while sample_item == item_id and len(cand_items) > 1:
                    sample_item = np.random.choice(cand_items)
                aug_items.append(sample_item)
        
        # 步骤2: 用unique索引切片（与官方实现完全一致）
        unique_users_idx = np.unique(users_np, return_index=True)[1]
        unique_items_idx = np.unique(items_np, return_index=True)[1]
        
        aug_users = torch.LongTensor(aug_users)[unique_users_idx].to(device)
        aug_items = torch.LongTensor(aug_items)[unique_items_idx].to(device)
        
        return aug_users, aug_items

    def calculate_cl_loss(self, x1, x2):
        """
        完整的 InfoNCE 对比学习损失
        """
        x1, x2 = F.normalize(x1, dim=-1), F.normalize(x2, dim=-1)
        
        # 计算相似度矩阵
        sim_12 = torch.mm(x1, x2.T)  # [N, N]
        sim_21 = sim_12.T
        sim_11 = torch.mm(x1, x1.T)
        sim_22 = torch.mm(x2, x2.T)
        
        # 应用温度并指数化
        sim_12 = torch.exp(sim_12 / self.cl_tau)
        sim_21 = torch.exp(sim_21 / self.cl_tau)
        sim_11 = torch.exp(sim_11 / self.cl_tau)
        sim_22 = torch.exp(sim_22 / self.cl_tau)
        
        # InfoNCE 损失
        # 分子：正样本对（对角线）
        # 分母：所有样本对（排除自身）
        loss_12 = -torch.log(
            sim_12.diag() / (sim_12.sum(1) + sim_11.sum(1) - sim_11.diag())
        )
        loss_21 = -torch.log(
            sim_21.diag() / (sim_21.sum(1) + sim_22.sum(1) - sim_22.diag())
        )
        
        # 双向损失的平均
        loss = 0.5 * (loss_12 + loss_21).mean()
        return loss

    def loss(self, out_dict, feed_dict):
        bpr_loss = super().loss(out_dict, feed_dict)
        user_all_embeddings, item_all_embeddings = self._forward_gcn(drop=self.drop_fwd)
        
        users = feed_dict['user_id']
        pos_items = feed_dict['item_id'][:, 0]
        neg_items = feed_dict['item_id'][:, 1]

        u_ego_embeddings = self.user_embedding(users)
        pos_ego_embeddings = self.item_embedding(pos_items)
        neg_ego_embeddings = self.item_embedding(neg_items)
        reg_loss = (u_ego_embeddings.norm(2).pow(2) + pos_ego_embeddings.norm(2).pow(2) + neg_ego_embeddings.norm(2).pow(2)) / len(users)

        unique_users = torch.unique(users)
        unique_items = torch.unique(pos_items)
        vq_loss = self.quantize(unique_users, unique_items, user_all_embeddings, item_all_embeddings)

        if self.epoch_num >= self.data_aug_delay:
            user_aug1_embeddings, item_aug1_embeddings = self.aug_forward(self.aug_edge_index_1, self.aug_edge_weight_1)
            user_aug2_embeddings, item_aug2_embeddings = self.aug_forward(self.aug_edge_index_2, self.aug_edge_weight_2)
            u_x1, u_x2 = user_aug1_embeddings[unique_users], user_aug2_embeddings[unique_users]
            i_x1, i_x2 = item_aug1_embeddings[unique_items], item_aug2_embeddings[unique_items]
            cl_loss = self.calculate_cl_loss(u_x1, u_x2) + self.calculate_cl_loss(i_x1, i_x2)
            
            # 相似性对比学习损失 (sim_cl_loss)
            if self.sim_cl_weight > 0 and self.code_similar_users is not None:
                # 采样增强的用户-物品对（对batch中所有样本采样，然后用unique索引切片）
                aug_users, aug_items = self.sample_augmented_pairs(users, pos_items)
                
                # 获取增强样本的embedding
                u_x3 = user_all_embeddings[aug_users]
                i_x3 = item_all_embeddings[aug_items]
                
                # 计算相似性对比学习损失
                # 让增强视图1和增强样本对比，增强视图2和增强样本对比
                sim_cl_loss_1 = self.calculate_cl_loss(u_x1, u_x3) + self.calculate_cl_loss(i_x1, i_x3)
                sim_cl_loss_2 = self.calculate_cl_loss(u_x2, u_x3) + self.calculate_cl_loss(i_x2, i_x3)
                sim_cl_loss = 0.5 * (sim_cl_loss_1 + sim_cl_loss_2)
            else:
                sim_cl_loss = torch.zeros(1).to(self.device)
        else:
            cl_loss = torch.zeros(1).to(self.device)
            sim_cl_loss = torch.zeros(1).to(self.device)

        total_loss = bpr_loss + self.reg_weight * reg_loss + self.vq_loss_weight * vq_loss + \
                     self.cl_weight * cl_loss + self.sim_cl_weight * sim_cl_loss
        
        # 打印各个损失分量（用于调试）
        if hasattr(self, '_step_counter'):
            self._step_counter += 1
        else:
            self._step_counter = 0
        
        if self._step_counter % 500 == 0:
            print(f"\n[Loss Components] BPR={bpr_loss.item():.4f}, Reg={reg_loss.item():.4f}, VQ={vq_loss.item():.4f}, CL={cl_loss.item():.4f}, SimCL={sim_cl_loss.item():.4f}")
        
        return total_loss

