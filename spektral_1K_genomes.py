import subprocess
import os.path, os
import numpy as np
import pandas as pd
import snap
import pickle
import yaml
from tqdm import tqdm
from time import time
from scipy import sparse as sp
from spektral.data import Dataset, Graph
from sklearn.model_selection import train_test_split
from random import shuffle


class snp_graph(Dataset):
    """
    1000 genomes snp graph dataset
    https://docs.google.com/presentation/d/1OZS95WpLmscVKUnstQax5e6Ulplx53OsJjxyM50NMvU/edit?usp=sharing
    **Arguments**
    - `amount`: int, load this many molecules instead of the full dataset
    (useful for debugging).
    edge_features : weight, zscore
    node_features : p, chromosome, ref, centromere_rel_pos, homhet
    """

    def __init__(self, amount=None,
                 node_features = ['p'], # укзать фичи для вершин через list 
                 #'p', 'centromere_rel_pos' - float, 
                 #'chromosome', 'ref', 'alt', 'homhet' - категориальные
                 edge_features = ['zref'], # укзать фичи для рёбер через list
                 # 'weight', 'zref'
                 use_weight_in_adjency = True, # использовать в adjency [0, 1] 
                 # или weight
                 labels = ['p'], # укзать выход через list
                 # 'centromere_rel_pos', 'p' - float
                 # 'chromosome', 'ref', 'alt', 'homhet' - категориальные
                 **kwargs):
        self.amount = amount
        self.edge_features = edge_features
        self.node_features = node_features
        self.use_weight_in_adjency = use_weight_in_adjency
        self.labels = labels
        self.dtype = np.float32
        self.mask_tr = self.mask_va = self.mask_te = None
        super().__init__(**kwargs)

    def run_bash(self, bashCommand:str, nameCommand = ''):
        process = subprocess.Popen([bashCommand], 
                           shell=True)
        _, error = process.communicate()
        if error:
            print(f'{nameCommand} error:\n', error)

    def read(self):
        def subset_nodes(start_nodes, G, subgraph_set, sub_graph_size = 1000):
            # Большая часть кода создает однокомпанентный подграф с вершинами
            # по возможности равноудаленными от исходной
            hop_set = set()
            for start_node in start_nodes:
                subgraph_set.add(start_node) #
                if len(subgraph_set) >= sub_graph_size:
                    return subgraph_set
                id_node = G.GetNI(start_node)
                deg_node = id_node.GetDeg()
                for cnt in range(deg_node):
                    hop_set.add(id_node.GetNbrNId(cnt))
            hop_set = hop_set - subgraph_set
            if not hop_set:
                return subgraph_set
            return(subset_nodes(hop_set, G, subgraph_set, sub_graph_size = sub_graph_size))

        usecols = ['source', 'target']
        if self.edge_features:
            usecols += self.edge_features
        if self.use_weight_in_adjency:
            if 'weight' not in usecols: #if 'weight' not in usecols:
                usecols += ['weight'] #usecols += ['weight']
        edges_df = pd.read_csv(
                './1K_graph_edges_with_zscore.csv.gz',
                compression='gzip',
                sep = ' ', names = ['source', 'target', 'weight', 'zref', 'zalt'], 
                usecols = usecols
                                )
        node_set = set(edges_df.source.unique()) | set(edges_df.target.unique())
        if self.amount:
            print(f'Creating subgraph for {self.amount} nodes...')
            S = snap.TUNGraph.New()
            selfedges = 0
            for item in tqdm(node_set):
                S.AddNode(int(item))
            del node_set
            gc.collect()
            for row in tqdm(edges_df[['source', 'target']].itertuples(), total = len(edges_df)):
                if not(S.IsEdge(row[1], row[2])):
                    if row[1] != row[2]:
                        S.AddEdge(row[1], row[2])
                    else:
                        selfedges += 1
                elif S.IsEdge(row[2], row[1]):
                    print(f'Is Edge {row[2]} - {row[1]}')
                elif row[1] != row[2]:
                    print(f'self{row1}')
            print(f'find & skiped self edges: {selfedges}')    
            rnd_node = S.GetRndNId(snap.TRnd(42))
            subgraph_set = subset_nodes({rnd_node}, S, {rnd_node}, sub_graph_size = self.amount)
            s = S.GetSubGraph(list(subgraph_set))
            print('Extracting nodes from subgraph...')
            subgraph_edges_idx = [cnt for cnt, edge in tqdm(enumerate(edges_df[['source', 'target']].values), 
                                                total = len(edges_df)) if s.IsNode(int(edge[0])) & s.IsNode(int(edge[1]))]
            del S
            gc.collect()
        edges_stack = edges_df[['source', 'target']].values
        if self.amount:
            edges_stack = edges_stack[subgraph_edges_idx]
        #node_array = np.copy(edges_stack)
        #print(node_array.shape)
        edges_stack_idx = np.empty(edges_stack.max() + 1, dtype=int)
        edges_stack_idx[np.unique(edges_stack)] = np.arange(np.unique(edges_stack).shape[0])
        edges_stack = np.vstack((edges_stack, np.flip(edges_stack, axis = 1)))
        if self.use_weight_in_adjency:
            if self.amount:
                adj = np.hstack((edges_df.weight.values[subgraph_edges_idx], 
                            edges_df.weight.values[subgraph_edges_idx]))
            else:
                adj = np.hstack((edges_df.weight.values, 
                            edges_df.weight.values))
        else:
            adj = np.ones(edges_stack[:,0].shape[0])
        row = edges_stack_idx[edges_stack[:, 0]]
        col = edges_stack_idx[edges_stack[:, 1]]
        del edges_stack_idx
        gc.collect()
        a = sp.csr_matrix((adj, (row, col)), shape=None).astype(self.dtype)
        del adj
        gc.collect()
        csr_index = np.arange(edges_stack[:,0].shape[0])
        #del edges_stack
        #gc.collect()
        a_idx = sp.csr_matrix((csr_index, (row, col)), shape=None).astype(np.float32)
        csr_index = 0
        cx = sp.coo_matrix(a_idx)
        del a_idx
        gc.collect()
        index_list = []
        for i, j, v in zip(cx.row, cx.col, cx.data):
            index_list.append(int(v))
        del cx
        gc.collect()
        print('Adjency matrix created...')
        if self.edge_features:
            e = edges_df[self.edge_features].values
            del edges_df
            gc.collect()
            if self.amount:
                e = e[subgraph_edges_idx]           
            e = e.astype(self.dtype)
            e = np.vstack((e, e))
            e = e[index_list]
            del index_list
            gc.collect()
            print('Edges features assigned...')
        else:
            e = None
            print('..................No edges features')

        print('Load nodes features & labels...')
        usecols = []
        if self.node_features:
            usecols += self.node_features
        if self.labels:
            usecols += self.labels
        nodes_df = pd.read_csv('./1K_nodes.csv.gz', 
                               compression='gzip',
                               usecols = usecols)
        nodes_df = nodes_df[nodes_df.index.isin(edges_stack)]
        del edges_stack
        gc.collect()
        categorical_features = ['chromosome', 'ref', 'alt', 'homhet']
        float_features = ['p', 'centromere_rel_pos']
        if self.node_features:
            x = np.empty([len(nodes_df), 1])
            for ff in float_features:
                if ff in self.node_features:
                    x = np.hstack((x, nodes_df[ff].values.reshape(-1, 1)))
            for cf in categorical_features:
                if cf in self.node_features:
                    x = np.hstack((x, pd.get_dummies(nodes_df[cf]).values))
            x = x[:, 1:].astype(self.dtype)
        else:
            x = None
            print('No node features...')
        categorical_features_flag = False
        if self.labels:
            y = np.empty([len(nodes_df), 1])
            for ff in float_features:
                if ff in self.labels:
                    y = np.hstack((y, nodes_df[ff].values.reshape(-1, 1)))
            categorical_features_start = y.shape[1]
            for cf in categorical_features:
                if cf in self.labels:
                    categorical_features_flag = True
                    y = np.hstack((y, pd.get_dummies(nodes_df[cf]).values))
            y = y[:, 1:].astype(self.dtype)
        else:
            y = None
            print('......No labels..............')            

        # Public Planetoid splits. This is the default
        print('Split train...validation...test')
        if categorical_features_flag:
            stratify_array = y[:,categorical_features_start:]
        else:
            stratify_array = None
        self.mask_tr = np.zeros(y.shape[0], dtype=np.bool)
        self.mask_va = np.zeros(y.shape[0], dtype=np.bool)
        self.mask_te = np.zeros(y.shape[0], dtype=np.bool)
        #print('zeros shape: ', self.mask_tr.shape, self.mask_va.shape, self.mask_te.shape)
        init_mask_tr, init_mask_va = train_test_split(
                np.arange(y.shape[0]), train_size=0.5, 
                random_state = 149, stratify = stratify_array
            )
        print('Train done')
        if categorical_features_flag:
            stratify_array = stratify_array[init_mask_va]
        print('stratify done')
        init_mask_va, init_mask_te = train_test_split(
                init_mask_va, train_size=0.6, 
                random_state = 149, stratify = stratify_array
            )
        #print(init_mask_tr, init_mask_va, init_mask_te)
        #print(init_mask_tr.shape, init_mask_va.shape, init_mask_te.shape)
        self.mask_tr[init_mask_tr] = True
        self.mask_va[init_mask_va] = True
        self.mask_te[init_mask_te] = True
        return [Graph(x = x, 
                      a = a, 
                      e = e,
                      y = y)]
    
    def download(self):
        if not os.path.isfile('./1K_nodes.csv.gz'):
            print('Downloading nodes...')
            bashCommand = f"""
            wget -q https://raw.githubusercontent.com/cappelchi/Datasets/master/1K_nodes.csv.gz
            """
            self.run_bash (bashCommand, 'Downloading nodes error: ')
        if not os.path.isfile('./1K_graph_edges_with_zscore.csv.gz'):
            print('Downloading edges...')
            bashCommand = f"""
            wget -q -O 1K_graph_edges_with_zscore.csv.gz https://getfile.dokpub.com/yandex/get/https://disk.yandex.ru/d/7--he9iQyVPhHg
            """
            self.run_bash (bashCommand, 'Downloading edges error: ')
