import subprocess
import os.path, os
import random
import numpy as np
import pandas as pd
import snap
from tqdm import tqdm
from scipy import sparse as sp
from spektral.data import Dataset, Graph
from sklearn.model_selection import train_test_split

class nodes_dataset(Dataset):
    """
    1000 genomes snp graph dataset
    https://docs.google.com/presentation/d/1OZS95WpLmscVKUnstQax5e6Ulplx53OsJjxyM50NMvU/edit?usp=sharing
    **Arguments**
    - `amount`: int, subgraphs quantity.
    edge_features : use sign of zscore
    node_features : without node features
    """

    def __init__(self,
                 node_features = 1, #любое число
                 edge_features = False, # используем знак zscore как фичу для вершины
                 use_weight_in_adjency = True, # использовать zscore в adjency [0, 1]
                 labels = ['p'], # укзать выход через list
                 # 'centromere_rel_pos', 'p' - float
                 # 'chromosome', 'ref', 'alt', 'homhet' - категориальные
                 **kwargs):
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
        # Загружаем ноды в датафрейм
        usecols = ['p']
        print('Read nodes.....')
        nodes_df = pd.read_csv('./1K_nodes.csv.gz',
                               compression='gzip',
                               usecols = usecols)
        # Загружаем рёбра в датафрейм
        usecols = ['source', 'target', 'zref']
        print('Read edges.......')
        edges_df = pd.read_csv(
                './1K_graph_edges_with_zscore.csv.gz',
                compression='gzip',
                sep = ' ', names = ['source', 'target', 'weight', 'zref', 'zalt'], 
                usecols = usecols
                                )
        node_set = set(edges_df.source.unique()) | set(edges_df.target.unique())
        S = snap.TNGraph.New() #Объявляем направленный граф, т.к. важно сохранить source->target
        selfedges = 0
        for item in tqdm(node_set):
            S.AddNode(int(item))
        for row in tqdm(edges_df[['source', 'target']].itertuples(), total = len(edges_df)):
            if not(S.IsEdge(row[1], row[2])):
                if not(S.IsEdge(row[2], row[1])):
                    if row[1] != row[2]:
                        S.AddEdge(row[1], row[2])
                    else:
                        selfedges += 1
                else:
                    print(f'Reverse repeat Edge {row[2]} - {row[1]}')
            else:
                print(f'Exist Edge {row[1]} - {row[2]}')
        print(f'find & skiped self edges: {selfedges}')
        if self.use_weight_in_adjency:
            adj = np.hstack((np.abs(edges_df['zref'].values / 30),
                             np.abs(edges_df['zref'].values / 30)))
        else:
            adj = np.ones(len(edges_df) * 2)
        idx_dict = {idx:nidx for idx, nidx in zip(np.array(list(node_set)), np.arange(len(node_set)))}
        row = [idx_dict[idx] for idx in edges_df.source.values]
        col = [idx_dict[idx] for idx in edges_df.target.values]
        a = sp.csr_matrix((adj, (row + col, col + row)), shape=None).astype(np.float32)

        csr_index = np.arange(len(row + col))
        a_idx = sp.csr_matrix((csr_index, (row + col, col + row)), shape=None).astype(np.float32)
        cx = sp.coo_matrix(a_idx)
        index_list = [int(v) for v in cx.data]
        if self.edge_features:
            e = np.hstack(((edges_df['zref'].values > 0),
                               ~(edges_df['zref'].values > 0)))
            e = np.vstack((e, ~e)).T.astype(np.int16)
            e = e[index_list,:]
        else:
            e = None
        x = np.ones(len(node_set)).reshape(-1, 1).astype(np.float32) * self.node_features
        y = nodes_df['p'].values
        self.mask_tr = np.zeros(y.shape[0], dtype=np.bool_)
        self.mask_va = np.zeros(y.shape[0], dtype=np.bool_)
        self.mask_te = np.zeros(y.shape[0], dtype=np.bool_)
        init_mask_tr, init_mask_va = train_test_split(np.arange(y.shape[0]), train_size=0.5, random_state=149)
        init_mask_va, init_mask_te = train_test_split(init_mask_va, train_size=0.6,random_state=149)
        self.mask_tr[init_mask_tr] = True
        self.mask_va[init_mask_va] = True
        self.mask_te[init_mask_te] = True
        return [Graph(x=x,
                      a=a,
                      e=e,
                      y=y)]

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
                wget -q -O 1K_graph_edges_with_zscore.csv.gz https://getfile.dokpub.com/yandex/get/https://disk.yandex.ru/d/J-SWE1p9hSWTqw
                """
                self.run_bash (bashCommand, 'Downloading edges error: ')
