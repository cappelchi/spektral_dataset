import subprocess
import os.path, os
import random
import numpy as np
import pandas as pd
import snap
from tqdm import tqdm
from scipy import sparse as sp
from spektral.data import Dataset, Graph


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

    def __init__(self, amount=1000,
                 node_features = None, # укзать фичи для вершин через list
                 #'p', 'centromere_rel_pos' - float, 
                 #'chromosome', 'ref', 'alt', 'homhet' - категориальные
                 edge_features = True, # укзать фичи для рёбер через list
                 # 'weight', 'zref'
                 use_weight_in_adjency = True, # использовать в adjency [0, 1] 
                 # или weight
                 labels = ['p'], # укзать выход через list
                 # 'centromere_rel_pos', 'p' - float
                 # 'chromosome', 'ref', 'alt', 'homhet' - категориальные
                 graph_size = 100, # желаемый размер графа
                 hops = 1, # минимальное количество hop(ов)
                 **kwargs):
        self.amount = amount
        self.edge_features = edge_features
        self.node_features = node_features
        self.use_weight_in_adjency = use_weight_in_adjency
        self.labels = labels
        self.graph_size = graph_size
        self.hops = hops
        self.dtype = np.float32
        super().__init__(**kwargs)

    def run_bash(self, bashCommand:str, nameCommand = ''):
        process = subprocess.Popen([bashCommand], 
                           shell=True)
        _, error = process.communicate()
        if error:
            print(f'{nameCommand} error:\n', error)
    def read(self):
        def subset_nodes(start_nodes, G, subgraph_set, sub_graph_size=1000, min_hops=1):
            # 1. Собираем всех соседей для всех start_nodes (вначале нулевая нода)
            # 2. Если выполняются условия (1) и (2) выходим из текщего вложения функции
            # 3. Рекурсивно идем ниже собирать соседей для соседей
            hop_set = set()  # создаем сет куда будем складывать ноды для результирующего подграфа
            for start_node in start_nodes:  #
                if min_hops < 0:  # условие (1)
                    if len(subgraph_set) >= sub_graph_size:  # условие (2)
                        return subgraph_set
                subgraph_set.add(start_node)  #
                id_node = G.GetNI(start_node)
                deg_node = id_node.GetDeg()
                for cnt in range(deg_node):
                    hop_set.add(id_node.GetNbrNId(cnt))
            min_hops = min_hops - 1
            hop_set = hop_set - subgraph_set
            if not hop_set:
                return subgraph_set
            return (subset_nodes(hop_set, G, subgraph_set, sub_graph_size=sub_graph_size, min_hops=min_hops))
        # Загружаем ноды в датафрейм
        usecols = ['p']
        #if self.node_features:
        #    usecols += self.node_features
        #if self.labels:
        #    usecols += self.labels
        print('Read nodes.....')
        nodes_df = pd.read_csv('./1K_nodes.csv.gz',
                               compression='gzip',
                               usecols = usecols)
        # Загружаем рёбра в датафрейм
        usecols = ['source', 'target', 'zref']
        #if self.edge_features:
        #    usecols += self.edge_features
        #if self.use_weight_in_adjency:
        #    if 'weight' not in usecols: #if 'weight' not in usecols:
        #        usecols += ['weight'] #usecols += ['weight']
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
        edges_df = edges_df.set_index(['source', 'target']) # Переиндексация на мультииндекс
        graphs_list = []
        print(f'Generating {self.amount} subgraphs...........')
        for sample_node in tqdm(random.sample(node_set, self.amount)):
            rnd_node = int(sample_node)
            subgraph_set = subset_nodes({rnd_node}, S, {rnd_node},
                                        sub_graph_size = self.graph_size,
                                        min_hops = self.hops)
            s = S.GetSubGraph(list(subgraph_set))
            subgraph_edges_idx = [[ed.GetSrcNId(), ed.GetDstNId()] for ed in s.Edges()]
            if self.use_weight_in_adjency:
                adj = np.hstack((np.abs(edges_df['zref'][subgraph_edges_idx].values / 30),
                                 np.abs(edges_df['zref'][subgraph_edges_idx].values / 30)))
            else:
                adj = np.ones(len(subgraph_edges_idx) * 2)
            adj[np.flip(np.array(subgraph_edges_idx), axis= 1).flatten('F') == rnd_node] = 0
            idx_dict = {idx: nidx for idx, nidx in
                        zip(np.unique(subgraph_edges_idx), np.arange(len(np.unique(subgraph_edges_idx))))}
            row = [idx_dict[idx] for idx in np.array(subgraph_edges_idx)[:, 0]]
            col = [idx_dict[idx] for idx in np.array(subgraph_edges_idx)[:, 1]]
            a = sp.csr_matrix((adj, (row + col, col + row)), shape=None).astype(np.float32)

            csr_index = np.arange(len(row + col))
            a_idx = sp.csr_matrix((csr_index, (row + col, col + row)), shape=None).astype(np.float32)
            cx = sp.coo_matrix(a_idx)
            index_list = [int(v) for v in cx.data]
            if self.edge_features:
                e = np.hstack(((edges_df.zref[subgraph_edges_idx].values > 0),
                                   ~(edges_df.zref[subgraph_edges_idx].values > 0)))
                e = np.vstack((e, ~e)).T.astype(np.int16)
                e = e[index_list,:]
            else:
                e = None
            if self.node_features:
                x = np.ones(len(subgraph_set)).reshape(-1, 1).astype(np.float32) * self.node_features
            else:
                x = None
            y = nodes_df.values[rnd_node].copy()
            #x[list(subgraph_set).index(rnd_node)] = 0 #np.nan
            graphs_list.append(Graph(x = x,
                                      a = a,
                                      e = e,
                                      y = y.astype(self.dtype)))

        return graphs_list
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
