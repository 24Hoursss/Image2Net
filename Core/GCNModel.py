import dgl
import torch
import torch.nn as nn
from dgl.nn.pytorch import GraphConv, HeteroGraphConv


# 自定义网络
class Classifier(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_classes, norm='none'):
        super(Classifier, self).__init__()
        self.conv1 = GraphConv(in_dim, hidden_dim, weight=True, bias=True, allow_zero_in_degree=True,
                               norm=norm)  # 定义第一层图卷积
        self.conv2 = GraphConv(hidden_dim, hidden_dim, weight=True, bias=True, allow_zero_in_degree=True,
                               norm=norm)  # 定义第二层图卷积
        self.conv3 = GraphConv(hidden_dim, hidden_dim, weight=True, bias=True, allow_zero_in_degree=True,
                               norm=norm)  # 定义第三层图卷积
        # self.conv4 = GraphConv(hidden_dim, hidden_dim)  # 定义第四层图卷积
        # self.conv5 = GraphConv(hidden_dim, hidden_dim)  # 定义第五层图卷积
        # self.conv6 = GraphConv(hidden_dim, hidden_dim)  # 定义第六层图卷积
        # self.classify = nn.Sequential(nn.Linear(hidden_dim, n_classes))  # 定义分类器
        self.classify = nn.Sequential(nn.Linear(hidden_dim, hidden_dim),
                                      nn.ReLU(),
                                      nn.Linear(hidden_dim, hidden_dim),
                                      nn.ReLU(),
                                      nn.Linear(hidden_dim, n_classes))  # 定义分类器

    def forward(self, g):
        """
        g表示批处理后的大图，N表示大图的所有节点数量，n表示图的数量
        """
        # 执行图卷积和激活函数
        h = g.ndata['h']
        h = torch.relu(self.conv1(g, h))  # [N, hidden_dim]
        h = torch.relu(self.conv2(g, h))  # [N, hidden_dim]
        h = torch.relu(self.conv3(g, h))  # [N, hidden_dim]
        # h = torch.relu(self.conv4(g, h))  # [N, hidden_dim]
        # h = torch.relu(self.conv5(g, h))  # [N, hidden_dim]
        # h = torch.relu(self.conv6(g, h))  # [N, hidden_dim]
        g.ndata['x'] = h  # 将特征赋予到图的节点
        with g.local_scope():
            # 通过平均池化每个节点的表示得到图表示
            hg = dgl.mean_nodes(g, 'x')  # [n, hidden_dim]
            h = self.classify(hg)
            return h  # [n, n_classes]


class ClassifierHetero(nn.Module):
    def __init__(self, hidden_dim, n_classes, norm='none'):
        super(ClassifierHetero, self).__init__()

        in_dims = {
            'component': 1,
            'port': 2,
            'net': 1
        }

        self.conv1 = HeteroGraphConv({
            ('component', 'component_to_port', 'port'): GraphConv(in_dims['component'], hidden_dim, norm=norm,
                                                                  allow_zero_in_degree=True),
            ('port', 'port_to_net', 'net'): GraphConv(in_dims['port'], hidden_dim, norm=norm,
                                                      allow_zero_in_degree=True),
        }, aggregate='sum')

        self.conv2 = HeteroGraphConv({
            ('component', 'component_to_port', 'port'): GraphConv(hidden_dim, hidden_dim, norm=norm,
                                                                  allow_zero_in_degree=True),
            ('port', 'port_to_net', 'net'): GraphConv(hidden_dim, hidden_dim, norm=norm, allow_zero_in_degree=True)
        }, aggregate='sum')

        self.conv3 = HeteroGraphConv({
            ('component', 'component_to_port', 'port'): GraphConv(hidden_dim, hidden_dim, norm=norm,
                                                                  allow_zero_in_degree=True),
            ('port', 'port_to_net', 'net'): GraphConv(hidden_dim, hidden_dim, norm=norm, allow_zero_in_degree=True)
        }, aggregate='sum')

        self.classify = nn.Sequential(
            nn.Linear(4, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_classes)
        )

    def forward(self, g):
        h_dict = {
            'component': g.nodes['component'].data['h'],
            'port': g.nodes['port'].data['h'],
            'net': g.nodes['net'].data['h']
        }

        with g.local_scope():
            h_dict = self.conv1(g, h_dict)
            h_dict = {k: torch.relu(v) for k, v in h_dict.items()}

            h_dict = self.conv2(g, h_dict)
            h_dict = {k: torch.relu(v) for k, v in h_dict.items()}

            h_dict = self.conv3(g, h_dict)
            h_dict = {k: torch.relu(v) for k, v in h_dict.items()}

            # hg = dgl.mean_nodes(g, 'h', ntype='component')

            hg_components = dgl.mean_nodes(g, 'h', ntype='component')
            hg_ports = dgl.mean_nodes(g, 'h', ntype='port')
            hg_nets = dgl.mean_nodes(g, 'h', ntype='net')
            hg = torch.cat([hg_components, hg_ports, hg_nets], dim=1)

            # Classify
            h = self.classify(hg)
            return h  # [n, n_classes]
