import networkx as nx
import re
import dgl
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from collections import defaultdict
from typing import List
from Core.interface import Point


# 可视化 batch 后图的连接关系
def g_view(g):
    nx.draw(g.to_networkx(), with_labels=True)
    plt.show()


def g_view_detail(g):
    nodeLabel = 'h'
    plt.figure(figsize=(8, 8))
    G = g.to_networkx(node_attrs=nodeLabel.split())  # 转换 dgl graph to networks
    pos = nx.spring_layout(G)
    nx.draw(G, pos, edge_color="grey", node_size=100, with_labels=True, font_size=12)  # 画图，设置节点大小
    node_data = nx.get_node_attributes(G, nodeLabel)  # 获取节点的desc属性
    node_labels = {index: "N{}:".format(index) + str(data.tolist()) for index, data in
                   enumerate(node_data.values())}  # 重新组合数据， 节点标签是dict, {nodeid:value,nodeid2,value2} 这样的形式
    pos_higher = {}

    for k, v in pos.items():  # 调整下顶点属性显示的位置，不要跟顶点的序号重复了
        if (v[1] > 0):
            pos_higher[k] = (v[0] - 0.04, v[1] + 0.04)
        else:
            pos_higher[k] = (v[0] - 0.04, v[1] - 0.04)
    nx.draw_networkx_labels(G, pos_higher, labels=node_labels, font_color="brown", font_size=6)  # 将desc属性，显示在节点上
    plt.show()


def g_view_hetero(g):
    G = dgl.to_networkx(g, node_attrs=['h'], edge_attrs=None)
    pos = nx.spring_layout(G)

    plt.figure(figsize=(8, 8))
    nx.draw(G, pos, with_labels=True, node_size=100, edge_color="grey", font_size=12)
    plt.show()


def g_view_detail_hetero(g):
    plt.figure(figsize=(10, 10))

    G = dgl.to_networkx(g, node_attrs=['h'])
    pos = nx.spring_layout(G)

    for ntype in g.ntypes:
        nids = g.nodes(ntype).tolist()
        nx.draw_networkx_nodes(G, pos, nodelist=nids, node_color='lightgrey', node_size=100)
        labels = {nid: ntype for nid in nids}
        nx.draw_networkx_labels(G, pos, labels=labels, font_color="black", font_size=8)

    nx.draw_networkx_edges(G, pos, edge_color='grey', alpha=0.5)

    node_data = nx.get_node_attributes(G, 'h')
    node_labels = {index: str(data.tolist()) for index, data in node_data.items()}
    pos_higher = {k: (v[0], v[1] + 0.05) for k, v in pos.items()}
    nx.draw_networkx_labels(G, pos_higher, labels=node_labels, font_color="brown", font_size=6)

    plt.show()


# 可视化训练曲线，从 line 中保存的数据而来
def view_r2_curve():
    f = open("line.txt", 'r').readlines()
    r_train = list(map(float, re.findall(r"-?(?:[1-9]\d*\.\d*|0\.\d*[1-9]\d*|0\.0+|0)", f[0])))
    # r_test = list(map(float, re.findall(r"-?(?:[1-9]\d*\.\d*|0\.\d*[1-9]\d*|0\.0+|0)", f[1])))
    plt.plot(np.linspace(0, len(r_train), len(r_train)), r_train, 'b-', label='r_train')
    # plt.plot(np.linspace(0, len(r_test), len(r_test)), r_test, 'r-', label='r_test')
    plt.ylim(0, 0.7)
    plt.legend()
    plt.show()


def layerFromgraph(g, h=None):
    print(f"{g=}")
    print(f"{h=}")
    if h is None:
        point = g.ndata['h'].numpy()
    else:
        point = h.detach().numpy()
        print(point.shape)
    point = point.tolist()
    print(point)
    edges = g.to_networkx().edges()
    for edge in edges:
        x, y, _ = list(zip(point[edge[0]], point[edge[1]]))
        plt.plot(x, y, 'b-')
    plt.show()


def plot_result(corners: List[Point], type: str, image, save_path=None):
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    plt.title(type)

    ax = plt.gca()

    parent_groups = defaultdict(list)
    for point in corners:
        if point.parent:
            parent_groups[point.parent].append(point)

    for parent, points in parent_groups.items():
        component_type = parent.classification
        if component_type == 'BasicCircuit':
            continue

        x1, y1, x2, y2 = parent.xyxy
        rect = Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        plt.text(x1, y1, f'{component_type[0]}', color='white', fontsize=8, bbox=dict(facecolor='red', alpha=0.5))

        for point in points:
            if point.outputName:
                x, y = point.x, point.y
                plt.scatter(x, y, c='blue', s=10)
                # plt.text(x, y, f'{point.name}:{point.outputName}', color='blue', fontsize=8)
                plt.text(x, y, f'{point.name[0]}:{point.outputName}', color='blue', fontsize=8)

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()
