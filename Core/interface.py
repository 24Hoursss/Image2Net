from typing import List, Optional, Set
from Core.DeviceType import DeviceType, swappedDeviceType, key_indices
from collections import defaultdict
import dgl
import torch
from copy import deepcopy


# print(swappedDeviceType)
# print(DeviceType)

class Node:
    def __init__(self, type: str = '', xyxy: List[float] = None):
        self.type = type
        self.classification = swappedDeviceType[type] if type in swappedDeviceType else None
        self.xyxy = xyxy if xyxy is not None else []
        self.children: List['Point'] = []

    def add_child(self, child: 'Point'):
        child.parent = self
        self.children.append(child)

    def __repr__(self):
        return f"Node(type={self.type}, classification={self.classification})"


class Point:
    def __init__(self, x: float = 0, y: float = 0, name: str = '', parent: Optional[Node] = None):
        self.x = x
        self.y = y
        self.name = name
        self.parent = parent
        self.connect = {self}
        self.outputName = ''
        self.nCorner = 0

    def add_connect(self, point: 'Point'):
        self.connect.add(point)

    def add_connects(self, points: Set['Point']):
        self.connect.update(points)

    def __repr__(self):
        return f"Point(x={self.x}, y={self.y}, name={self.name})"


def iter_rename(point: Point, name):
    for vdd_point in point.connect:
        if vdd_point.outputName != name:
            vdd_point.outputName = name
            if vdd_point.connect is not point.connect:
                iter_rename(vdd_point, name)

def rename(point: Point, name):
    for vdd_point in point.connect:
        vdd_point.outputName = name


def output(corners: List[Point]):
    # Moses = ['pmos', 'pmos-cross', 'nmos', 'nmos-cross']
    Moses = ['nmos-bulk', 'pmos-bulk']

    parent_groups = defaultdict(list)
    for point in corners:
        if point.parent.type == 'vdd':
            iter_rename(point, 'VDD')
            # rename(point, 'VDD')
            point.outputName = 'VDD'
            continue
        elif point.parent.type == 'gnd':
            iter_rename(point, 'GND')
            # rename(point, 'GND')
            point.outputName = 'GND'
            continue
        if point.parent:
            parent_groups[point.parent].append(point)

    result = []
    non_connect_index = 0
    for parent, points in parent_groups.items():
        component_type = parent.classification
        if component_type == 'BasicCircuit':
            continue
        port_connection = {}

        all_port = deepcopy(DeviceType[component_type][parent.type])
        for point in points:
            if point.outputName:
                connection_name = point.outputName
                port_connection[point.name] = connection_name
            else:
                port_connection[point.name] = f'U{non_connect_index}'
                point.outputName = f'U{non_connect_index}'
                non_connect_index += 1
            all_port.remove(point.name)

        if all_port:
            for port in all_port:
                port_connection[port] = f'U{non_connect_index}'
                non_connect_index += 1

        if parent.type in Moses and 'Body' in port_connection and port_connection['Body'].startswith('U'):
            port_connection['Body'] = port_connection['Source']
            for body in filter(lambda x: x.name == 'Body', parent.children):
                body.outputName = port_connection['Body']
        elif parent.type in Moses and 'Body' in port_connection and port_connection['Body'] and port_connection[
            'Source'].startswith('U'):
            port_connection['Source'] = port_connection['Body']
            for body in filter(lambda x: x.name == 'Source', parent.children):
                body.outputName = port_connection['Source']

        result.append({
            'component_type': component_type,
            'port_connection': port_connection
        })

    return result


def parseSingleInput(components, normalize=False):
    _count = 0
    _u = []
    _v = []
    _feature = []
    _map = {}
    for component in components:
        component_feature = key_indices[component['component_type']]
        _feature.append([1, component_feature, 1])
        component_index = _count
        _count += 1
        for port, net in component['port_connection'].items():
            port_feature = DeviceType[component['component_type']]['default'].index(port) + 2
            _feature.append([1, component_feature, port_feature])

            _u.append(component_index)
            _v.append(_count)

            _u.append(_count)
            _count += 1
            if net in _map:
                _v.append(_map[net])
            else:
                _map[net] = _count
                _feature.append([2, 0, 0])
                _v.append(_count)
                _count += 1

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    g = dgl.graph((_u + _v, _v + _u), num_nodes=_count, device=device)
    if normalize:
        g.ndata['h'] = torch.nn.functional.normalize(torch.tensor(_feature, dtype=torch.float32)).to(device)
    else:
        g.ndata['h'] = torch.tensor(_feature, dtype=torch.float32).to(device)

    # return g.to(device), label
    return g


def parseSingleInputHeterogeneous(components, normalize=False):
    component_index = 0
    port_index = 0
    net_index = 0
    _component_features = []
    _port_features = []
    _net_features = []
    _map = {}
    _edges_component_to_port = []
    _edges_port_to_net = []

    for component in components:
        component_feature = key_indices[component['component_type']]
        _component_features.append([component_feature])
        _edges_component_to_port.append((component_index, port_index))
        component_index += 1

        for port, net in component['port_connection'].items():
            port_feature = DeviceType[component['component_type']]['default'].index(port) + 2
            _port_features.append([component_feature, port_feature])
            _edges_component_to_port.append((component_index - 1, port_index))
            port_index += 1

            if net in _map:
                net_idx = _map[net]
                _net_features[net_idx][0] += 1
            else:
                net_idx = net_index
                _map[net] = net_idx
                net_index += 1
                _net_features.append([1])

            _edges_port_to_net.append((port_index - 1, net_idx))

    num_component_nodes = component_index
    num_port_nodes = port_index
    num_net_nodes = net_index

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    g = dgl.heterograph({
        ('component', 'component_to_port', 'port'): _edges_component_to_port,
        ('port', 'port_to_net', 'net'): _edges_port_to_net
    }, num_nodes_dict={
        'component': num_component_nodes,
        'port': num_port_nodes,
        'net': num_net_nodes
    }, device=device)

    if normalize:
        g.nodes['component'].data['h'] = torch.nn.functional.normalize(
            torch.tensor(_component_features, dtype=torch.float32)).to(device)
        g.nodes['port'].data['h'] = torch.nn.functional.normalize(torch.tensor(_port_features, dtype=torch.float32)).to(
            device)
        g.nodes['net'].data['h'] = torch.nn.functional.normalize(torch.tensor(_net_features, dtype=torch.float32)).to(
            device)
    else:
        g.nodes['component'].data['h'] = torch.tensor(_component_features, dtype=torch.float32).to(device)
        g.nodes['port'].data['h'] = torch.tensor(_port_features, dtype=torch.float32).to(device)
        g.nodes['net'].data['h'] = torch.tensor(_net_features, dtype=torch.float32).to(device)

    return g
