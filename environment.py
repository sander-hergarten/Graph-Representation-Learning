import gymnasium
from gymnasium.spaces import Dict, Box, Tuple, Space, Sequence
import numpy as np
import rustworkx as rx
from typing import TypeVar, Any, TypedDict 
import torch
from torch import nn

class ActType(TypedDict):
    edges: tuple[torch.Tensor, torch.Tensor]
    embedd: torch.Tensor

ObsType = TypeVar("ActType")
RewardType = TypeVar("RewardType")
Terminated = bool
Truncated = bool
Info = dict[str, Any]
Done = bool

CollectionBuffer = list[tuple[rx.PyDiGraph, list[torch.Tensor]]]

def get_edge_list_data(graph:rx.PyDiGraph):
    edge_list_indices = graph.edge_list()
    return[(graph[node_0], graph[node_1]) for node_0, node_1 in edge_list_indices]


def _quotient_graph_edges(partition, graph):
    nodes = np.zeros(len(graph.node_indices(0)))
    for block in partition:
        for index in block:
            nodes[index] = graph[block[-1]]


    quotient_graph = rx.PyDiGraph()
    quotient_graph.add_nodes_from(nodes)
    quotient_graph.add_edges_from_no_data(graph.edge_list())


    for block in partition:
        for node_index in block[:-1]:
            quotient_graph.merge(node_index, block[-1])

    
    edge_list = get_edge_list_data(quotient_graph) 

    return edge_list
        

class Decoder(nn.module):
    def __init__(self, recurrent_layer_size, input_size, output_size):
        super().__init__()
        self.hidden_initializer = lambda batch_size: torch.randn()
        self.gru_cell = nn.GRUCell(input_size, recurrent_layer_size)
        self.linear = nn.Linear(recurrent_layer_size, output_size)
        self.relu = nn.ReLU()
    
    def forward(self, input):
        hidden = self.hidden_initializer(1)
        for element in input:
            hidden = self.gru_cell(element, hidden)

        embedd = self.relu(self.linear(hidden))

        return embedd


class GraphLearnerEnv(gymnasium.env):
    graph: rx.PyDigraph
    decoder: Decoder
    record: list[CollectionBuffer]

    def __init__(self, node_data_size, embedding_size, render_mode, decoder_learning_rate = 1e-3):
        super().__init__()
        mse = nn.MSELoss()


        # ?
        self.input_shape= node_data_size
        self.loss_function = lambda y_true, y_pred: torch.sqrt(mse[y_true, y_pred])
        #?/
        self.decoder = Decoder(64, node_data_size, embedding_size)
        self.optimizer = torch.optim.Adam(self.decoder.parameters(), lr=decoder_learning_rate)

        self.node_data_size = node_data_size
        self.embedding_size = embedding_size
        self.render_mode = render_mode

    @property
    def action_space(self) -> Space[ActType]:
        node_data_space = Box(low=0, high=1, shape=self.node_data_size)
        edge_data_space = Tuple((node_data_space, node_data_space))
        return Dict({
            "edges": edge_data_space,
            "embedd": Box(low=-np.inf, high = np.inf, shape=self.embedding_size)
        })

    @property
    def observation_space(self) -> Space[ObsType]:
        node_data_space = Box(low=0, high=1, shape=self.node_data_size)
        edge_data_space = Tuple((node_data_space, node_data_space))
        edge_sequence_space = Sequence(edge_data_space)

        return edge_sequence_space 

    def metadata(self) -> dict[str, Any]:
        return {"render_modes": ["human"]}

    def step(self, action: ActType) -> tuple[ObsType, RewardType, Terminated, Truncated, Info]:
                
        nodes = []
        for node_payload in action["edges"]:
            if not (node:=self.graph.find_node_by_weight(node_payload)):
                node = self.graph.add_node(node_payload)

            nodes.append(node)

        self.graph.add_edge(nodes[0], node[1], None)

        partition = rx.digraph_maximum_bisimulation(self.graph)
        edge_list = _quotient_graph_edges(partition, self.graph) 

        embedding = self.decoder(edge_list)
        loss = self.loss_function(action["embedd"], embedding)

        reward = 1/loss
        observation = get_edge_list_data(self.graph)
        terminated = False
        truncated = False
        info = {"predicted_embedding": embedding}

        self.record[-1].append((self.graph, action["embedd"]))

        return (observation, reward, terminated, truncated, info)
    
    def train_decoder(self, data, collection:int = -1):
        collection_buffer = self.record[collection]

        for sample in collection_buffer:
            edge_list = get_edge_list_data(sample[0])
            result = self.decoder(edge_list)
            loss = self.loss_function(sample[1], result)
            loss.backward()

            self.optimizer.step()

        return loss 


    def add_collection_buffer(self):
        self.record.append([])
        
        
    def reset(self):
        super().reset(seed=None)
        self.graph = rx.PyDiGraph()        

        return ([], {})
    

    def render(self):
        raise NotImplementedError() 

