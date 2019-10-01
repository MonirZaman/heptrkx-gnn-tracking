"""
This module implements the PyTorch modules that define the sparse
message-passing graph neural networks for segment classification.
In particular, this implementation utilizes the pytorch_geometric
and supporting libraries:
https://github.com/rusty1s/pytorch_geometric
"""

# Externals
import torch
import torch.nn as nn
from torch_scatter import scatter_add

# Locals
from .utils import make_mlp

class EdgeNetwork(nn.Module):
    """
    A module which computes weights for edges of the graph.
    For each edge, it selects the associated nodes' features
    and applies some fully-connected network layers with a final
    sigmoid activation.
    """
    def __init__(self, input_dim, hidden_dim=8, hidden_activation=nn.Tanh,
                 layer_norm=True):
        super(EdgeNetwork, self).__init__()
        self.network = make_mlp(input_dim*2,
                                [hidden_dim, hidden_dim, hidden_dim, 1],
                                hidden_activation=hidden_activation,
                                output_activation=None,
                                layer_norm=layer_norm)

    def forward(self, x, edge_index):
        # Select the features of the associated nodes
        start, end = edge_index
        print("x")
        print(len(x))
        print(x)
        print(edge_index, start, end)
        x1, x2 = x[start], x[end]
        edge_inputs = torch.cat([x[start], x[end]], dim=1)
        return self.network(edge_inputs).squeeze(-1)

class NodeNetwork(nn.Module):
    """
    A module which computes new node features on the graph.
    For each node, it aggregates the neighbor node features
    (separately on the input and output side), and combines
    them with the node's previous features in a fully-connected
    network to compute the new features.
    """
    def __init__(self, input_dim, output_dim, hidden_activation=nn.Tanh,
                 layer_norm=True):
        super(NodeNetwork, self).__init__()
        self.network = make_mlp(input_dim*3, [output_dim]*4,
                                hidden_activation=hidden_activation,
                                output_activation=hidden_activation,
                                layer_norm=layer_norm)

    def forward(self, x, e, edge_index):
        start, end = edge_index
        # Aggregate edge-weighted incoming/outgoing features
        mi = scatter_add(e[:, None] * x[start], end, dim=0, dim_size=x.shape[0])
        mo = scatter_add(e[:, None] * x[end], start, dim=0, dim_size=x.shape[0])
        node_inputs = torch.cat([mi, mo, x], dim=1)
        return self.network(node_inputs)

class GNNSegmentClassifier(nn.Module):
    """
    Segment classification graph neural network model.
    Consists of an input network, an edge network, and a node network.
    """
    def __init__(self, input_dim=3, hidden_dim=8, n_graph_iters=3,
                 hidden_activation=nn.Tanh, layer_norm=True):
        super(GNNSegmentClassifier, self).__init__()
        self.n_graph_iters = n_graph_iters
        
        self.input_dim=input_dim 
        self.hidden_dim=hidden_dim
        self.n_graph_iters=n_graph_iters
        self.hidden_activation=hidden_activation
        self.layer_norm=layer_norm
        
        # Setup the input network
        self.input_network = make_mlp(input_dim, [hidden_dim],
                                      output_activation=hidden_activation,
                                      layer_norm=layer_norm)
        # Setup the edge network
        self.edge_network = EdgeNetwork(input_dim+hidden_dim, hidden_dim,
                                        hidden_activation, layer_norm=layer_norm)
        # Setup the node layers
        self.node_network = NodeNetwork(input_dim+hidden_dim, hidden_dim,
                                        hidden_activation, layer_norm=layer_norm)

    
    def get_attention_score(self, inputs):

        input_network = make_mlp(self.input_dim, [self.hidden_dim],
                                  output_activation=self.hidden_activation,
                                  layer_norm=self.layer_norm)
        # Setup the edge network
        edge_network = EdgeNetwork(self.input_dim + self.hidden_dim, self.hidden_dim,
                                    self.hidden_activation, layer_norm=self.layer_norm)
        # Setup the node layers
        node_network = NodeNetwork(self.input_dim + self.hidden_dim, self.hidden_dim,
                                    self.hidden_activation, layer_norm=self.layer_norm)
        
        print("before input network")
        print("len(inputs.x) ", len(inputs.x))

        # Apply input network to get hidden representation
        x = input_network(inputs.x)
        
        print("after input net ", len(x), len(inputs.x))



        # Shortcut connect the inputs onto the hidden representation
        x = torch.cat([x, inputs.x], dim=-1)
        
        print("after cat ",len(x), len(inputs.x))

        # Loop over iterations of edge and node networks
        for i in range(self.n_graph_iters):
            # Apply edge network
            e = torch.sigmoid(edge_network(x, inputs.edge_index))
            
            print("after edge network ", len(x), len(inputs.x))

            # Apply node network
            x = node_network(x, e, inputs.edge_index)
            print("after node network ", len(x), len(inputs.x))
            
            # Shortcut connect the inputs onto the hidden representation
            x = torch.cat([x, inputs.x], dim=-1)
            print("after cat ", len(x), len(inputs.x))
        
        return x

    def forward(self, inputs):

        heads = []
        num_heads = 2
        for nh in range(num_heads):
            heads.append(self.get_attention_score(inputs)) 

        print(heads)
        x = torch.mean(torch.stack(heads))
        print("after mean ")
        print(len(x))
        return self.edge_network(x, inputs.edge_index)

    def forward_l(self, inputs):
        """Apply forward pass of the model"""
        # Apply input network to get hidden representation
        x = self.input_network(inputs.x)
        # Shortcut connect the inputs onto the hidden representation
        x = torch.cat([x, inputs.x], dim=-1)
        # Loop over iterations of edge and node networks
        for i in range(self.n_graph_iters):
            # Apply edge network
            e = torch.sigmoid(self.edge_network(x, inputs.edge_index))
            # Apply node network
            x = self.node_network(x, e, inputs.edge_index)
            # Shortcut connect the inputs onto the hidden representation
            x = torch.cat([x, inputs.x], dim=-1)
        # Apply final edge network
        return self.edge_network(x, inputs.edge_index)

