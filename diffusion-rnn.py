import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from main import URBAN_MIX_CSV, URBAN_CORE_CSV, ADJ_URBAN_CORE_CSV, ADJ_URBAN_MIX_CSV
import dgl
import torch
import torch.nn as nn
import torch.optim as optim


def show_dataset(adj_file_name, file_name):
    df = pd.read_csv(file_name, header=None) 
    df = df.drop(df.columns[0:7], axis=1).reset_index(drop=True)

    adj_df = pd.read_csv(adj_file_name, header=None) 

    adjacency_np = adj_df.to_numpy()
    traffic_speed_data = df.to_numpy()
    
    # Create a graph
    G = nx.from_numpy_array(adjacency_np)

    # Create a directed graph without self-loops
    subgraph_no_selfloops = nx.DiGraph(G)

    # Create a directed graph without self-loops
    for node in subgraph_no_selfloops.nodes():
        if (node, node) in subgraph_no_selfloops.edges():
            subgraph_no_selfloops.remove_edge(node, node)

    num_timesteps = len(traffic_speed_data[0])
    for t in range(num_timesteps):
        speeds = [speed[t] for speed in traffic_speed_data]

        max_speed = max(speeds)
        min_speed = min(speeds)

        # Normalize the speed data for each time step and segment
        norm_speeds = [(speed - min_speed) / (max_speed - min_speed) for speed in speeds]

        # Set node attributes for the current time step
        for i, node in enumerate(subgraph_no_selfloops.nodes()):
            nx.set_node_attributes(subgraph_no_selfloops, {node: {'speed': norm_speeds[i]}}, name='speed')

        # Draw the graph for the current time step
        plt.figure(figsize=(8, 6))
        plt.title(f"Graph with Traffic Speed at Time Step {t}")
        pos = nx.spring_layout(subgraph_no_selfloops)
        
        # Extract normalized speed values for node colors
        node_colors = [subgraph_no_selfloops.nodes[node]['speed'] for node in subgraph_no_selfloops]
        
        # Convert dictionary elements to scalar values for node colors
        node_colors = [val['speed'] for val in node_colors]
        
        nx.draw_networkx(subgraph_no_selfloops, pos, with_labels=True, node_size=500, node_color=node_colors, cmap=plt.cm.plasma)
        plt.colorbar(plt.cm.ScalarMappable(cmap=plt.cm.plasma))
        plt.show()

class DiffusionConvGRU(nn.Module):
    def __init__(self, in_feats, hidden_feats, num_layers, num_hops):
        super(DiffusionConvGRU, self).__init__()
        self.num_layers = num_layers
        self.num_hops = num_hops

        # Graph convolutional layer
        self.graph_conv = dgl.nn.GraphConv(in_feats, hidden_feats)

        # GRU cell
        self.gru = nn.GRU(hidden_feats, hidden_feats, num_layers=num_layers, batch_first=True)

        # ReLU activation
        self.relu = nn.ReLU()

    def forward(self, graphs, features):
        output = []

        for i, graph in enumerate(graphs):
            node_feats = features[i]
            graph = graph.to('cpu')

            # Apply multi-hop diffusion
            for _ in range(self.num_hops):
                node_feats = self.graph_conv(graph, node_feats)
                graph = dgl.add_self_loop(graph)  # Adding self-loop to capture local information

            # GRU cell
            _, h_n = self.gru(node_feats.unsqueeze(0))  # Assuming batch size is 1
            node_feats = self.relu(h_n[-1])  # Applying ReLU activation

            output.append(node_feats)

        return torch.stack(output, dim=0)

class GraphGeneratorDecoder(nn.Module):
    def __init__(self, in_feats, out_feats):
        super(GraphGeneratorDecoder, self).__init__()

        # Linear layer for generating graph features
        self.linear = nn.Linear(in_feats, out_feats)

    def forward(self, encoded_outputs):
        # Decoder (linear layer)
        decoded_outputs = self.linear(encoded_outputs[-1])  # Using the last hidden state

        return decoded_outputs

class EncoderDecoderModel(nn.Module):
    def __init__(self, in_feats, hidden_feats, out_feats, num_layers, num_hops):
        super(EncoderDecoderModel, self).__init__()

        # Diffusion Convolutional GRU as the encoder
        self.encoder = DiffusionConvGRU(in_feats, hidden_feats, num_layers, num_hops)

        # Graph generator decoder
        self.decoder = GraphGeneratorDecoder(hidden_feats, out_feats)

    def forward(self, input_graphs, input_features):
        # Encoder
        encoder_outputs = self.encoder(input_graphs, input_features)

        # Decoder
        decoded_outputs = self.decoder(encoder_outputs)

        return decoded_outputs

# Function to convert adjacency matrix and speed features to DGLGraph and features
def convert_to_dgl_data(adj_file_name, file_name):
    graphs = []
    features = []

    df = pd.read_csv(file_name, header=None) 
    df = df.drop(df.columns[0:7], axis=1).reset_index(drop=True)

    adj_df = pd.read_csv(adj_file_name, header=None) 

    adjacency_np = adj_df.to_numpy()
    traffic_speed_data = df.to_numpy()
    traffic_speed_data = np.transpose(traffic_speed_data)

    # Find the non-zero entries in the adjacency matrix
    src_nodes, dst_nodes = np.where(adjacency_np != 0)

    # road segment level
    for i, datas in enumerate(traffic_speed_data):
        graph = dgl.graph((src_nodes, dst_nodes))
        graphs.append(graph)

        node_features = torch.tensor(datas, dtype=torch.float32).view(-1, 1)
        features.append(node_features)

    return graphs, features

if __name__ == "__main__":
    # show_dataset(ADJ_URBAN_MIX_CSV, URBAN_MIX_CSV)

    graphs, features = convert_to_dgl_data(ADJ_URBAN_CORE_CSV, URBAN_CORE_CSV)

    # Create the Encoder-Decoder model
    encoder_decoder_model = EncoderDecoderModel(in_feats=304, hidden_feats=128, out_feats=1, num_layers=2, num_hops=2)

    # Forward pass through the model
    outputs = encoder_decoder_model(graphs, features)

    # Print the outputs (just for demonstration)
    print(outputs)

    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    num_epochs = 10
    for epoch in range(num_epochs):
        for batch in dataloader:
            input_graphs, input_features, targets = batch

            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(input_graphs, input_features)

            # Compute the loss
            loss = criterion(outputs, targets)

            # Backward pass
            loss.backward()

            # Update model parameters
            optimizer.step()

        # Optionally, print or log the training loss for the epoch
        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}')