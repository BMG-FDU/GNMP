import torch
import torch_geometric.nn as pyg_nn

class SymmetricalResidualGAT(torch.nn.Module):
    def __init__(self, num_node_features, hid_features, num_output_features, num_heads=2, num_layers=2, dropout=0.1):
        super(SymmetricalResidualGAT, self).__init__()
        self.edge_conv = pyg_nn.EdgeConv(nn=torch.nn.Sequential(
            torch.nn.Linear(2 * num_node_features, hid_features),
            torch.nn.ReLU(),
            torch.nn.Linear(hid_features, hid_features)
        ))
        self.conv1 = pyg_nn.GATConv(hid_features, hid_features, heads=num_heads, dropout=dropout)
        self.convs = torch.nn.ModuleList([
            pyg_nn.GATConv(hid_features * num_heads, hid_features, heads=num_heads, dropout=dropout)
            for _ in range(num_layers - 1)
        ])
        self.regressor = torch.nn.Linear(hid_features * num_heads, hid_features * num_heads)
        self.convs_reverse = torch.nn.ModuleList([
            pyg_nn.GATConv(hid_features * num_heads, hid_features, heads=num_heads, dropout=dropout)
            for _ in range(num_layers - 1)
        ])
        self.conv1_reverse = pyg_nn.GATConv(hid_features * num_heads, hid_features, heads=num_heads, dropout=dropout)
        self.final_regressor = torch.nn.Linear(hid_features * num_heads, num_output_features)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        x = self.edge_conv(x, edge_index)
        x = self.conv1(x, edge_index, edge_attr)
        x = torch.relu(x)
        for conv in self.convs:
            residual = x
            x = conv(x, edge_index, edge_attr)
            x = torch.relu(x) + residual
        x = self.regressor(x)
        for conv in reversed(self.convs_reverse):
            residual = x
            x = conv(x, edge_index, edge_attr)
            x = torch.relu(x) + residual
        x = self.conv1_reverse(x, edge_index, edge_attr)
        x = torch.relu(x)
        x = self.final_regressor(x)
        return x


class ResidualGAT(torch.nn.Module):
    def __init__(self, num_node_features, hid_features, num_output_features, num_heads=2, num_layers=2, dropout=0.1):
        super(ResidualGAT, self).__init__()
        self.edge_conv = pyg_nn.EdgeConv(nn=torch.nn.Sequential(
            torch.nn.Linear(2 * num_node_features, hid_features),
            torch.nn.ReLU(),
            torch.nn.Linear(hid_features, hid_features)
        ))
        self.conv1 = pyg_nn.GATConv(hid_features, hid_features, heads=num_heads, dropout=dropout)
        self.convs = torch.nn.ModuleList([
            pyg_nn.GATConv(hid_features * num_heads, hid_features, heads=num_heads, dropout=dropout)
            for _ in range(num_layers - 1)
        ])
        self.regressor = torch.nn.Linear(hid_features * num_heads, num_output_features)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        x = self.edge_conv(x, edge_index)
        x = self.conv1(x, edge_index, edge_attr)
        x = torch.relu(x)
        for conv in self.convs:
            residual = x
            x = conv(x, edge_index, edge_attr)
            x = torch.relu(x) + residual
        x = self.regressor(x)
        return x


class SimpleResidualGAT(torch.nn.Module):
    def __init__(self, num_node_features, hid_features, num_output_features, num_heads=2, num_layers=2, dropout=0.1):
        super(SimpleResidualGAT, self).__init__()
        self.conv1 = pyg_nn.GATConv(num_node_features, hid_features, heads=num_heads, dropout=dropout)
        self.convs = torch.nn.ModuleList([
            pyg_nn.GATConv(hid_features * num_heads, hid_features, heads=num_heads, dropout=dropout)
            for _ in range(num_layers - 1)
        ])
        self.regressor = torch.nn.Linear(hid_features * num_heads, num_output_features)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        for conv in self.convs:
            residual = x
            x = conv(x, edge_index)
            x = torch.relu(x) + residual
        x = self.regressor(x)
        return x