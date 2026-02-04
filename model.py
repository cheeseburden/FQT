import torch
import torch.nn as nn
import pennylane as qml

# Quantum config
N_QUBITS = 17
N_LAYERS = 2
dev = qml.device("default.qubit", wires=N_QUBITS)

@qml.qnode(dev, interface="torch")
def qnn_circuit(inputs, weights):
    qml.AngleEmbedding(inputs, wires=range(N_QUBITS))
    qml.BasicEntanglerLayers(weights, wires=range(N_QUBITS))
    return [qml.expval(qml.PauliZ(i)) for i in range(N_QUBITS)]

class TargetCNN(nn.Module):
    def __init__(self, n_mfcc=20, max_len=100):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, 3, padding=1)
        self.conv2 = nn.Conv2d(8, 16, 3, padding=1)
        self.pool = nn.MaxPool2d(2,2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

        h = (n_mfcc//2)//2
        w = (max_len//2)//2
        self.fc1 = nn.Linear(16*h*w, 64)
        self.fc2 = nn.Linear(64, 2)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.dropout(self.relu(self.fc1(x)))
        return self.fc2(x)

class QuantumTrainGenerator(nn.Module):
    def __init__(self, target_model):
        super().__init__()
        self.num_params = sum(p.numel() for p in target_model.parameters())
        self.qnn_weights = nn.Parameter(0.01*torch.randn(N_LAYERS, N_QUBITS))
        self.mapper = nn.Sequential(
            nn.Linear(N_QUBITS, 128),
            nn.ReLU(),
            nn.Linear(128, self.num_params)
        )

    def forward(self):
        qnn_input = torch.zeros(N_QUBITS)
        q_out = qnn_circuit(qnn_input, self.qnn_weights)
        q_out = torch.tensor(q_out, dtype=torch.float32)
        return self.mapper(q_out)
