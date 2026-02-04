import flwr as fl
import torch
from model import QuantumTrainGenerator, TargetCNN
from data_utils import load_local_dataset

DEVICE = "cpu"

class FLClient(fl.client.NumPyClient):
    def __init__(self):
        self.model = QuantumTrainGenerator(TargetCNN()).to(DEVICE)
        self.loader = load_local_dataset("dataset")

    def get_parameters(self, config=None):
        return [p.detach().cpu().numpy() for p in self.model.parameters()]

    def set_parameters(self, params):
        for p, new in zip(self.model.parameters(), params):
            p.data = torch.tensor(new)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        opt = torch.optim.Adam(self.model.parameters(), lr=0.01)
        loss_fn = torch.nn.CrossEntropyLoss()
        cnn = TargetCNN().to(DEVICE)

        for data, labels in self.loader:
            opt.zero_grad()
            gen_params = self.model()
            offset = 0
            for p in cnn.parameters():
                n = p.numel()
                p.data.copy_(gen_params[offset:offset+n].view(p.size()))
                offset += n
            loss = loss_fn(cnn(data), labels)
            loss.backward()
            opt.step()

        return self.get_parameters(), len(self.loader.dataset), {}

fl.client.start_numpy_client(
    server_address="192.168.1.10:8080",  # CHANGE TO SERVER IP
    client=FLClient()
)
