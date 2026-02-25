import flwr as fl
import torch
import torch.nn as nn
import numpy as np
from model import QuantumTrainGenerator, TargetCNN
from data_utils import load_local_dataset

# ==============================
# CONFIGURATION
# ==============================

SERVER_IP = "192.168.1.10"   # CHANGE THIS
SERVER_PORT = "8080"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LOCAL_EPOCHS = 1
LEARNING_RATE = 0.01

print("=" * 60)
print("Federated Client Starting...")
print(f"Device: {DEVICE}")
print(f"Connecting to: {SERVER_IP}:{SERVER_PORT}")
print("=" * 60)


# ==============================
# FEDERATED CLIENT CLASS
# ==============================

class FLClient(fl.client.NumPyClient):

    def __init__(self):
        print("Initializing model and loading local dataset...")

        self.generator = QuantumTrainGenerator(TargetCNN()).to(DEVICE)
        self.train_loader = load_local_dataset("dataset")

        print(f"Local dataset size: {len(self.train_loader.dataset)} samples")
        print("Initialization complete.\n")

    # --------------------------------
    # Send model parameters to server
    # --------------------------------
    def get_parameters(self, config=None):
        print("Server requested parameters → Sending current parameters.")
        return [p.detach().cpu().numpy() for p in self.generator.parameters()]

    # --------------------------------
    # Receive global parameters
    # --------------------------------
    def set_parameters(self, parameters):
        print("Receiving global parameters from server...")
        for param, new_param in zip(self.generator.parameters(), parameters):
            param.data = torch.tensor(new_param, dtype=param.dtype).to(DEVICE)
        print("Global parameters loaded.\n")

    # --------------------------------
    # Local Training
    # --------------------------------
    def fit(self, parameters, config):
        print("\n" + "-" * 50)
        print("FIT ROUND STARTED")
        print("-" * 50)

        self.set_parameters(parameters)

        optimizer = torch.optim.Adam(self.generator.parameters(), lr=LEARNING_RATE)
        criterion = nn.CrossEntropyLoss()

        total_loss = 0.0
        batch_count = 0

        self.generator.train()

        for epoch in range(LOCAL_EPOCHS):
            print(f"\nEpoch {epoch+1}/{LOCAL_EPOCHS}")

            for batch_idx, (data, labels) in enumerate(self.train_loader):
                data = data.to(DEVICE)
                labels = labels.to(DEVICE)

                optimizer.zero_grad()

                # Generate CNN parameters from quantum model
                generated_params = self.generator()

                # Build CNN dynamically
                cnn_model = TargetCNN().to(DEVICE)

                # Set CNN weights
                offset = 0
                for param in cnn_model.parameters():
                    numel = param.numel()
                    param.data.copy_(
                        generated_params[offset:offset+numel].view(param.size())
                    )
                    offset += numel

                outputs = cnn_model(data)
                loss = criterion(outputs, labels)

                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                batch_count += 1

                print(f"Batch {batch_idx+1}/{len(self.train_loader)} | Loss: {loss.item():.4f}")

        avg_loss = total_loss / batch_count
        print("\nLocal training complete.")
        print(f"Average Loss: {avg_loss:.4f}")
        print("Sending updated parameters back to server.")
        print("-" * 50 + "\n")

        return (
            [p.detach().cpu().numpy() for p in self.generator.parameters()],
            len(self.train_loader.dataset),
            {"loss": avg_loss},
        )

    # --------------------------------
    # Evaluation (Optional)
    # --------------------------------
    def evaluate(self, parameters, config):
        print("Evaluation requested (skipping evaluation).")
        return 0.0, len(self.train_loader.dataset), {}


# ==============================
# START CLIENT
# ==============================

fl.client.start_numpy_client(
    server_address=f"{SERVER_IP}:{SERVER_PORT}",
    client=FLClient(),
)
