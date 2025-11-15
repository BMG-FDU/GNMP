from .model import SimpleResidualGAT,ResidualGAT, SymmetricalResidualGAT
from .dataset import load_data
from torch.optim import Adam
import torch
import torch.nn as nn
from tqdm.auto import tqdm

class MAPELoss(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, y_pred, y_true):
        return torch.mean(torch.abs((y_true - y_pred) / (y_true + 1e-8))) * 100

class network_diffusion_trainer:
    def __init__(self, 
                 data_file:dict, 
                 model_parameter:dict, 
                 train_parameter:dict,
                 model_type:str):
        self.model_type = model_type
        self.train_loader, self.test_loader = load_data(file_path = data_file["file_path"], 
                                                        batch_size = data_file["batch_size"],
                                                        max_data = data_file["max_data"],
                                                        train_rate = train_parameter["train_rate"],
                                                        specific_time = data_file["time_step"])
        print(f"Train:\t{len(self.train_loader)} pieces")
        print(f"Test :\t{len(self.test_loader)} pieces")
        self.model = self.model_initialize(model_parameter) # A torch model
        self.train_parameter = train_parameter
        self.optimizer = Adam(self.model.parameters(), lr=self.train_parameter["lr"])
        self.criterion = torch.nn.L1Loss()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.batch_size = data_file["batch_size"]
        print("Example data")
    def write_model_parameter_into_txt(self,data_file, model_parameter, train_parameter, file_path):
        with open(file_path, 'w') as f:
            f.write("data_file:\n")
            for key, value in data_file.items():
                f.write(f"{key}: {value}\n")
            
            f.write("\nmodel_parameter:\n")
            for key, value in model_parameter.items():
                f.write(f"{key}: {value}\n")
            
            f.write("\ntrain_parameter:\n")
            for key, value in train_parameter.items():
                f.write(f"{key}: {value}\n")

            f.write("\nmodel_type_:\n")
            f.write(f"{self.model_type}")       
    def model_initialize(self, model_parameter):
        if self.model_type == "ResidualGAT":
            model = ResidualGAT(hid_features = model_parameter["hid_features"],
                                    num_node_features = model_parameter["num_node_features"],
                                    num_output_features = model_parameter["num_output_features"],
                                    num_heads = model_parameter["num_heads"],
                                    num_layers = model_parameter["num_layers"],
                                    dropout = model_parameter["dropout"])
        elif self.model_type == "SimpleResidualGAT":
            model = SimpleResidualGAT(hid_features = model_parameter["hid_features"],
                                      num_node_features = model_parameter["num_node_features"],
                                      num_output_features = model_parameter["num_output_features"],
                                      num_heads = model_parameter["num_heads"],
                                      num_layers = model_parameter["num_layers"],
                                      dropout = model_parameter["dropout"])
        elif self.model_type == "SymmetricalResidualGAT":
            model = SymmetricalResidualGAT(hid_features = model_parameter["hid_features"],
                                      num_node_features = model_parameter["num_node_features"],
                                      num_output_features = model_parameter["num_output_features"],
                                      num_heads = model_parameter["num_heads"],
                                      num_layers = model_parameter["num_layers"],
                                      dropout = model_parameter["dropout"])
        else:
            print("Unrecognized model type.")
            model = 0
        return model
    
    def train(self):
        self.model = self.model.to(self.device)
        self.model.train()
        self.history = {}
        train_history = []
        test_history = []
        for epoch in tqdm(range(self.train_parameter["epoch"])):
            total_loss = 0
            for batch in self.train_loader:
                batch = batch.to(self.device)
                self.optimizer.zero_grad()
                out = self.model(batch)
                target = batch.y - batch.x[:,:2]
                loss = self.criterion(out, target)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()

                # Detach and delete tensors
                batch.detach()
                out.detach()
                target.detach()
                del batch
                del out
                del target

            avg_loss = total_loss / len(self.train_loader) / self.batch_size
            train_history.append(avg_loss)

            # Test the model
            self.model.eval()
            with torch.no_grad():
                test_loss = 0
                for batch in self.test_loader:
                    batch = batch.to(self.device)
                    out = self.model(batch)
                    target = batch.y - batch.x[:,:2]
                    loss = self.criterion(out, target)
                    test_loss += loss.item()

                    # Detach and delete tensors
                    batch.detach()
                    out.detach()
                    target.detach()
                    del batch
                    del out
                    del target

                avg_test_loss = test_loss / len(self.test_loader) / self.batch_size
                test_history.append(avg_test_loss)

            # Switch back to train mode
            self.model.train()

            # Empty the cache
            torch.cuda.empty_cache()
        self.history["train"] = train_history
        self.history["test"] = test_history