import os
import torch
from torch_geometric.loader import DataLoader
import random
from tqdm.auto import tqdm

def load_data(file_path, batch_size, max_data, train_rate, specific_time = "false"):
    files = os.listdir(file_path)
    random.shuffle(files)
    dataset = []
    if specific_time == "false":
        for file in tqdm(files[:max_data]):
            if file.endswith('.pt'):
                data = torch.load(os.path.join(file_path, file))
                dataset.append(data)
    else:
        assert type(specific_time) == int
        # If the part of the file name other than ".pt" 
        # becomes an integer and the remainder divided by 6 is equal to "specific_time", 
        # load the data and put it into the "dataset ".
        count = 0
        for file in tqdm(files):
            if int(file[:-3]) % 6 == specific_time:
                data = torch.load(os.path.join(file_path, file))
                dataset.append(data)
                count += 1
            if count == max_data:
                break

    dataset = torch.utils.data.ConcatDataset(dataset)
    train_size = int(train_rate * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle = True)  # Adjust batch_size as needed
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle = True)  # Adjust batch_size as needed

    return train_loader, test_loader