'''
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
    
    train_size = int(train_rate * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle = True)  # Adjust batch_size as needed
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle = True)  # Adjust batch_size as needed

    return train_loader, test_loader
'''
import os
import torch
from torch_geometric.loader import DataLoader
from torch.utils.data import Dataset
from tqdm.auto import tqdm


class CustomDataset(Dataset):
    def __init__(self, file_list):
        self.file_list = file_list

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        return torch.load(self.file_list[idx])

def load_data(file_path, batch_size, max_data, train_rate, specific_time = "all"):
    files = [os.path.join(file_path, file) for file in os.listdir(file_path) if file.endswith('.pt')]

    if specific_time == "all":
        files.sort(key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))  # Sort files by integer value of filename
        datasets = [[] for _ in range(6)]  # Create 6 groups
        for file in files:
            datasets[int(os.path.splitext(os.path.basename(file))[0]) % 6].append(file)

        train_datasets = []
        test_datasets = []
        for dataset in datasets:
            train_size = int(train_rate * len(dataset))
            train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, len(dataset) - train_size])
            train_datasets.append(CustomDataset(train_dataset))
            test_datasets.append(CustomDataset(test_dataset))

        train_loader = DataLoader(torch.utils.data.ConcatDataset(train_datasets), batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(torch.utils.data.ConcatDataset(test_datasets), batch_size=batch_size, shuffle=True)
    
    elif specific_time == "any":
        import random
        random.shuffle(files)
        dataset = []
        for file in files[:max_data]:
            data = torch.load(file)
            dataset.append(data)
        train_size = int(train_rate * len(dataset))
        test_size = len(dataset) - train_size
        train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle = True)  # Adjust batch_size as needed
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle = True)  # Adjust batch_size as needed
    '''
    if specific_time != "all":
        assert type(specific_time) == int
        files = [file for file in files if int(os.path.splitext(os.path.basename(file))[0]) % 6 == specific_time]
        files = files[:max_data]  # Limit the number of files
    '''
    return train_loader, test_loader

