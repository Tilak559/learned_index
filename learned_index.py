import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
from sklearn.model_selection import train_test_split
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim.lr_scheduler import OneCycleLR
import time
import os

def generate_log_normal_integer_data(size, mean, sigma, max_key):
    """Generates data following a log-normal distribution and converts it to integer values."""
    log_normal_data = np.random.lognormal(mean, sigma, size)
    integer_data = np.clip(np.round(log_normal_data), 1, max_key).astype(int)
    return integer_data

def normalize_data(data):
    """Normalizes data to have zero mean and unit variance."""
    mean = np.mean(data)
    std = np.std(data)
    return (data - mean) / std

def prepare_index_distance_mapping(data):
    """Creates a mapping from data values to their first occurrence index, which is used as labels for training."""
    sorted_data = np.sort(data)
    unique_values, first_indices = np.unique(sorted_data, return_index=True)
    value_to_first_index = {value: index for value, index in zip(unique_values, first_indices)}
    distances = np.array([value_to_first_index[value] for value in data])
    return data, distances

def distribute_keys_to_models(key, num_models, max_key):
    """Distributes keys across models based on intervals, determining which model should predict for the given key."""
    interval_size = max_key // num_models
    model_index = min(key // interval_size, num_models - 1)
    return model_index

class KeyValueDataset(Dataset):
    """Custom dataset for handling key-value pairs where keys are inputs and values are targets for the models."""
    def __init__(self, keys, values):
        self.keys = keys
        self.values = values

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        return torch.tensor([self.keys[idx]], dtype=torch.float32), torch.tensor([self.values[idx]], dtype=torch.float32)

def create_model(config, device):
    layers = [nn.Linear(1, config['0']), nn.LeakyReLU(0.2)]
    for i in range(1, len(config)):
        layers.append(nn.Linear(config[str(i-1)], config[str(i)]))
        layers.append(nn.LeakyReLU(0.2))
    layers.append(nn.Linear(config[str(len(config) - 1)], 1))
    layers.append(nn.LeakyReLU(0.2))
    model = nn.Sequential(*layers)
    return model.to(device)

def train_model(model, dataloader, device, validation_loader, epochs=50):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # You can adjust the learning rate as necessary
    scheduler = OneCycleLR(optimizer, max_lr=0.01, steps_per_epoch=len(dataloader), epochs=epochs)
    
    model.train()
    for epoch in range(epochs):
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels.view(-1, 1))
            loss.backward()
            optimizer.step()
            scheduler.step()  # Update the learning rate after each batch
        
        # Evaluate and print every few epochs
        if epoch % 10 == 0 or epoch == epochs-1:
            avg_error = evaluate_model(model, validation_loader, device)
            print(f"Epoch {epoch+1} - Validation Error: {avg_error:.4f}")


def evaluate_model(model, dataloader, device):
    model.eval()  # Ensure the model is in evaluation mode
    total_error = 0
    count = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            prediction = model(inputs).view(-1)
            error = torch.abs(prediction - labels.view(-1))
            total_error += error.sum().item()
            count += len(inputs)
    return total_error / count if count != 0 else 0

def get_model_size(model):
    """Calculates the file size of the model's state dictionary, providing an estimate of its memory footprint."""
    torch.save(model.state_dict(), 'temp_model.pt')
    model_size = os.path.getsize('temp_model.pt') / (1024 * 1024)
    os.remove('temp_model.pt')
    return model_size

def benchmark_model(model, dataloader, device):
    """Measures the time it takes for the model to process the entire test dataset."""
    start_time = time.time()
    for inputs, _ in dataloader:
        inputs = inputs.to(device)
        with torch.no_grad():
            _ = model(inputs)
    end_time = time.time()
    lookup_time_ns = (end_time - start_time) / len(dataloader.dataset) * 1e9
    return lookup_time_ns

def timeit(func):
    """Decorator to measure the execution time of functions."""
    def timed(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Execution time: {execution_time:.2f} seconds")
        return result
    return timed

@timeit
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Available GPU: ", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU available")

    data_size = 1000000
    mu = 5
    sigma = 0.7
    max_key = 1000000
    num_models = 5
    config = {'0': 500, '1': 550,'2': 500}

    data = generate_log_normal_integer_data(data_size, mu, sigma, max_key)
    keys = normalize_data(data)
    keys, labels = prepare_index_distance_mapping(keys)

    train_data, test_data, train_labels, test_labels = train_test_split(keys, labels, test_size=0.2, random_state=42)

    train_dataset = KeyValueDataset(train_data, train_labels)
    test_dataset = KeyValueDataset(test_data, test_labels)

    train_loader = DataLoader(train_dataset, batch_size=150, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=120, shuffle=False)

    models = [create_model(config, device) for _ in range(num_models)]

    for i, model in enumerate(models):
        print(f"Training model {i+1}/{num_models}...")
        train_model(model, train_loader, device, test_loader, epochs=50)
        average_error = evaluate_model(model, test_loader, device)
        print(f"Model {i+1} Average Prediction Error on Test Data: {average_error:.4f}")

    errors = [evaluate_model(m, test_loader, device) for m in models]
    average_error = sum(errors) / len(errors)
    print(f"Average Prediction Error of Ensemble on Test Data: {average_error:.4f}")
    # Post-training model evaluation and metrics...
    model_sizes = [get_model_size(model) for model in models]
    lookup_times = [benchmark_model(model, test_loader, device) for model in models]

    print(f"Config: {config}")
    print(f"Average Size of Models (MB): {np.mean(model_sizes):.2f}")
    print(f"Average Lookup Time (ns): {np.mean(lookup_times):.2f}")

if __name__ == '__main__':
    main()