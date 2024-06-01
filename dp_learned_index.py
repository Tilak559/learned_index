from opacus import PrivacyEngine 
import torch 
import torch.nn as nn 
from torch.utils.data import DataLoader, Dataset 
import numpy as np 
from sklearn.model_selection import train_test_split 
import torch.optim as optim 
from torch.optim.lr_scheduler import OneCycleLR 
import time
import os


def generate_log_normal_integer_data(size, mean, sigma, max_key):
    log_normal_data = np.random.lognormal(mean, sigma, size)
    integer_data = np.clip(np.round(log_normal_data), 1, max_key).astype(int)
    return integer_data

def normalize_data(data):
    mean = np.mean(data)
    std = np.std(data)
    return (data - mean) / std

def prepare_index_distance_mapping(data):
    sorted_data = np.sort(data)
    unique_values, first_indices = np.unique(sorted_data, return_index=True)
    value_to_first_index = {value: index for value, index in zip(unique_values, first_indices)}
    distances = np.array([value_to_first_index[value] for value in data])
    return data, distances

class KeyValueDataset(Dataset):
    def __init__(self, keys, values):
        self.keys = keys
        self.values = values

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        return torch.tensor([self.keys[idx]], dtype=torch.float32), torch.tensor([self.values[idx]], dtype=torch.float32)

def replace_batchnorm_with_groupnorm(model):
    for child_name, child in model.named_children():
        if isinstance(child, nn.BatchNorm1d):
            setattr(model, child_name, nn.GroupNorm(1, child.num_features))
        elif isinstance(child, nn.BatchNorm2d):
            setattr(model, child_name, nn.GroupNorm(1, child.num_features))
        else:
            replace_batchnorm_with_groupnorm(child)
    return model

def create_model(config, device):
    model = nn.Sequential(
        nn.Linear(1, config['0']),
        nn.ReLU(),
        *[nn.Sequential(
            nn.Linear(config[str(i-1)], config[str(i)]),
            nn.ReLU()
          ) for i in range(1, len(config))],
        nn.Linear(config[str(len(config) - 1)], 1),
        #nn.ReLU()
    )
    model = replace_batchnorm_with_groupnorm(model)  # Replacing BatchNorm with GroupNorm
    return model.to(device)

def train_model(model, dataloader, device, validation_loader, epochs=50, privacy=True, lr=0.0005, max_lr=0.005, epsilon=None, delta=1e-5, accumulation_steps=4):

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  
    scheduler = OneCycleLR(optimizer, max_lr=0.01, steps_per_epoch=len(dataloader)//accumulation_steps, epochs=epochs)

    if privacy:
        privacy_engine = PrivacyEngine()
        model, optimizer, dataloader = privacy_engine.make_private(
            module=model,
            optimizer=optimizer,
            data_loader=dataloader,
            noise_multiplier=2.0,
            max_grad_norm=0.1,
        )

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        model.apply(lambda m: m.train())  # Ensure model is in train mode after eval
        for i, (inputs, labels) in enumerate(dataloader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels.view_as(outputs))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            if (i + 1) % accumulation_steps == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            total_loss += loss.item()
            # if i < 3:  # Debugging output for the first few batches
            #     print(f"Inputs: {inputs[:3]}")
            #     print(f"Outputs: {outputs[:3]}")
            #     print(f"Scaled Labels: {labels[:3]}")
            # print(f"Batch {i+1}, Loss: {loss.item():.4f}")

            total_loss += loss.item() * inputs.size(0)
        
        average_train_loss = total_loss / len(dataloader.dataset)

        avg_error = evaluate_model(model, validation_loader, device)
        print(f"Epoch {epoch+1} - Validation Error: {avg_error:.4f}")
        
        # Validation phase
        model.eval()
        with torch.no_grad():
            validation_loss = 0
            for inputs, labels in validation_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels.view_as(outputs))
                validation_loss += loss.item() * inputs.size(0)
            average_validation_loss = validation_loss / len(validation_loader.dataset)

        # Logging the information
        if privacy:
            epsilon = privacy_engine.get_epsilon(delta=delta)
            print(f"Epoch {epoch + 1}: Train Loss: {average_train_loss:.2f}, Val Loss: {average_validation_loss:.2f}, Epsilon: {epsilon:.2f}")
        else:
            print(f"Epoch {epoch + 1}: Train Loss: {average_train_loss:.2f}, Val Loss: {average_validation_loss:.2f}")



def evaluate_model(model, dataloader, device):
    model.eval()
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
    max_key = 100000
    num_models = 5
    config = {'0': 1200, '1': 1250, '2': 1200}

    data = generate_log_normal_integer_data(data_size, mu, sigma, max_key)
    keys = normalize_data(data)
    keys, labels = prepare_index_distance_mapping(keys)

    train_data, test_data, train_labels, test_labels = train_test_split(keys, labels, test_size=0.2, random_state=42)
    train_dataset = KeyValueDataset(train_data, train_labels)
    test_dataset = KeyValueDataset(test_data, test_labels)

    train_loader = DataLoader(train_dataset, batch_size=150, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=120, shuffle=False, num_workers=4, pin_memory=True)


    models = [create_model(config, device) for _ in range(num_models)]
    for i, model in enumerate(models):
        print(f"Training model {i+1}/{num_models}...")
        train_model(model, train_loader, device, test_loader, epochs=50)
        average_error = evaluate_model(model, test_loader, device)
        print(f"Model {i+1} Average Prediction Error on Test Data: {average_error:.4f}")

    errors = [evaluate_model(m, test_loader, device) for m in models]
    average_error = sum(errors) / len(errors)
    print(f"Average Prediction Error of Ensemble on Test Data: {average_error:.4f}")
    
    model_sizes = [get_model_size(model) for model in models]
    lookup_times = [benchmark_model(model, test_loader, device) for model in models]

    print(f"Config: {config}")
    print(f"Average Size of Models (MB): {np.mean(model_sizes):.2f}")
    print(f"Average Lookup Time (ns): {np.mean(lookup_times):.2f}")


if __name__ == '__main__':
    main()