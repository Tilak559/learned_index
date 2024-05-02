import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import OneCycleLR
import numpy as np
from sklearn.model_selection import train_test_split
from opacus import PrivacyEngine

def generate_log_normal_integer_data(size, mean, sigma, max_key):
    log_normal_data = np.random.lognormal(mean, sigma, size)
    integer_data = np.clip(np.round(log_normal_data), 1, max_key).astype(int)
    return integer_data

def normalize_data(data):
    return (data - np.mean(data)) / np.std(data)

def scale_labels(labels):
    return labels / 1000000.0  # Example: Scaling labels down to a similar range as outputs

def log_transform_labels(labels):
    return np.log1p(labels)  # log1p ensures that log(0) does not cause an error


def log_scale_labels(labels):
    return torch.log(labels + 1)  # Adds 1 to labels to ensure there are no log(0) errors

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
        if isinstance(child, nn.BatchNorm1d) or isinstance(child, nn.BatchNorm2d):
            num_channels = child.num_features
            setattr(model, child_name, nn.GroupNorm(1, num_channels))
        else:
            replace_batchnorm_with_groupnorm(child)
    return model

def create_model(config, device):
    model = nn.Sequential(
        nn.Linear(1, config['0']),
        nn.LeakyReLU(0.2),
        *[nn.Sequential(
            nn.Linear(config[str(i-1)], config[str(i)]),
            nn.LeakyReLU(0.2)
          ) for i in range(1, len(config))],
        nn.Linear(config[str(len(config) - 1)], 1),
        nn.LeakyReLU(0.2)
    )
    model = replace_batchnorm_with_groupnorm(model)  # Replacing BatchNorm with GroupNorm
    return model.to(device)



def train_model(model, dataloader, device, validation_loader, epochs=5, privacy=True, lr=0.0001, max_lr=0.001, delta=1e-5, accumulation_steps=4):
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.01)
    scheduler = OneCycleLR(optimizer, max_lr=max_lr, total_steps=epochs * len(dataloader) // accumulation_steps)

    if privacy:
        from opacus import PrivacyEngine
        privacy_engine = PrivacyEngine()
        model, optimizer, dataloader = privacy_engine.make_private(
            module=model,
            optimizer=optimizer,
            data_loader=dataloader,
            noise_multiplier=2,
            max_grad_norm=1.5
        )

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for i, (inputs, labels) in enumerate(dataloader):
            inputs = inputs.to(device)
            labels = log_scale_labels(labels).to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
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

            average_train_loss = total_loss / len(dataloader.dataset)
        #print(f"Epoch {epoch + 1}, Average Train Loss: {average_train_loss:.4f}")

        if privacy:
            epsilon = privacy_engine.get_epsilon(delta=delta)
            print(f"Epoch {epoch + 1}: Total Loss: {total_loss:.6f}, Epsilon: {epsilon:.4f}, Train Loss: {average_train_loss:.6f}")


    model.eval()
    with torch.no_grad():
        validation_loss = 0
        for inputs, labels in validation_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels.view_as(outputs))
            validation_loss += loss.item()
        average_validation_loss = validation_loss / len(validation_loader.dataset)
        print(f"Epoch {epoch + 1}, Average Validation Loss: {average_validation_loss:.4f}")



    
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

 

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Available GPU: ", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU available")
    data_size = 1000000
    mu = 5
    sigma = 0.7
    max_key = 1000000
    num_models = 5
    config = {'0': 300, '1': 350, '2': 300}

    data = generate_log_normal_integer_data(data_size, mu, sigma, max_key)
    keys = normalize_data(data)
    labels = log_transform_labels(keys)
    labels = scale_labels(labels)
    keys, labels = prepare_index_distance_mapping(keys)

    train_data, test_data, train_labels, test_labels = train_test_split(keys, labels, test_size=0.2, random_state=42)
    train_dataset = KeyValueDataset(train_data, train_labels)
    test_dataset = KeyValueDataset(test_data, test_labels)

    train_loader = DataLoader(train_dataset, batch_size=3000, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=2800, shuffle=False, num_workers=4, pin_memory=True)

    models = [create_model(config, device) for _ in range(num_models)]
    for i, model in enumerate(models):
        print(f"Training model {i+1}/{num_models}...")
        train_model(model, train_loader, device, test_loader, epochs=5)
        average_error = evaluate_model(model, test_loader, device)
        print(f"Model {i+1} Average Prediction Error on Test Data: {average_error:.4f}")
        # model.eval()
        # with torch.no_grad():
        #     validation_loss = 0
        #     for inputs, labels in test_loader:
        #         inputs, labels = inputs.to(device), labels.to(device)
        #         outputs = model(inputs)
        #         loss = nn.MSELoss()(outputs, labels.view_as(outputs))
        #         validation_loss += loss.item()
        #     average_validation_loss = validation_loss / len(test_loader.dataset)
        #     print(f"Model {i+1} Average Validation Loss: {average_validation_loss:.4f}")

if __name__ == '__main__':
    main()


