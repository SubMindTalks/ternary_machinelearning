import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np

class TernaryMNISTNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        # Convolutional layers for feature extraction
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.25)
        )
        
        # Fully connected layers for classification
        self.fc_layers = nn.Sequential(
            nn.Linear(64 * 5 * 5, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 10),  # 10 digits
            nn.Softmax(dim=1)
        )
        
        # Uncertainty estimation branch
        self.uncertainty = nn.Sequential(
            nn.Linear(64 * 5 * 5, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # Feature extraction
        features = self.conv_layers(x)
        features_flat = features.view(features.size(0), -1)
        
        # Main classification branch
        logits = self.fc_layers(features_flat)
        
        # Uncertainty estimation
        uncertainty = self.uncertainty(features_flat)
        
        return logits, uncertainty

class TernaryMNISTTrainer:
    def __init__(self, model, learning_rate=0.001):
        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        self.criterion = nn.CrossEntropyLoss()
        
    def train_epoch(self, train_loader, device):
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            self.optimizer.zero_grad()
            logits, uncertainty = self.model(data)
            
            # Classification loss
            loss = self.criterion(logits, target)
            
            # Add uncertainty regularization
            pred = logits.argmax(dim=1)
            error = (pred != target).float()
            uncertainty_loss = F.mse_loss(uncertainty.squeeze(), error)
            
            total_loss = loss + 0.1 * uncertainty_loss
            
            total_loss.backward()
            self.optimizer.step()
            
            # Calculate accuracy
            pred = logits.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
            
        return total_loss.item(), correct / total
    
    def evaluate(self, test_loader, device):
        self.model.eval()
        test_loss = 0
        correct = 0
        total = 0
        uncertainties = []
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                logits, uncertainty = self.model(data)
                
                # Sum up batch loss
                test_loss += self.criterion(logits, target).item()
                
                # Calculate accuracy
                pred = logits.argmax(dim=1)
                correct += pred.eq(target).sum().item()
                total += target.size(0)
                
                # Store uncertainties
                uncertainties.extend(uncertainty.cpu().numpy())
        
        return test_loss / len(test_loader), correct / total, np.mean(uncertainties)

def compare_models():
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load MNIST dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    # Initialize models
    ternary_model = TernaryMNISTNetwork().to(device)
    ternary_trainer = TernaryMNISTTrainer(ternary_model)
    
    # Training loop
    num_epochs = 10
    ternary_results = {
        'train_acc': [],
        'test_acc': [],
        'uncertainty': []
    }
    
    print("Training Ternary Model...")
    for epoch in range(num_epochs):
        # Train
        train_loss, train_acc = ternary_trainer.train_epoch(train_loader, device)
        
        # Evaluate
        test_loss, test_acc, uncertainty = ternary_trainer.evaluate(test_loader, device)
        
        ternary_results['train_acc'].append(train_acc)
        ternary_results['test_acc'].append(test_acc)
        ternary_results['uncertainty'].append(uncertainty)
        
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}, Uncertainty: {uncertainty:.4f}')
    
    return ternary_results

def analyze_uncertain_predictions(model, test_loader, device):
    model.eval()
    high_uncertainty_samples = []
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            logits, uncertainty = model(data)
            
            # Find samples with high uncertainty
            high_unc_mask = uncertainty.squeeze() > 0.5
            if high_unc_mask.any():
                high_uncertainty_samples.extend([
                    (data[i], target[i], uncertainty[i].item())
                    for i in range(len(data)) if high_unc_mask[i]
                ])
    
    return high_uncertainty_samples

if __name__ == "__main__":
    results = compare_models()
    print("\nFinal Results:")
    print(f"Final Test Accuracy: {results['test_acc'][-1]:.4f}")
    print(f"Average Uncertainty: {np.mean(results['uncertainty']):.4f}")

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns

class StandardMNISTNetwork(nn.Module):
    """Traditional CNN for MNIST classification"""
    def __init__(self):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.25)
        )
        
        self.fc_layers = nn.Sequential(
            nn.Linear(64 * 5 * 5, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 10),
            nn.Softmax(dim=1)
        )
    
    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        return self.fc_layers(x)

class ModelComparison:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.prepare_data()
        
    def prepare_data(self):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        self.train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
        self.test_dataset = datasets.MNIST('./data', train=False, transform=transform)
        
        self.train_loader = DataLoader(self.train_dataset, batch_size=64, shuffle=True)
        self.test_loader = DataLoader(self.test_dataset, batch_size=64, shuffle=False)
    
    def train_standard_model(self, num_epochs=10):
        model = StandardMNISTNetwork().to(self.device)
        optimizer = torch.optim.Adam(model.parameters())
        criterion = nn.CrossEntropyLoss()
        
        results = {
            'train_acc': [],
            'test_acc': [],
            'train_loss': [],
            'test_loss': []
        }
        
        for epoch in range(num_epochs):
            # Training
            model.train()
            train_loss = 0
            correct = 0
            total = 0
            
            for data, target in self.train_loader:
                data, target = data.to(self.device), target.to(self.device)
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
                total += target.size(0)
            
            train_acc = correct / total
            
            # Testing
            model.eval()
            test_loss = 0
            correct = 0
            total = 0
            
            with torch.no_grad():
                for data, target in self.test_loader:
                    data, target = data.to(self.device), target.to(self.device)
                    output = model(data)
                    test_loss += criterion(output, target).item()
                    pred = output.argmax(dim=1)
                    correct += pred.eq(target).sum().item()
                    total += target.size(0)
            
            test_acc = correct / total
            
            results['train_acc'].append(train_acc)
            results['test_acc'].append(test_acc)
            results['train_loss'].append(train_loss / len(self.train_loader))
            results['test_loss'].append(test_loss / len(self.test_loader))
            
            print(f'Epoch {epoch+1}/{num_epochs}:')
            print(f'Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')
        
        return model, results
    
    def compare_models(self, num_epochs=10):
        # Train standard model
        print("Training Standard Model...")
        standard_model, standard_results = self.train_standard_model(num_epochs)
        
        # Train ternary model
        print("\nTraining Ternary Model...")
        ternary_model = TernaryMNISTNetwork().to(self.device)
        ternary_trainer = TernaryMNISTTrainer(ternary_model)
        ternary_results = {
            'train_acc': [],
            'test_acc': [],
            'uncertainty': []
        }
        
        for epoch in range(num_epochs):
            train_loss, train_acc = ternary_trainer.train_epoch(self.train_loader, self.device)
            test_loss, test_acc, uncertainty = ternary_trainer.evaluate(self.test_loader, self.device)
            
            ternary_results['train_acc'].append(train_acc)
            ternary_results['test_acc'].append(test_acc)
            ternary_results['uncertainty'].append(uncertainty)
            
            print(f'Epoch {epoch+1}/{num_epochs}:')
            print(f'Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}, Uncertainty: {uncertainty:.4f}')
        
        return standard_results, ternary_results
    
    def plot_comparison(self, standard_results, ternary_results):
        plt.figure(figsize=(15, 5))
        
        # Plot training accuracy
        plt.subplot(1, 2, 1)
        plt.plot(standard_results['train_acc'], label='Standard - Train')
        plt.plot(standard_results['test_acc'], label='Standard - Test')
        plt.plot(ternary_results['train_acc'], label='Ternary - Train')
        plt.plot(ternary_results['test_acc'], label='Ternary - Test')
        plt.title('Accuracy Comparison')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        
        # Plot uncertainty for ternary model
        plt.subplot(1, 2, 2)
        plt.plot(ternary_results['uncertainty'], label='Ternary Uncertainty')
        plt.title('Ternary Model Uncertainty')
        plt.xlabel('Epoch')
        plt.ylabel('Average Uncertainty')
        plt.legend()
        
        plt.tight_layout()
        plt.show()

def main():
    comparison = ModelComparison()
    standard_results, ternary_results = comparison.compare_models()
    comparison.plot_comparison(standard_results, ternary_results)
    
    print("\nFinal Results:")
    print(f"Standard Model Final Test Accuracy: {standard_results['test_acc'][-1]:.4f}")
    print(f"Ternary Model Final Test Accuracy: {ternary_results['test_acc'][-1]:.4f}")
    print(f"Ternary Model Final Uncertainty: {ternary_results['uncertainty'][-1]:.4f}")

if __name__ == "__main__":
    main()
