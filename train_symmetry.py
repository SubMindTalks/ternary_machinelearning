import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import torchvision.transforms.functional as TF
from PIL import Image
import numpy as np
from models import TernaryMNISTNetwork, StandardMNISTNetwork, TernaryMNISTTrainer
import argparse
import os


class AugmentedMNIST(Dataset):
    def __init__(self, original_dataset, transform_type='all'):
        self.original_dataset = original_dataset
        self.transform_type = transform_type

    def __len__(self):
        return len(self.original_dataset)

    def __getitem__(self, idx):
        image, label = self.original_dataset[idx]

        # Convert tensor to PIL for transformations
        if isinstance(image, torch.Tensor):
            image = TF.to_pil_image(image)

        if self.transform_type == 'mirror':
            transformed_image = TF.hflip(image)
        elif self.transform_type == 'flip':
            transformed_image = TF.vflip(image)
        elif self.transform_type == 'both':
            transformed_image = TF.hflip(TF.vflip(image))
        else:
            transformed_image = image

        # Convert back to tensor
        transformed_image = TF.to_tensor(transformed_image)
        return transformed_image, label


def create_augmented_dataset():
    # Original transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # Load original dataset
    original_train = datasets.MNIST('./data', train=True, download=True, transform=transform)
    original_test = datasets.MNIST('./data', train=False, download=True, transform=transform)

    # Create augmented versions
    mirrored_train = AugmentedMNIST(original_train, 'mirror')
    flipped_train = AugmentedMNIST(original_train, 'flip')
    both_train = AugmentedMNIST(original_train, 'both')

    # Combine datasets
    combined_train = ConcatDataset([
        original_train,
        mirrored_train,
        flipped_train,
        both_train
    ])

    return combined_train, original_test


def train_model(model_type, train_loader, test_loader, device, num_epochs=10):
    if model_type == 'ternary':
        model = TernaryMNISTNetwork().to(device)
        trainer = TernaryMNISTTrainer(model)

        results = {
            'train_acc': [],
            'test_acc': [],
            'uncertainty': []
        }

        for epoch in range(num_epochs):
            train_loss, train_acc = trainer.train_epoch(train_loader, device)
            test_loss, test_acc, uncertainty = trainer.evaluate(test_loader, device)

            results['train_acc'].append(train_acc)
            results['test_acc'].append(test_acc)
            results['uncertainty'].append(uncertainty)

            print(f'Epoch {epoch + 1}/{num_epochs}:')
            print(f'Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}, '
                  f'Uncertainty: {uncertainty:.4f}')

    else:  # standard model
        model = StandardMNISTNetwork().to(device)
        optimizer = torch.optim.Adam(model.parameters())
        criterion = torch.nn.CrossEntropyLoss()

        results = {
            'train_acc': [],
            'test_acc': []
        }

        for epoch in range(num_epochs):
            # Training
            model.train()
            correct = 0
            total = 0

            for data, target in train_loader:
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()

                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
                total += target.size(0)

            train_acc = correct / total

            # Testing
            model.eval()
            correct = 0
            total = 0

            with torch.no_grad():
                for data, target in test_loader:
                    data, target = data.to(device), target.to(device)
                    output = model(data)
                    pred = output.argmax(dim=1)
                    correct += pred.eq(target).sum().item()
                    total += target.size(0)

            test_acc = correct / total

            results['train_acc'].append(train_acc)
            results['test_acc'].append(test_acc)

            print(f'Epoch {epoch + 1}/{num_epochs}:')
            print(f'Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')

    return model, results


def analyze_symmetry_performance(model, test_loader, device, transform_type):
    model.eval()
    correct = 0
    total = 0
    uncertainties = [] if isinstance(model, TernaryMNISTNetwork) else None

    with torch.no_grad():
        for data, target in test_loader:
            # Apply transformation
            if transform_type == 'mirror':
                data = TF.hflip(data)
            elif transform_type == 'flip':
                data = TF.vflip(data)
            elif transform_type == 'both':
                data = TF.hflip(TF.vflip(data))

            data, target = data.to(device), target.to(device)

            if isinstance(model, TernaryMNISTNetwork):
                output, uncertainty = model(data)
                uncertainties.extend(uncertainty.cpu().numpy())
            else:
                output = model(data)

            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)

    accuracy = correct / total
    return accuracy, np.mean(uncertainties) if uncertainties else None


def main():
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create augmented dataset
    train_dataset, test_dataset = create_augmented_dataset()
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # Train both models
    print("Training Standard Model...")
    standard_model, standard_results = train_model('standard', train_loader, test_loader, device)

    print("\nTraining Ternary Model...")
    ternary_model, ternary_results = train_model('ternary', train_loader, test_loader, device)

    # Analyze performance on different transformations
    transformations = ['original', 'mirror', 'flip', 'both']
    results = {
        'standard': {},
        'ternary': {},
        'ternary_uncertainty': {}
    }

    for transform in transformations:
        # Standard model performance
        acc, _ = analyze_symmetry_performance(standard_model, test_loader, device, transform)
        results['standard'][transform] = acc

        # Ternary model performance
        acc, unc = analyze_symmetry_performance(ternary_model, test_loader, device, transform)
        results['ternary'][transform] = acc
        results['ternary_uncertainty'][transform] = unc

    # Save results
    print("\nResults on Different Transformations:")
    print("=====================================")
    for transform in transformations:
        print(f"\n{transform.capitalize()} Transform:")
        print(f"Standard Model Accuracy: {results['standard'][transform]:.4f}")
        print(f"Ternary Model Accuracy: {results['ternary'][transform]:.4f}")
        if transform in results['ternary_uncertainty']:
            print(f"Ternary Model Uncertainty: {results['ternary_uncertainty'][transform]:.4f}")


if __name__ == "__main__":
    main()