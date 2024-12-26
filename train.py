import argparse
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from models import TernaryMNISTNetwork, StandardMNISTNetwork, TernaryMNISTTrainer
import os


def train(args):
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Data loading
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    # Model initialization
    if args.model == 'ternary':
        model = TernaryMNISTNetwork().to(device)
        trainer = TernaryMNISTTrainer(model, args.lr)
    else:
        model = StandardMNISTNetwork().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        criterion = torch.nn.CrossEntropyLoss()
    # Training loop
    for epoch in range(args.epochs):
        if args.model == 'ternary':
            train_loss, train_acc = trainer.train_epoch(train_loader, device)
            test_loss, test_acc, uncertainty = trainer.evaluate(test_loader, device)
            print(f'Epoch {epoch + 1}/{args.epochs}:')
            print(f'Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}, Uncertainty: {uncertainty:.4f}')
        else:
            # Standard model training
            model.train()
            for data, target in train_loader:
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
    # Save model
    os.makedirs(args.save_dir, exist_ok=True)
    model_path = os.path.join(args.save_dir, f'{args.model}_model.pth')
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")


def main():
    parser = argparse.ArgumentParser(description='Train MNIST Models')
    parser.add_argument('--model', type=str, default='ternary', choices=['ternary', 'standard'],
                        help='Model type to train')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of epochs to train')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Input batch size')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--save-dir', type=str, default='./models',
                        help='Directory to save models')
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()