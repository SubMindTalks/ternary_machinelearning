import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os
from models import TernaryMNISTNetwork, StandardMNISTNetwork
import pandas as pd

class ModelComparator:
    def __init__(self, model_dir, results_dir):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_dir = model_dir
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)
        self.prepare_data()
        self.load_models()

    def prepare_data(self):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        self.test_dataset = datasets.MNIST('./data', train=False, transform=transform)
        self.test_loader = DataLoader(self.test_dataset, batch_size=100, shuffle=False)

    # In compare.py, update the load_models method:
    def load_models(self):
        self.standard_model = StandardMNISTNetwork().to(self.device)
        self.ternary_model = TernaryMNISTNetwork().to(self.device)

        self.standard_model.load_state_dict(
            torch.load(os.path.join(self.model_dir, 'standard_model.pth'), weights_only=True)
        )
        self.ternary_model.load_state_dict(
            torch.load(os.path.join(self.model_dir, 'ternary_model.pth'), weights_only=True)
        )

        self.standard_model.eval()
        self.ternary_model.eval()

    def evaluate_models(self):
        standard_preds = []
        standard_labels = []
        ternary_preds = []
        ternary_labels = []
        uncertainties = []
        
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                
                # Standard model predictions
                standard_output = self.standard_model(data)
                standard_pred = standard_output.argmax(dim=1)
                standard_preds.extend(standard_pred.cpu().numpy())
                standard_labels.extend(target.cpu().numpy())
                
                # Ternary model predictions
                ternary_output, uncertainty = self.ternary_model(data)
                ternary_pred = ternary_output.argmax(dim=1)
                ternary_preds.extend(ternary_pred.cpu().numpy())
                ternary_labels.extend(target.cpu().numpy())
                uncertainties.extend(uncertainty.cpu().numpy())
        
        return {
            'standard': {
                'predictions': np.array(standard_preds),
                'labels': np.array(standard_labels)
            },
            'ternary': {
                'predictions': np.array(ternary_preds),
                'labels': np.array(ternary_labels),
                'uncertainties': np.array(uncertainties).flatten()
            }
        }

    def plot_confusion_matrices(self, results):
        plt.figure(figsize=(20, 8))
        
        # Standard model confusion matrix
        plt.subplot(1, 2, 1)
        cm_standard = confusion_matrix(
            results['standard']['labels'],
            results['standard']['predictions']
        )
        sns.heatmap(cm_standard, annot=True, fmt='d', cmap='Blues')
        plt.title('Standard Model Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        
        # Ternary model confusion matrix
        plt.subplot(1, 2, 2)
        cm_ternary = confusion_matrix(
            results['ternary']['labels'],
            results['ternary']['predictions']
        )
        sns.heatmap(cm_ternary, annot=True, fmt='d', cmap='Blues')
        plt.title('Ternary Model Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'confusion_matrices.png'))
        plt.close()

    def analyze_uncertainty(self, results):
        predictions = results['ternary']['predictions']
        labels = results['ternary']['labels']
        uncertainties = results['ternary']['uncertainties']
        
        # Analyze uncertainty for correct vs incorrect predictions
        correct_mask = predictions == labels
        incorrect_mask = ~correct_mask
        
        uncertainty_analysis = {
            'correct_mean': uncertainties[correct_mask].mean(),
            'correct_std': uncertainties[correct_mask].std(),
            'incorrect_mean': uncertainties[incorrect_mask].mean(),
            'incorrect_std': uncertainties[incorrect_mask].std()
        }
        
        # Plot uncertainty distribution
        plt.figure(figsize=(10, 6))
        plt.hist(uncertainties[correct_mask], alpha=0.5, label='Correct Predictions',
                bins=50, density=True)
        plt.hist(uncertainties[incorrect_mask], alpha=0.5, label='Incorrect Predictions',
                bins=50, density=True)
        plt.xlabel('Uncertainty')
        plt.ylabel('Density')
        plt.title('Uncertainty Distribution for Correct vs Incorrect Predictions')
        plt.legend()
        plt.savefig(os.path.join(self.results_dir, 'uncertainty_distribution.png'))
        plt.close()
        
        return uncertainty_analysis

    def analyze_digit_pairs(self, results):
        pairs = [(4, 9), (3, 8), (1, 7), (0, 1)]
        pair_analysis = {}
        
        for d1, d2 in pairs:
            mask = (results['ternary']['labels'] == d1) | (results['ternary']['labels'] == d2)
            uncertainties = results['ternary']['uncertainties'][mask]
            predictions = results['ternary']['predictions'][mask]
            labels = results['ternary']['labels'][mask]
            
            pair_analysis[f'{d1}vs{d2}'] = {
                'mean_uncertainty': uncertainties.mean(),
                'accuracy': (predictions == labels).mean(),
                'num_samples': len(uncertainties)
            }
        
        return pair_analysis

    def save_results(self, results):
        standard_acc = (results['standard']['predictions'] == results['standard']['labels']).mean()
        ternary_acc = (results['ternary']['predictions'] == results['ternary']['labels']).mean()
        
        # Generate classification reports
        standard_report = classification_report(
            results['standard']['labels'],
            results['standard']['predictions'],
            output_dict=True
        )
        ternary_report = classification_report(
            results['ternary']['labels'],
            results['ternary']['predictions'],
            output_dict=True
        )
        
        uncertainty_analysis = self.analyze_uncertainty(results)
        digit_pair_analysis = self.analyze_digit_pairs(results)
        
        # Save all results
        final_results = {
            'accuracy': {
                'standard': standard_acc,
                'ternary': ternary_acc
            },
            'uncertainty_analysis': uncertainty_analysis,
            'digit_pair_analysis': digit_pair_analysis
        }
        
        # Save as text file
        with open(os.path.join(self.results_dir, 'comparison_results.txt'), 'w') as f:
            f.write("Model Comparison Results\n")
            f.write("=======================\n\n")
            f.write(f"Standard Model Accuracy: {standard_acc:.4f}\n")
            f.write(f"Ternary Model Accuracy: {ternary_acc:.4f}\n\n")
            
            f.write("Uncertainty Analysis\n")
            f.write("-----------------\n")
            for k, v in uncertainty_analysis.items():
                f.write(f"{k}: {v:.4f}\n")
            
            f.write("\nDigit Pair Analysis\n")
            f.write("-----------------\n")
            for pair, analysis in digit_pair_analysis.items():
                f.write(f"{pair}:\n")
                for k, v in analysis.items():
                    f.write(f"  {k}: {v:.4f}\n")

def main():
    parser = argparse.ArgumentParser(description='Compare MNIST Models')
    parser.add_argument('--model-dir', type=str, default='./models',
                        help='Directory containing trained models')
    parser.add_argument('--results-dir', type=str, default='./results',
                        help='Directory to save comparison results')
    
    args = parser.parse_args()
    
    comparator = ModelComparator(args.model_dir, args.results_dir)
    results = comparator.evaluate_models()
    
    comparator.plot_confusion_matrices(results)
    comparator.save_results(results)
    
    print("Comparison complete. Results saved in", args.results_dir)

if __name__ == "__main__":
    main()
