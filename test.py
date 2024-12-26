from models import TernaryMNISTNetwork, StandardMNISTNetwork
from torchvision import transforms
from PIL import Image
import torch

def test_single_image(image_path, model_path):
    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    image = Image.open(image_path)
    input_tensor = transform(image).unsqueeze(0)
    
    model = TernaryMNISTNetwork()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    with torch.no_grad():
        prediction, uncertainty = model(input_tensor)
    
    return prediction.argmax().item(), uncertainty.item()

# Usage
if __name__ == "__main__":
    digit, conf = test_single_image("test_digit.png", "models/ternary_model.pth")
    print(f"Predicted: {digit}, Uncertainty: {conf:.4f}")
