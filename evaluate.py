import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from autoencoder import Autoencoder
from classifier import LatentClassifier

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

autoencoder = Autoencoder().to(device)
classifier = LatentClassifier().to(device)
autoencoder.load_state_dict(torch.load("autoencoder.pth"))
classifier.load_state_dict(torch.load("classifier.pth"))
autoencoder.eval()
classifier.eval()

transform = transforms.ToTensor()
test_data = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

recon_loss_fn = nn.MSELoss()
class_loss_fn = nn.CrossEntropyLoss()

total, correct, recon_loss_total, class_loss_total = 0, 0, 0.0, 0.0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        recon, latent = autoencoder(images)
        outputs = classifier(latent)

        recon_loss = recon_loss_fn(recon, images)
        class_loss = class_loss_fn(outputs, labels)

        recon_loss_total += recon_loss.item() * images.size(0)
        class_loss_total += class_loss.item() * images.size(0)

        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

print(f"Reconstruction Loss: {recon_loss_total/len(test_loader.dataset):.4f}")
print(f"Classification Loss: {class_loss_total/len(test_loader.dataset):.4f}")
print(f"Classification Accuracy: {correct/total:.4f}")
