import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from src.dataset.dataset import DeepfakeDataset
from src.models.videomae import VideoMAEClassifier

def visualize_tsne():
    os.makedirs("outputs/visualizations", exist_ok=True)
    device = torch.device("cuda:6" if torch.cuda.is_available() else "cpu")
    
    # Load model
    model = VideoMAEClassifier().to(device)
    model.load_state_dict(torch.load("outputs/checkpoints/best_model.pth", map_location=device))
    model.eval()
    
    # Load dataset
    dataset = DeepfakeDataset("data/faces", multi_view=False)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    _, val_set = random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))
    
    val_loader = DataLoader(val_set, batch_size=8, num_workers=4)
    
    all_embeddings = []
    all_labels = []
    
    print("Extracting embeddings for t-SNE...")
    with torch.no_grad():
        for x, y in tqdm(val_loader):
            x = x.to(device)
            # Forward pass: get logits and embeddings
            _, embeddings = model(x)
            
            all_embeddings.append(embeddings.cpu().numpy())
            all_labels.append(y.numpy())
            
    all_embeddings = np.concatenate(all_embeddings, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    
    print(f"Running t-SNE on {len(all_embeddings)} samples...")
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    embeddings_2d = tsne.fit_transform(all_embeddings)
    
    # Plotting
    plt.figure(figsize=(10, 8))
    
    # Convert labels 0->Real, 1->Fake
    label_names = ["Real" if l == 0 else "Fake" for l in all_labels]
    
    sns.scatterplot(
        x=embeddings_2d[:, 0], 
        y=embeddings_2d[:, 1], 
        hue=label_names,
        palette={"Real": "blue", "Fake": "red"},
        alpha=0.8,
        s=100
    )
    
    plt.title("t-SNE Visualization of Domain SSL Embeddings", fontsize=16)
    plt.savefig("outputs/visualizations/tsne_plot.png", dpi=300)
    plt.close()
    
    print("t-SNE plot saved in outputs/visualizations/tsne_plot.png")

if __name__ == "__main__":
    visualize_tsne()
