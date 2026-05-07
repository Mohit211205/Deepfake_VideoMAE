from src.dataset.dataset import DeepfakeDataset

dataset = DeepfakeDataset("data/faces")

print("Total samples:", len(dataset))

x, y = dataset[0]

print("Shape:", x.shape)   # expect (16, 3, 224, 224)
print("Label:", y)
