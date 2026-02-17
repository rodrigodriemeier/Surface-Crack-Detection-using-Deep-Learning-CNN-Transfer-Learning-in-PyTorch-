import random
from pathlib import Path

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageFolder
from torchvision.transforms import InterpolationMode
import torch.nn as nn
import torch.nn.functional as F

from utils import stratified_train_test_indices, make_train_val_split_from_train_indices
from model import SimpleCNN, SimpleCNNDropout
from evaluation import (
    train_one_epoch,
    evaluate,
    collect_outputs,
    confusion_from_probs,
    metrics_from_confusion,
    train_epochs
)


BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_PATH = BASE_DIR / "data"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", DEVICE)

SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

dataset_raw = ImageFolder(root=DATASET_PATH)
print("Classes:", dataset_raw.classes)
print("Total images:", len(dataset_raw))

targets = dataset_raw.targets
class_counts = torch.bincount(torch.tensor(targets))
print("Class counts (raw):", class_counts.tolist())

train_ratio = 0.8
train_indices, test_indices = stratified_train_test_indices(targets, train_ratio, SEED)

train_transform = transforms.Compose([
    transforms.Resize((224, 224), interpolation=InterpolationMode.BILINEAR),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
])

test_transform = transforms.Compose([
    transforms.Resize((224, 224), interpolation=InterpolationMode.BILINEAR),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
])

dataset_train_full = ImageFolder(root=DATASET_PATH, transform=train_transform)
dataset_test_full = ImageFolder(root=DATASET_PATH, transform=test_transform)

train_dataset = Subset(dataset_train_full, train_indices)
test_dataset = Subset(dataset_test_full, test_indices)

print("Train size:", len(train_dataset))
print("Test size:", len(test_dataset))

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0, pin_memory=False)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=0, pin_memory=False)

x, y = next(iter(train_loader))
print("Batch images shape:", tuple(x.shape))
print("Batch labels shape:", tuple(y.shape))
print("Batch label sample:", y[:10].tolist())

train_counts = torch.bincount(y, minlength=len(dataset_raw.classes))
print("Batch class counts (train sample batch):", train_counts.tolist())

model = SimpleCNN().to(DEVICE)

x, y = next(iter(train_loader))
x = x.to(DEVICE)

output = model(x)

print("Input shape:", tuple(x.shape))
print("Output shape:", tuple(output.shape))

loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

train_loss, train_acc = train_one_epoch(model, train_loader, loss_fn, optimizer, DEVICE)
test_loss, test_acc = evaluate(model, test_loader, loss_fn, DEVICE)

print("Train loss:", train_loss)
print("Train acc:", train_acc)
print("Test loss:", test_loss)
print("Test acc:", test_acc)

probs_test, targets_test = collect_outputs(model, test_loader, DEVICE)

for thr in [0.3, 0.5, 0.7]:
    tp, tn, fp, fn = confusion_from_probs(probs_test, targets_test, threshold=thr)
    acc, precision, recall, f1 = metrics_from_confusion(tp, tn, fp, fn)

    print("Threshold:", thr)
    print("Confusion [TN FP; FN TP]:", [[tn, fp], [fn, tp]])
    print("Acc:", acc)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1:", f1)

VAL_RATIO = 0.1
SEED = 42

train_indices2, val_indices = make_train_val_split_from_train_indices(train_indices, targets, VAL_RATIO, SEED)

train_dataset2 = Subset(dataset_train_full, train_indices2)
val_dataset = Subset(dataset_train_full, val_indices)

train_loader2 = DataLoader(train_dataset2, batch_size=32, shuffle=True, num_workers=0, pin_memory=False)
val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=0, pin_memory=False)

model_reg = SimpleCNNDropout(dropout_p=0.3).to(DEVICE)

loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model_reg.parameters(), lr=1e-3, weight_decay=1e-4)

history_reg = train_epochs(
    model_reg,
    train_loader2,
    val_loader,
    loss_fn,
    optimizer,
    DEVICE,
    epochs=1,
    patience=3
)

test_loss_reg, test_acc_reg = evaluate(model_reg, test_loader, loss_fn, DEVICE)

print("Regularized test loss:", test_loss_reg)
print("Regularized test acc:", test_acc_reg)

model_tl = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)

in_features = model_tl.fc.in_features
model_tl.fc = nn.Linear(in_features, 1)

for p in model_tl.parameters():
    p.requires_grad = False

for p in model_tl.fc.parameters():
    p.requires_grad = True

model_tl = model_tl.to(DEVICE)

loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model_tl.fc.parameters(), lr=1e-3)

history_tl = train_epochs(
    model_tl,
    train_loader2,
    val_loader,
    loss_fn,
    optimizer,
    DEVICE,
    epochs=3,
    patience=2
)

test_loss_tl, test_acc_tl = evaluate(model_tl, test_loader, loss_fn, DEVICE)

print("Transfer (frozen backbone) test loss:", test_loss_tl)
print("Transfer (frozen backbone) test acc:", test_acc_tl)

for p in model_tl.layer4.parameters():
    p.requires_grad = True

optimizer_ft = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model_tl.parameters()),
    lr=1e-4
)

history_ft = train_epochs(
    model_tl,
    train_loader2,
    val_loader,
    loss_fn,
    optimizer_ft,
    DEVICE,
    epochs=5,
    patience=2
)

test_loss_ft, test_acc_ft = evaluate(model_tl, test_loader, loss_fn, DEVICE)

print("Fine-tuned test loss:", test_loss_ft)
print("Fine-tuned test acc:", test_acc_ft)
