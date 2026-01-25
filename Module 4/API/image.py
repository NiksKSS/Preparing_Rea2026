from torch.utils.data import Dataset
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

from PIL import Image
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

# Загружаем train-директорию для получения списка классов
train_ds = datasets.ImageFolder("image_models/data_fruits/train")
class_names = train_ds.classes
num_classes = len(class_names)

# Автоматический выбор устройства
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Классическая функция потерь для классификации
criterion = torch.nn.CrossEntropyLoss()

# Базовые трансформации (resize + to tensor)
basic_tfms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])


class SimpleImageDataset(Dataset):
    """Минимальный датасет для загрузки новых изображений в fine-tuning."""

    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Загружаем изображение и конвертируем в RGB
        img = Image.open(self.image_paths[idx]).convert("RGB")
        label = self.labels[idx]

        # Применяем трансформации, если есть
        if self.transform:
            img = self.transform(img)

        return img, label


def train_one_epoch(model, loader, optimizer):
    """Одна эпоха обучения модели на батчах."""

    model.train()     

    total_loss = 0.0             
    total_correct = 0        
    total = 0                        

    for images, labels in loader:

        # Переносим данные на устройство (CPU/GPU)
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        # Forward-pass
        logits = model(images)

        # Loss
        loss = criterion(logits, labels)

        # Backward-pass
        loss.backward()

        # Обновление весов
        optimizer.step()

        # Сбор статистик
        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        preds = logits.argmax(1)
        total_correct += (preds == labels).sum().item()
        total += batch_size

    # Средние метрики
    avg_loss = total_loss / total
    avg_acc = total_correct / total
    return avg_loss, avg_acc



def evaluate(model, loader):
    """Оценка модели: loss + accuracy + массивы предиктов для метрик."""

    model.eval()
    total_loss = 0.0
    total_correct = 0
    total = 0

    all_labels = []
    all_preds = []

    with torch.no_grad(): 
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            loss = criterion(logits, labels)

            batch_size = images.size(0)
            total_loss += loss.item() * batch_size

            preds = logits.argmax(1)
            total_correct += (preds == labels).sum().item()
            total += batch_size

            all_labels.append(labels.cpu().numpy())
            all_preds.append(preds.cpu().numpy())

    # Формируем массивы для sklearn-метрик
    y_true = np.concatenate(all_labels)
    y_pred = np.concatenate(all_preds)

    avg_loss = total_loss / total
    avg_acc = total_correct / total

    return avg_loss, avg_acc, y_true, y_pred


def fine_tuning_fruit(new_path: str, label: str) -> None:
    """Дообучение модели ResNet18 на одном новом изображении."""

    # Находим числовой id класса
    label_num = class_names.index(label)

    # Создаём датасет с одним изображением
    new_images = [new_path]
    new_labels = [label_num]

    dataset = SimpleImageDataset(new_images, new_labels, transform=basic_tfms)
    new_data = DataLoader(dataset, batch_size=4, shuffle=True)

    # Загружаем модель
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(torch.load("image_models/fruit_model.pth"))
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # Несколько эпох обучения на новых данных
    for epoch in range(5):
        loss, acc = train_one_epoch(model, new_data, optimizer)
        print(f"{epoch}: loss={loss:.4f}, acc={acc:.4f}")


    # Оценка обновлённой модели на test-директории
    test_ds  = datasets.ImageFolder("image_models/data_fruits/test", transform=basic_tfms)
    test_dl  = DataLoader(test_ds, batch_size=32, shuffle=False)
    
    test_loss, test_acc, y_true_res, y_pred_res = evaluate(model, test_dl)

    # Расчёт классических метрик
    prec = precision_score(y_true_res, y_pred_res, average="macro", zero_division=0)
    rec  = recall_score(y_true_res, y_pred_res, average="macro", zero_division=0)
    f1   = f1_score(y_true_res, y_pred_res, average="macro", zero_division=0)
    acc = accuracy_score(y_true_res, y_pred_res)

    metrics = {
        "accuracy": acc,
        "macro_precision": prec,
        "macro_recall": rec,
        "macro_f1": f1
    }

    # Сохраняем обновлённую модель
    torch.save(model.state_dict(), "image_models/fruit_model.pth")

    return metrics


def inference_image(image_path):
    """Инференс модели ResNet18: предсказание класса по изображению."""

    img = Image.open(image_path).convert("RGB")
    
    # Преобразуем изображение в Tensor и добавляем batch dimension
    tensor = basic_tfms(img).unsqueeze(0)
    tensor = tensor.to(device)

    # Загружаем модель
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(torch.load("image_models/fruit_model.pth"))
    model.to(device)
    model.eval()

    # Предсказание
    with torch.no_grad():
        logits = model(tensor)
        pred = logits.argmax(1).item()

    return class_names[pred]