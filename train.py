import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from sklearn.metrics import f1_score, recall_score


def train(model, 
        train_loader, val_loader,
        num_epochs=10, 
        learning_rate=1e-4, 
        use_amp=True, 
        device=None):
    
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)

    # 設定類別不平衡的權重 (pos_weight 只針對正類，1類越少權重越大)
    pos_weight = torch.tensor([12.0]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.3, patience=7)
    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    # early stopping
    best_val_recall = 0.0
    patience = 20
    no_improve_count = 0

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        all_preds = []
        all_labels = []

        loop = tqdm(train_loader, desc=f"[Epoch {epoch+1}/{num_epochs}] Training", leave=False)
        for inputs, labels in loop:
            inputs = inputs.to(device)
            labels = labels.float().to(device).unsqueeze(1)  # 轉成 float 並加一個維度

            optimizer.zero_grad()

            if use_amp:
                with torch.cuda.amp.autocast():
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * inputs.size(0)

            preds = (torch.sigmoid(outputs) >= 0.5).int()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            loop.set_postfix(loss=loss.item())

        train_loss = total_loss / len(train_loader.dataset)
        train_f1 = f1_score(all_labels, all_preds, average='macro')
        train_recall = recall_score(all_labels, all_preds, average='macro')
        print(f"[Epoch {epoch+1}]")
        print(f"\tTrain Loss: {train_loss:.4f} | Train Recall: {train_recall:.4f} | Train F1: {train_f1:.4f}")

        # 驗證階段
        _, val_recall = validate(model, val_loader)

        scheduler.step(val_recall)

        if val_recall > best_val_recall:
            best_val_recall = val_recall
            no_improve_count = 0
            torch.save(model.state_dict(), "best_model.pth")
        else:
            no_improve_count += 1

        if no_improve_count >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

def validate(model, val_loader, threshold=0.5, device=None):
    model.eval()
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    criterion = nn.BCEWithLogitsLoss()

    total_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            labels = labels.float().to(device).unsqueeze(1)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * inputs.size(0)

            preds = (torch.sigmoid(outputs) >= threshold).int()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    val_loss = total_loss / len(val_loader.dataset)
    val_f1 = f1_score(all_labels, all_preds, average='macro')
    val_recall = recall_score(all_labels, all_preds, average='macro')
    print(f"\t  Val Loss: {val_loss:.4f} |   Val Recall: {val_recall:.4f} |   Val F1: {val_f1:.4f}\n")

    return all_preds, val_recall
