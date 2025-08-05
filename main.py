from data_preparation import process
from hitnet import HitNet
from train import train, validate
from sklearn.metrics import classification_report, confusion_matrix, f1_score, recall_score
from torch.nn.functional import sigmoid
from torch.utils.data import DataLoader, TensorDataset
import torch
import numpy as np
import os
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

MATCHES_FOR_TRAIN = 27
MATCHES_FOR_TEST = 4
MATCHES_DIR = 'D:/AmeHibiki/Desktop/monotrack/matches'

def upsample_positive(x, y, repeat=5):
    pos_idx = (y == 1)
    neg_idx = (y == 0)

    x_pos = x[pos_idx]
    y_pos = y[pos_idx]

    x_neg = x[neg_idx]
    y_neg = y[neg_idx]

    x_pos_upsampled = np.repeat(x_pos, repeat, axis=0)
    y_pos_upsampled = np.repeat(y_pos, repeat, axis=0)

    x_new = np.concatenate([x_neg, x_pos_upsampled], axis=0)
    y_new = np.concatenate([y_neg, y_pos_upsampled], axis=0)

    # optional shuffle
    perm = np.random.permutation(len(x_new))
    return x_new[perm], y_new[perm]

def make_data(matches):
    X_lst, y_lst = [], []
    for match in matches:
        basedir = f'{MATCHES_DIR}/{match}'
        for video in os.listdir(f'{basedir}/rally_video/'):
            rally = video.split('.')[0]
            data = process(basedir, rally)
            if data is None:
                continue
            x, y = data
            X_lst.append(x)
            y_lst.append(y)
    X = np.vstack(X_lst)
    y = np.hstack(y_lst)
    return X, y

def get_logits_labels(model, data_loader, device=None):
    model.eval()
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logits_list = []
    labels_list = []

    model.cuda()

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)

            logits_list.append(outputs.cpu())
            labels_list.append(labels.cpu())

    all_logits = torch.cat(logits_list, dim=0)
    all_labels = torch.cat(labels_list, dim=0)
    return all_logits, all_labels

def find_best_threshold(labels, probs):
    best_thresh = 0.5
    best_recall = 0.0

    thresholds = [x * 0.01 for x in range(100)]  # 測試不同的閾值

    for t in thresholds:
        preds = (probs >= t).astype(int)
        recall = recall_score(labels, preds, average='macro')

        if recall > best_recall:
            best_recall = recall
            best_thresh = t

    return best_thresh, best_recall


if __name__ == '__main__':

    matches = list('match' + str(i) for i in range(1, MATCHES_FOR_TRAIN + 1))
    test_matches = list('test_match' + str(i) for i in range(1, MATCHES_FOR_TEST + 1))

    # check if data already exists
    if os.path.exists('X_train.npy') and os.path.exists('y_train.npy'):
        X_train = np.load('X_train.npy')
        y_train = np.load('y_train.npy')
    else:
        X_train, y_train = make_data(matches)
        np.save('X_train.npy', X_train)
        np.save('y_train.npy', y_train)

    if os.path.exists('X_test.npy') and os.path.exists('y_test.npy'):
        X_test = np.load('X_test.npy')
        y_test = np.load('y_test.npy')
    else:
        X_test, y_test = make_data(test_matches)
        np.save('X_test.npy', X_test)
        np.save('y_test.npy', y_test)

    # X_train_balanced, y_train_balanced = upsample_positive(X_train, y_train, repeat=5)
    print("Train labels distribution:")
    unique, counts = np.unique(y_train, return_counts=True)
    for label, count in zip(unique, counts):
        print(f"Label {label}: {count} samples ({count / len(y_train):.2%})")

    print("\nTest labels distribution:")
    unique, counts = np.unique(y_test, return_counts=True)
    for label, count in zip(unique, counts):
        print(f"Label {label}: {count} samples ({count / len(y_test):.2%})")

    # Print size
    print(f'\nX_train shape: {X_train.shape}')
    print(f'y_train shape: {y_train.shape}')
    print(f'X_test shape: {X_test.shape}')
    print(f'y_test shape: {y_test.shape}\n')

    # 資料轉換
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    X_val_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_test, dtype=torch.long)

    # Dataset & Dataloader
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)

    batch_size = 4096
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # train
    num_consec = 7 
    model = HitNet(X_train.shape[2], num_consec=num_consec)
    train(model, train_loader, val_loader, num_epochs=200, learning_rate=1e-3, use_amp=False)

    # load the best model
    model.load_state_dict(torch.load('best_model.pth'))

    # find best threshold
    logits, labels = get_logits_labels(model, val_loader)
    probs = sigmoid(logits).squeeze().cpu().numpy()
    best_thresh, best_recall = find_best_threshold(labels.numpy(), probs)

    print(f"Best threshold = {best_thresh:.2f}, Recall = {best_recall:.4f}")

    # evaluate 
    all_preds, _ = validate(model, val_loader, threshold=best_thresh)
    print(classification_report(y_test, all_preds))
    print(confusion_matrix(y_test, all_preds, normalize='true'))

    # save model
    torch.save(model.state_dict(), 'hitnet_recall.pth')