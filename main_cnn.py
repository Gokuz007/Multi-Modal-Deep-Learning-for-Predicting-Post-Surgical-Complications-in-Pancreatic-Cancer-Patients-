import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import utils_new as utils
import data_augmentation as aug
import sys
import random
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix
from torch.optim.lr_scheduler import ReduceLROnPlateau


# Parameters
batch_size = 8
learning_rate = 1e-3
num_epochs = 150
train_split_ratio = 0.8

# Paths
clinical_data_path = '/mnt/localscratch/maastro/Gokul/survival_pred/data/PORSCH_complete.xlsx'
raw_ct_dir = '/mnt/localscratch/maastro/Gokul/survival_pred/data/L3_all'
annotated_ct_dir = '/mnt/localscratch/maastro/Gokul/survival_pred/ct_images'
save_location = '/mnt/localscratch/maastro/Gokul/survival_pred/saved_models/cnn_trials/cnn_model_compl1.pth'

#target_col = 'major_complications'  
#target_col = 'crpopf'                 
target_col = 'compl'                


# Step 1: Data Loading
df = utils.load_ct_scans_2d(clinical_data_path, raw_ct_dir, annotated_ct_dir, target_col)

print("Unique values in target column:", df[target_col].unique())
print("Value counts in target column:\n", df[target_col].value_counts())

print("Number of missing values in target column:", df[target_col].isnull().sum())
df = df.dropna(subset=[target_col])  # Drop rows with NaN targets

print("Number of missing values in target column after dropping null rows:", df[target_col].isnull().sum())

# Step 2: Split into Training and Test Data
train_size = int(len(df) * train_split_ratio)
test_size = len(df) - train_size
indices = list(range(len(df)))
train_indices, test_indices = train_test_split(
    indices, train_size=train_split_ratio, shuffle=True
)

print("Class distribution in training set:\n", df.iloc[train_indices][target_col].value_counts())

np.save("/mnt/localscratch/maastro/Gokul/survival_pred/data/ct_images_test_indices.npy", test_indices)

def prepare_data(df, indices, augment=False):
    images = []
    targets = []

    for idx in indices:
        row = df.iloc[idx]
        ct_scan = row['ct_scan']
        target = row[target_col]
        
        if augment:
            #Apply augmentation 50% of time
            if random.random()< 0.8:
                ct_scan = aug.apply_augmentations(ct_scan, num_augmentations=2)
        
        images.append(ct_scan)
        targets.append(target)
    
    images = np.array(images)
    targets = np.array(targets)
    
    images_tensor = torch.tensor(np.array(images), dtype=torch.float32)
    targets_tensor = torch.tensor(targets, dtype=torch.float32)
    
    return DataLoader(torch.utils.data.TensorDataset(images_tensor, targets_tensor), batch_size=batch_size, shuffle=True)

# Prepare train and test datasets
train_loader = prepare_data(df, train_indices, augment=True)  # Augmentation for training data
test_loader = prepare_data(df, test_indices, augment=False)   

# Step 3: Define the Custom CNN Model
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(2, 16, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        #self.conv4 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        #self.conv5 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        #self.conv6 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)  
        
        # Pooling layer 
        self.pool = nn.MaxPool2d(2, 2)  
        self.dropout = nn.Dropout(0.2)
        
        # Batch normalization layers
        self.batch_norm1 = nn.BatchNorm2d(16)
        self.batch_norm2 = nn.BatchNorm2d(32)
        self.batch_norm3 = nn.BatchNorm2d(64)
        #self.batch_norm4 = nn.BatchNorm2d(128)
        #self.batch_norm5 = nn.BatchNorm2d(256)
        #self.batch_norm6 = nn.BatchNorm2d(512)
        
        # Fully connected layers
        self.fc1 = nn.Linear(64 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 1)  # Binary classification (logit output)

    def forward(self, x):
        #print("Input Imge shape :", x.shape)
        x = self.pool(F.relu(self.batch_norm1(self.conv1(x))))
         #print("After conv1 + pool:", x.shape)
        x = self.pool(F.relu(self.batch_norm2(self.conv2(x))))
        #print("After conv2 + pool:", x.shape)
        x = self.pool(F.relu(self.batch_norm3(self.conv3(x)))) 
        #print("After conv3 + pool:", x.shape)
        #x = self.pool(F.relu(self.batch_norm4(self.conv4(x))))
        #print("After conv4 + pool:", x.shape)

        #x = F.relu(self.batch_norm5(self.conv5(x)))  
        #x = F.relu(self.batch_norm6(self.conv6(x)))  
        

        # Flatten for fully connected layers
        x = x.view(-1, 64 * 8 * 8)  
        
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        
        x = self.fc3(x)  # Output raw logits
        return x


# Step 4: Initialize Model, Loss, Optimizer, and Scheduler
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = CNN().to(device)


criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-3)
#scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)


# Step 5: Training Loop

#initialize variables for ealy stopping
patience = 25
best_auc = 0
counter = 0

# To store training history
logs = {
    'train_loss': [],
    'train_auc': [],
    'val_loss': [],
    'val_auc': [],
    'val_accuracy': [],
    'val_precision': [],
    'val_recall': [],
    'val_f1': []
}

# FOR SAVING logs for the best epoch
best_logs = {
    'epoch':None,
    'train_loss': None,
    'train_auc': None,
    'val_loss': None,
    'val_auc': None,
    'val_accuracy': None,
    'val_precision': None,
    'val_recall': None,
    'val_f1': None
}

for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    
    # Training Loop
    for images, target in train_loader:
        optimizer.zero_grad()
        images = images.to(device)
        target = target.to(device)
        
        # Forward pass
        outputs = model(images)
        #print("Training outputs:", outputs.detach().cpu().numpy())
        #print("Training targets:", target.cpu().numpy())
        loss = criterion(outputs, target.unsqueeze(1))
        
        # Backward and optimize
        loss.backward()
        optimizer.step()
        
        # Accumulate training loss
        train_loss += loss.item()

    #scheduler.step()

    # Validation Loop
    model.eval()
    val_loss = 0.0
    val_targets, val_predictions = [], []
    with torch.no_grad():
        for images, target in test_loader:
            images = images.to(device)
            target = target.to(device)

            outputs = model(images)
            val_loss += criterion(outputs, target.unsqueeze(1)).item()
            outputs = torch.sigmoid(outputs)  # Convert logits to probabilities
            #print("Validation outputs:", outputs.detach().cpu().numpy())
            #print("Validation targets:", target.cpu().numpy())
            
            val_predictions.append(outputs.cpu().numpy())
            val_targets.append(target.cpu().numpy())
            
    val_predictions = np.concatenate(val_predictions)
    val_targets = np.concatenate(val_targets)

    val_predictions_binary = (val_predictions > 0.6).astype(int)

    # Calculate confusion matrix
    tn, fp, fn, tp = confusion_matrix(val_targets, val_predictions_binary).ravel()

    acc = accuracy_score(val_targets, val_predictions_binary)
    precision = precision_score(val_targets, val_predictions_binary)
    recall = recall_score(val_targets, val_predictions_binary)
    f1 = f1_score(val_targets, val_predictions_binary)
    auc = roc_auc_score(val_targets, val_predictions)

    if (epoch + 1) % 1 == 0:
        print(f'Epoch {epoch+1}/{num_epochs}, '
              f'Train Loss: {train_loss/len(train_loader):.4f}, '
              f'Val Loss: {val_loss/len(test_loader):.4f}')
        print(f'Validation - Acc: {acc:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}, AUC: {auc:.4f}')
        print(f'True Positives (TP): {tp}, False Positives (FP): {fp}, False Negatives (FN): {fn}')
    
    # Compute average losses
    avg_train_loss = train_loss / len(train_loader)
    avg_val_loss = val_loss / len(test_loader)

    scheduler.step(avg_val_loss)

    #early stopping
    if auc  > best_auc:
      best_auc = auc
      best_logs['epoch'] = epoch + 1
      best_logs['val_loss'] = avg_val_loss
      best_logs['val_auc'] = auc
      best_logs['val_accuracy'] = acc
      best_logs['val_precision'] = precision
      best_logs['val_recall'] = recall
      best_logs['val_f1'] = f1

      # Save the best model
      torch.save(model.state_dict(), save_location)
      print(f"** Best model saved at epoch {epoch+1} with Val AUC: {auc:.4f} **")  #
      counter = 0
    else:
      counter += 1

      if counter >= patience:
        print('Early Stopping')
        break

print("\n===== Best Epoch Metrics =====")
print("\nThe model with following metrics was saved")
print(f"Epoch: {best_logs['epoch']}")
print(f"Validation Loss: {best_logs['val_loss']:.4f}")
print(f"Validation AUC: {best_logs['val_auc']:.4f}")
print(f"Validation Accuracy: {best_logs['val_accuracy']:.4f}")
print(f"Validation Precision: {best_logs['val_precision']:.4f}")
print(f"Validation Recall: {best_logs['val_recall']:.4f}")
print(f"Validation F1-Score: {best_logs['val_f1']:.4f}")
# Save the model
#torch.save(model.state_dict(), '/mnt/localscratch/maastro/Gokul/survival_pred/saved_models/cnn_model_compl.pth')

