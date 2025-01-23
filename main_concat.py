import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split,Dataset,TensorDataset
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import utils_multi as utils
import data_augmentation as aug
#import sys
import random
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from sklearn.preprocessing import StandardScaler


# Parameters 
batch_size = 8
learning_rate = 0.0015
weight_decay = 0.001
num_epochs = 100
train_split_ratio = 0.8


# Paths
clinical_data_path = '/mnt/localscratch/maastro/Gokul/survival_pred/data/PORSCH_complete.xlsx'
raw_ct_dir = '/mnt/localscratch/maastro/Gokul/survival_pred/data/L3_all'
annotated_ct_dir = '/mnt/localscratch/maastro/Gokul/survival_pred/ct_images'
save_location = '/mnt/localscratch/maastro/Gokul/survival_pred/saved_models/multi_model_trails/multi_model10.pth'

#target_col = 'major_complications'  
#target_col = 'crpopf'                 
target_col = 'compl'                


# Step 1: Data Loading for both CT scans and clinical data
print('Loading CT scans')
df, final_record_ids = utils.load_ct_scans_2d(clinical_data_path, raw_ct_dir, annotated_ct_dir, target_col)
print("Value counts in target column for CT scans:\n", df[target_col].value_counts())
print("Number of CT scans final :",len(final_record_ids))
print('Loading Clinical Data')

df_clinical = utils.load_and_filter_clinical_data(clinical_data_path, target_col, final_record_ids)
print("Number of record_ids in clinical data", len(df_clinical))
target_counts = df_clinical['compl'].value_counts()
print("Class Distribution for clinical data:" ,target_counts)

assert set(df['record_id']) == set(df_clinical['record_id']), "Record IDs do not match between clinical and CT datasets."


#Step 2: Create Custom Dataset Class and Split into Train and Test set

class MultiModalDataset(Dataset):
    def __init__(self, ct_images_df, clinical_data_df, target_col, record_ids, augment=False):
        # Drop the target column from ct_images_df if it exists
        if target_col in ct_images_df.columns:
            ct_images_df = ct_images_df.drop(columns=[target_col])
        
        # Merge dataframes on 'record_id'
        self.data = pd.merge(
            ct_images_df[ct_images_df['record_id'].isin(record_ids)],
            clinical_data_df[clinical_data_df['record_id'].isin(record_ids)],
            on='record_id'
        ).reset_index(drop=True)
        
        self.target_col = target_col
        self.augment = augment
        

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Load the CT image and clinical data
        ct_image = self.data.iloc[idx]['ct_scan']
        clinical_data = self.data.iloc[idx].drop(['ct_scan', 'record_id', self.target_col]).astype('float32').values
        target = self.data.iloc[idx][self.target_col]

        # Apply augmentations to CT images if specified
        if self.augment and random.random() < 0.8:  # 80% chance
            ct_image = aug.apply_augmentations(ct_image, num_augmentations=2)

        # Convert data to PyTorch tensors
        ct_image = torch.tensor(ct_image, dtype=torch.float32)
        clinical_data = torch.tensor(clinical_data, dtype=torch.float32)
        target = torch.tensor(target, dtype=torch.float32)

        return ct_image, clinical_data, target

# Get list of record_ids
record_ids = df['record_id'].tolist()

# Split the record_ids
train_record_ids, test_record_ids = train_test_split(
    record_ids, train_size=train_split_ratio, shuffle=True, random_state=42
)

# Initialize train and test datasets
train_dataset = MultiModalDataset(
    ct_images_df=df, clinical_data_df=df_clinical, 
    target_col=target_col, record_ids=train_record_ids, augment=False
)
test_dataset = MultiModalDataset(
    ct_images_df=df, clinical_data_df=df_clinical, 
    target_col=target_col, record_ids=test_record_ids, augment=False
)

# DataLoaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


print("Sample Shapes")
ct_images, clinical_data, target = next(iter(train_loader))
print("CT Image Shape:", ct_images.shape) 
print("Clinical Data Shape:", clinical_data.shape)  
print("Target Shape:", target.shape)  


#Step 3: Model Architecure 
#3.1: CNN Model for CT scans
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(2, 16, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)

        # Pooling layer 
        self.pool = nn.MaxPool2d(2, 2)  
        self.dropout = nn.Dropout(0.286)
        
        # Batch normalization layers
        self.batch_norm1 = nn.BatchNorm2d(16)
        self.batch_norm2 = nn.BatchNorm2d(32)
        self.batch_norm3 = nn.BatchNorm2d(64)

        # Fully connected layers to produce a 1D feature vector
        self.fc1 = nn.Linear(64 * 8 * 8, 256)


    def forward(self, x):
        #print("Input Imge shape :", x.shape)
        x = self.pool(F.relu(self.batch_norm1(self.conv1(x))))
         #print("After conv1 + pool:", x.shape)
        x = self.pool(F.relu(self.batch_norm2(self.conv2(x))))
        #print("After conv2 + pool:", x.shape)
        x = self.pool(F.relu(self.batch_norm3(self.conv3(x)))) 
        #print("After conv3 + pool:", x.shape)

        # Flatten for fully connected layers
        x = x.view(-1, 64 * 8 * 8)  
        
        x = self.dropout(F.relu(self.fc1(x)))
        return x

#3.2: FCNN Model for Clinical Data    

class Comp_FNN(nn.Module):
    def __init__(self, input_size):
        super(Comp_FNN,self).__init__()   
        # Fully connected layers
        self.fc1 = nn.Linear(input_size,64)
        self.fc2 = nn.Linear(64,32)
        self.fc3 = nn.Linear(32,16)
        #self.fc4 = nn.Linear(16,1)   
        self.dropout = nn.Dropout(0.289)    
        self.batch_norm1 = nn.BatchNorm1d(64)
        self.batch_norm2 = nn.BatchNorm1d(32)
        #self.batch_norm3 = nn.BatchNorm1d(16)    
    def forward(self, x):
        x1 = F.relu(self.batch_norm1(self.dropout(self.fc1(x))))
        x2 = F.relu(self.batch_norm2(self.dropout(self.fc2(x1))))
        #x3 = F.relu(self.batch_norm3(self.dropout(self.fc3(x2))))
        x3 = F.relu(self.fc3(x2))
        return x3 
#3.3: Multi-Model combining both CNN and FCNN
class MultiModel(nn.Module):
    def __init__(self, clinical_input_size):
        super(MultiModel, self).__init__() 

        # Models
        self.cnn = CNN()
        self.fcnn = Comp_FNN(clinical_input_size)   

        # Fusion 
        self.fc1 = nn.Linear(256 + 16, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        #self.dropout = nn.Dropout(0.2)
    
    def forward(self, x_ct, x_clinical):
        # Process through individual networks
        x_ct = self.cnn(x_ct)
        x_clinical = self.fcnn(x_clinical)

        # Concat both the outputs
        x = torch.cat((x_ct, x_clinical), dim=1)

        # Final fully connected layers for classification
        x1 = F.relu(self.fc1(x))
        x2 = F.relu(self.fc2(x1))
        #x3 = torch.sigmoid(self.fc3(x2))
        x3 = self.fc3(x2)
        return x3



#Step 4 Initialize Model, Loss, Optimizer, and Scheduler



device = 'cuda' if torch.cuda.is_available() else 'cpu'
#clinical_input_size = X_train.shape[1]
clinical_input_size = df_clinical.drop(['record_id', target_col], axis=1).shape[1]


cnn_model = CNN()
fcnn_model = Comp_FNN(input_size=clinical_input_size)
multi_model = MultiModel(clinical_input_size=clinical_input_size).to(device)

# Print model architectures
print("CNN Model Architecture:\n", cnn_model)
print("\nFCNN Model Architecture (for Clinical Data):\n", fcnn_model)
print("\nMulti-Model Architecture:\n", multi_model)


loss_function = nn.BCEWithLogitsLoss()

optimizer = optim.Adam(multi_model.parameters(), lr = learning_rate, weight_decay= weight_decay)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)


#Step 5 Training Loop

patience = 60
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

# For savings logs for the best epoch
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
    multi_model.train()
    train_loss = 0.0
    preds = []
    true_labels = []
    #training
    for ct_images, clinical_data, targets in train_loader:
        optimizer.zero_grad()
        #Move to GPU
        ct_images, clinical_data, targets = ct_images.to(device), clinical_data.to(device), targets.to(device)
        #Forward Pass
        outputs = multi_model(ct_images, clinical_data)
        loss = loss_function(outputs,targets.unsqueeze(1))
        #Backwardpass and Optimization
        loss.backward()
        optimizer.step()
        #Accumulate train loss
        train_loss += loss.item() * ct_images.size(0)

        preds.append(outputs.detach().cpu().numpy())
        true_labels.append(targets.detach().cpu().numpy())

    #Avg training loss for each epoch
    epoch_train_loss = train_loss/ len(train_loader.dataset)
    preds = np.vstack(preds)
    true_labels = np.vstack(true_labels)
    epoch_train_auc = roc_auc_score(true_labels.ravel(), preds.ravel())
    print(f"Epoch [{epoch+1}/{num_epochs}],Train Loss: {epoch_train_loss:.4f}")

    logs['train_loss'].append(epoch_train_loss)
    logs['train_auc'].append(epoch_train_auc)
    
    #validation
    multi_model.eval()
    val_loss = 0.0
    val_targets = []
    val_outputs = []
    with torch.no_grad():
        for ct_images, clinical_data, targets in test_loader:
            #Move to GPU
            ct_images, clinical_data, targets = ct_images.to(device), clinical_data.to(device), targets.to(device)
            #Forward Pass
            outputs = multi_model(ct_images, clinical_data)
            loss = loss_function(outputs, targets.unsqueeze(1))
            #Accumulate Backward Loss
            val_loss += loss.item() * ct_images.size(0)
            # Convert logits to probabilities
            outputs = torch.sigmoid(outputs)  
            val_outputs.extend(outputs.cpu().numpy())
            val_targets.extend(targets.cpu().numpy())
    
    #Calculate metrics

    #compute avg val loss
    epoch_val_loss = val_loss / len(test_loader.dataset)
    # Compute validation metrics
    val_outputs = np.array(val_outputs).squeeze()
    val_targets = np.array(val_targets).astype(int)
    binary_predictions  = (np.array(val_outputs).squeeze() >= 0.5).astype(int)
    #val_targets = np.concatenate(val_targets)

    accuracy = accuracy_score(val_targets,binary_predictions )
    precision = precision_score(val_targets,binary_predictions , zero_division=0)
    recall = recall_score(val_targets,binary_predictions , zero_division=0)
    f1 = f1_score(val_targets, binary_predictions )
    auc = roc_auc_score(val_targets, np.array(val_outputs).squeeze())

    print(f"Validation Loss: {epoch_val_loss:.4f}, Accuracy: {accuracy:.4f}, Precision: {precision:.4f},Recall: {recall:.4f}, F1: {f1:.4f}, AUC: {auc:.4f}")

    logs['val_loss'].append(epoch_val_loss)
    logs['val_auc'].append(auc)
    logs['val_accuracy'].append(accuracy)
    logs['val_precision'].append(precision)
    logs['val_recall'].append(recall)
    logs['val_f1'].append(f1)

    
    # Early Stopping Check
    if auc > best_auc:
        best_auc = auc
        counter = 0  # Reset counter
        # Save the model
        torch.save(multi_model.state_dict(),save_location)
        # Update best logs
        best_logs['epoch'] = epoch + 1
        best_logs['train_loss'] = epoch_train_loss
        best_logs['train_auc'] = epoch_train_auc
        best_logs['val_loss'] = epoch_val_loss
        best_logs['val_auc'] = auc
        best_logs['val_accuracy'] = accuracy
        best_logs['val_precision'] = precision
        best_logs['val_recall'] = recall
        best_logs['val_f1'] = f1
        
        print(f"** Best model saved at epoch {epoch+1} with Val AUC: {auc:.4f} **")

    else:
        counter += 1
        if counter >= patience:
            print('Early Stopping')
            break

    # Update scheduler
    scheduler.step()

#print the best metrics
print("Training completed.")
print("Best Model Metrics:")
print(f"Epoch: {best_logs['epoch']}")
print(f"Train Loss: {best_logs['train_loss']:.4f}, Train AUC: {best_logs['train_auc']:.4f}")
print(f"Validation Loss: {best_logs['val_loss']:.4f}, Validation AUC: {best_logs['val_auc']:.4f}")
print(f"Validation Accuracy: {best_logs['val_accuracy']:.4f}, Precision: {best_logs['val_precision']:.4f}, Recall: {best_logs['val_recall']:.4f}, F1 Score: {best_logs['val_f1']:.4f}")