import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import utils_multi as utils
import data_augmentation as aug
import random
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import StandardScaler

# Parameters 
batch_size = 8
learning_rate = 0.0015
weight_decay = 0.0019
num_epochs = 100
train_split_ratio = 0.8

# Paths
clinical_data_path = '/mnt/localscratch/maastro/Gokul/survival_pred/data/PORSCH_complete.xlsx'
raw_ct_dir = '/mnt/localscratch/maastro/Gokul/survival_pred/data/L3_all'
annotated_ct_dir = '/mnt/localscratch/maastro/Gokul/survival_pred/ct_images'
save_location = '/mnt/localscratch/maastro/Gokul/survival_pred/saved_models/multi_model_attn_trails/multi_model_attnt10.pth'

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

#2 : Dataset Class
class MultiModalDataset(Dataset):
    def __init__(self, ct_images_df, clinical_data_df, target_col, record_ids, augment=False):
        if target_col in ct_images_df.columns:
            ct_images_df = ct_images_df.drop(columns=[target_col])
        
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
        ct_image = self.data.iloc[idx]['ct_scan']
        clinical_data = self.data.iloc[idx].drop(['ct_scan', 'record_id', self.target_col]).astype('float32').values
        target = self.data.iloc[idx][self.target_col]

        if self.augment and random.random() < 0.8:
            ct_image = aug.apply_augmentations(ct_image, num_augmentations=2)

        ct_image = torch.tensor(ct_image, dtype=torch.float32)
        clinical_data = torch.tensor(clinical_data, dtype=torch.float32)
        target = torch.tensor(target, dtype=torch.float32)

        return ct_image, clinical_data, target

record_ids = df['record_id'].tolist()
train_record_ids, test_record_ids = train_test_split(record_ids, train_size=train_split_ratio, shuffle=True, random_state=42)

# Initialize train and test datasets
train_dataset = MultiModalDataset(
    ct_images_df=df, clinical_data_df=df_clinical, 
    target_col=target_col, record_ids=train_record_ids, augment=True
)
test_dataset = MultiModalDataset(
    ct_images_df=df, clinical_data_df=df_clinical, 
    target_col=target_col, record_ids=test_record_ids, augment=False
)

#DataLoaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

print("Sample Shapes")
ct_images, clinical_data, target = next(iter(train_loader))
print("CT Image Shape:", ct_images.shape)   #[B, 2, 512, 512]
print("Clinical Data Shape:", clinical_data.shape) # [B, 48]
print("Target Shape:", target.shape)


#3 : Multi Model Architecture

#Cross Attention
class CrossAttentionLayer(nn.Module):
    def __init__(self, embed_dim_x, embed_dim_y, embed_dim_common=256, num_heads=4, dropout=0.1):
        super(CrossAttentionLayer, self).__init__()
        # Projection layers to map x and y to a common embedding dimension
        self.proj_x = nn.Linear(embed_dim_x, embed_dim_common)
        self.proj_y = nn.Linear(embed_dim_y, embed_dim_common)
        
        # Multihead Attention layers 
        self.mha_x_to_y = nn.MultiheadAttention(embed_dim=embed_dim_common, num_heads=num_heads, dropout=dropout)
        self.mha_y_to_x = nn.MultiheadAttention(embed_dim=embed_dim_common, num_heads=num_heads, dropout=dropout)
        
        self.norm1 = nn.LayerNorm(embed_dim_common)
        self.norm2 = nn.LayerNorm(embed_dim_common)
        self.dropout = nn.Dropout(dropout)
    #WIP  
    def forward(self, x, y):

        #x and y to a common embedding dimension
        x_proj = self.proj_x(x)  # [B, embed_dim]
        y_proj = self.proj_y(y) 
        
        #Include seq_len
        x_proj = x_proj.unsqueeze(1)  # [B, 1, embed_dim]
        y_proj = y_proj.unsqueeze(1)  
            
        x_proj = x_proj.permute(1, 0, 2)  
        y_proj = y_proj.permute(1, 0, 2)  
        
        # Attention: x_proj attends to y_proj
        attn_x_to_y, _ = self.mha_x_to_y(x_proj, y_proj, y_proj) 
        attn_x_to_y = attn_x_to_y.squeeze(0)  # [B, embed_dim_common]
        attn_x_to_y = self.norm1(attn_x_to_y + y_proj.squeeze(0))  
        attn_x_to_y = self.dropout(attn_x_to_y)
        
        # y_proj attends to x_proj
        attn_y_to_x, _ = self.mha_y_to_x(y_proj, x_proj, x_proj) 
        attn_y_to_x = attn_y_to_x.squeeze(0)  
        attn_y_to_x = self.norm2(attn_y_to_x + x_proj.squeeze(0))  
        attn_y_to_x = self.dropout(attn_y_to_x)
        
        # Concat attention outputs
        combined = torch.cat((attn_x_to_y, attn_y_to_x), dim=1)  # [B, 2 * embed_dim_common]
        
        return combined


# positional embeddings
class PositionalEncoding(nn.Module):
    def __init__(self, seq_len, d_model):
        super(PositionalEncoding, self).__init__()
        self.pos_embedding = nn.Parameter(torch.randn(seq_len, d_model))

    def forward(self, x):
        # pos_embedding [seq_len, d_model]
        return x + self.pos_embedding.unsqueeze(1)
    
# CNN Model for CT Images
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(2, 16, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)

        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.2864)
        self.batch_norm1 = nn.BatchNorm2d(16)
        self.batch_norm2 = nn.BatchNorm2d(32)
        self.batch_norm3 = nn.BatchNorm2d(64)

        self.mha = nn.MultiheadAttention(embed_dim=64, num_heads=4)
        self.pos_encoding = PositionalEncoding(seq_len=64, d_model=64)

        self.final_fc = nn.Linear(64, 256)

    def forward(self, x):
        # x: [B, 2, 512, 512]
        x = self.pool(F.relu(self.batch_norm1(self.conv1(x))))  
        x = self.pool(F.relu(self.batch_norm2(self.conv2(x))))  
        x = self.pool(F.relu(self.batch_norm3(self.conv3(x))))  # [B, 64, 64, 64]

        #Preparing the tensors for self attention
        batch_size, embed_dim, H, W = x.size() 
        #Reshape the tensors 
        x = x.view(batch_size, embed_dim, H * W) 

        x = x.permute(2, 0, 1)  # [seq_len, B, embed_dim]

        # Add positional encoding
        x = self.pos_encoding(x)  

        # Self-attention
        attn_output, attn_weights = self.mha(x, x, x)  

        # Aggregate: mean over sequence
        x = attn_output.mean(dim=0)  # [B, 64]

        #Final FC layer
        x = self.dropout(F.relu(self.final_fc(x)))  # [B, 256]

        return x

# Fully Connected Network for Clinical Data
class Comp_FNN(nn.Module):
    def __init__(self, input_size):
        super(Comp_FNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 16)
        self.dropout = nn.Dropout(0.2896)
        self.batch_norm1 = nn.BatchNorm1d(64)
        self.batch_norm2 = nn.BatchNorm1d(32)
        self.mha = nn.MultiheadAttention(embed_dim=16, num_heads=4, dropout=0.1)
        self.dropout2 = nn.Dropout(0.1)
        self.norm = nn.LayerNorm(16)
        self.fc = nn.Linear(16, 16)


    def forward(self, x):
        x1 = F.relu(self.batch_norm1(self.dropout(self.fc1(x))))  # [B, 64]
        x2 = F.relu(self.batch_norm2(self.dropout(self.fc2(x1))))  # [B, 32]
        x3 = self.fc3(x2)  # [B, 16]

        # Add sequence dimension
        x = x3.unsqueeze(1) # [B, 1, embed_dim]
        x = x.permute(1, 0, 2)

        #self attention
        attn_output, _ = self.mha(x, x, x)

        # Remove sequence dimension
        attn_output = attn_output.squeeze(0)

        # Residual connection and LayerNorm
        x = self.norm(attn_output + x.squeeze(0))
        x = self.norm(attn_output)
        x = self.dropout2(x)
        x = self.fc(x) # [B, 16]
        return x3 

# Final Multi Model
class MultiModel(nn.Module):
    def __init__(self, clinical_input_size):
        super(MultiModel, self).__init__()

        self.cnn = CNN()  
        self.fcnn = Comp_FNN(clinical_input_size)  
        #self.self_att_clinical = SelfAttentionLayer(embed_dim=16, num_heads=4) 

        #Cross-Attention Layer 
        self.cross_attn = CrossAttentionLayer(embed_dim_x=256, embed_dim_y=16, embed_dim_common=256, num_heads=4, dropout=0.1)

        self.fc1 = nn.Linear(256 + 16 + 512, 64) 
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1) 

    def forward(self, x_ct, x_clinical):

        # CT images
        x_ct = self.cnn(x_ct)  # [B, 256]

        # clinical data 
        x_clinical = self.fcnn(x_clinical)  # [B, 16]

        #Cross-attention between CNN and FCNN
        cross_attn_output = self.cross_attn(x_ct, x_clinical)  # [B, 512]

        # Concatenate all features
        x = torch.cat((x_ct, x_clinical, cross_attn_output), dim=1) 

        # Final dense layers
        x = F.relu(self.fc1(x))  
        x = F.relu(self.fc2(x)) 
        x = self.fc3(x)         

        return x  



device = 'cuda' if torch.cuda.is_available() else 'cpu'
clinical_input_size = df_clinical.drop(['record_id', target_col], axis=1).shape[1]

multi_model = MultiModel(clinical_input_size=clinical_input_size).to(device)


train_targets = df_clinical[df_clinical['record_id'].isin(train_record_ids)][target_col].values

#Step 4 Initialize Model, Loss, Optimizer, and Scheduler

loss_function = nn.BCEWithLogitsLoss()

optimizer = optim.Adam(multi_model.parameters(), lr=learning_rate, weight_decay=weight_decay)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

# Early Stopping parameters
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

# For saving logs for the best epoch
best_logs = {
    'epoch': None,
    'train_loss': None,
    'train_auc': None,
    'val_loss': None,
    'val_auc': None,
    'val_accuracy': None,
    'val_precision': None,
    'val_recall': None,
    'val_f1': None
}


#Step 5 Training Loop
for epoch in range(num_epochs):
    # Training Phase
    multi_model.train()
    train_loss = 0.0
    preds = []
    true_labels = []
    for ct_images, clinical_data, targets in train_loader:
        optimizer.zero_grad()
        ct_images = ct_images.to(device)
        clinical_data = clinical_data.to(device)
        targets = targets.to(device)

        outputs = multi_model(ct_images, clinical_data) 
        loss = loss_function(outputs, targets.unsqueeze(1))  
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * ct_images.size(0)
        
        # Flatten the outputs and targets
        preds.extend(outputs.detach().cpu().numpy().flatten())        
        true_labels.extend(targets.detach().cpu().numpy().flatten())  

    epoch_train_loss = train_loss / len(train_loader.dataset)
    
    # Convert lists to NumPy 
    preds = np.array(preds)  
    true_labels = np.array(true_labels).astype(int) 

    # Compute AUC
    epoch_train_auc = roc_auc_score(true_labels, preds)
    print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {epoch_train_loss:.4f}, Train AUC: {epoch_train_auc:.4f}")

    logs['train_loss'].append(epoch_train_loss)
    logs['train_auc'].append(epoch_train_auc)

    # Validation Phase
    multi_model.eval()
    val_loss = 0.0
    val_targets = []
    val_outputs = []
    with torch.no_grad():
        for ct_images, clinical_data, targets in test_loader:
            ct_images = ct_images.to(device)
            clinical_data = clinical_data.to(device)
            targets = targets.to(device)

            outputs = multi_model(ct_images, clinical_data)  
            loss = loss_function(outputs, targets.unsqueeze(1))
            val_loss += loss.item() * ct_images.size(0)
            # Convert logits to probabilities
            outputs = torch.sigmoid(outputs)  
            val_outputs.extend(outputs.cpu().numpy().flatten())    
            val_targets.extend(targets.cpu().numpy().flatten())   

    epoch_val_loss = val_loss / len(test_loader.dataset)

    # Convert lists to NumPy arrays
    val_outputs = np.array(val_outputs).astype(float)   
    val_targets = np.array(val_targets).astype(int)      

    binary_predictions = (val_outputs >= 0.5).astype(int)

    # Compute Metrics
    accuracy = accuracy_score(val_targets, binary_predictions)
    precision = precision_score(val_targets, binary_predictions, zero_division=0)
    recall = recall_score(val_targets, binary_predictions, zero_division=0)
    f1 = f1_score(val_targets, binary_predictions)
    auc = roc_auc_score(val_targets, val_outputs)

    print(f"Validation Loss: {epoch_val_loss:.4f}, Accuracy: {accuracy:.4f}, "
          f"Precision: {precision:.4f}, Recall: {recall:.4f}, "
          f"F1: {f1:.4f}, AUC: {auc:.4f}")

    logs['val_loss'].append(epoch_val_loss)
    logs['val_auc'].append(auc)
    logs['val_accuracy'].append(accuracy)
    logs['val_precision'].append(precision)
    logs['val_recall'].append(recall)
    logs['val_f1'].append(f1)

    # Early Stopping Check
    if auc > best_auc:
        best_auc = auc
        counter = 0  
        # Save 
        torch.save(multi_model.state_dict(), save_location)
        # Update logs
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

print("Training completed.")
print("Best Model Metrics:")
print(f"Epoch: {best_logs['epoch']}")
print(f"Train Loss: {best_logs['train_loss']:.4f}, Train AUC: {best_logs['train_auc']:.4f}")
print(f"Validation Loss: {best_logs['val_loss']:.4f}, Validation AUC: {best_logs['val_auc']:.4f}")
print(f"Validation Accuracy: {best_logs['val_accuracy']:.4f}, "
      f"Precision: {best_logs['val_precision']:.4f}, "
      f"Recall: {best_logs['val_recall']:.4f}, "
      f"F1 Score: {best_logs['val_f1']:.4f}")
