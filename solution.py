import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import torch.backends.cudnn as cudnn

# Check device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# --- 1. DATA GENERATION ---
np.random.seed(42)
n_samples = 5000 

def createRow(n, classes):
    base = np.sin(np.linspace((np.random.rand(3)),(np.random.rand(3) + np.array([10,15,7])),n))
    if classes[0] > 0:
        base[np.random.randint(0,n), 0] += 2
    if classes[1] > 0:
        base[np.random.randint(0,n), 1] -= 2
    if classes[2] > 0:
        x = np.random.randint(0,n-5)
        base[x:x+4,2] = 0
    if classes[3] > 0:
        x = np.random.randint(0,n-10)
        base[x:x+8,1] += 1.5
    if classes[4] > 0:
        x = np.random.randint(0,n-7)
        base[x:x+6,0] += 1.5
        base[x:x+6,2] -= 1.5
    base += np.random.rand(*base.shape)*.2
    return base

print("Generating Data...")
X_raw, y_raw = [], []
lengths = [] # Store lengths for packing

for _ in range(n_samples):
    cl = np.random.rand(5)<.25
    n = np.random.randint(40,60)
    X_raw.append(createRow(n, cl))
    y_raw.append(cl.astype(int))
    lengths.append(n)

# --- 2. PREPROCESSING ---
X_tensors = [torch.tensor(x, dtype=torch.float32) for x in X_raw]

# Pad sequences
X_padded = torch.nn.utils.rnn.pad_sequence(X_tensors, batch_first=True, padding_value=0.0)
y_tensor = torch.tensor(np.array(y_raw), dtype=torch.float32)
lengths_tensor = torch.tensor(lengths, dtype=torch.int64)

max_len = X_padded.shape[1]

# Split data (We must keep X, y, AND lengths together)
X_train, X_test, y_train, y_test, len_train, len_test = train_test_split(
    X_padded, y_tensor, lengths_tensor, test_size=0.2, random_state=42
)

# Dataset now includes lengths
train_dataset = TensorDataset(X_train, y_train, len_train)
test_dataset = TensorDataset(X_test, y_test, len_test)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

print(f"Input Shape: {X_train.shape}")

# --- 3. MODEL ARCHITECTURE ---
class BiLSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(BiLSTMClassifier, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True, bidirectional=True)
        self.fc1 = nn.Linear(hidden_size * 2, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, num_classes)
        
    def forward(self, x, x_lengths):
        # 1. Pack the sequence
        # We must place lengths on CPU for pack_padded_sequence
        x_lengths_cpu = x_lengths.cpu()
        
        # enforce_sorted=False lets us handle unsorted batches (slightly slower but easier)
        packed_input = nn.utils.rnn.pack_padded_sequence(x, x_lengths_cpu, batch_first=True, enforce_sorted=False)
        
        # 2. Pass through LSTM
        # We only care about the final hidden state (h_n)
        _, (h_n, c_n) = self.lstm(packed_input)
        
        # 3. Concatenate forward and backward hidden states
        # h_n shape: (num_layers * num_directions, batch, hidden_size)
        cat_hidden = torch.cat((h_n[-2, :, :], h_n[-1, :, :]), dim=1)
        
        x = self.fc1(cat_hidden)
        x = self.relu(x)
        logits = self.fc2(x)
        return logits

model = BiLSTMClassifier(input_size=3, hidden_size=64, num_classes=5).to(device)

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# --- 4. TRAINING ---
print("\nTraining Model...")
num_epochs = 100

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels, seq_lens in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        # Sequence lengths don't strictly need to be on GPU for packing, 
        # but good to have them handy.
        
        optimizer.zero_grad()
        
        # Pass inputs AND lengths
        outputs = model(inputs, seq_lens)
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}")

# --- 5. EVALUATION ---
print("\nEvaluating...")
model.eval()
all_preds = []
all_targets = []

with torch.no_grad():
    for inputs, labels, seq_lens in test_loader:
        inputs = inputs.to(device)
        outputs = model(inputs, seq_lens)
        
        probs = torch.sigmoid(outputs)
        preds = (probs > 0.5).float()
        
        all_preds.append(preds.cpu())
        all_targets.append(labels.cpu())

y_pred = torch.cat(all_preds).numpy()
y_true = torch.cat(all_targets).numpy()

target_names = ['C0: S0 Spike', 'C1: S1 Dip', 'C2: S2 Flat', 'C3: S1 Shift', 'C4: S0+S2 Shift']
print(classification_report(y_true, y_pred, target_names=target_names, zero_division=0))

# --- 6. EXPLANATION (Saliency Map) ---
def explain_prediction(sample_idx, X_data, y_true_data, length_data):
    """
    Generates saliency map. 
    FIX: Temporarily disables CUDNN to allow gradients in eval mode for RNNs.
    """
    model.eval()
    
    # Prepare input
    input_tensor = X_data[sample_idx].unsqueeze(0).to(device)
    input_tensor.requires_grad_()
    
    # Get the length for this single sample
    length_tensor = length_data[sample_idx].unsqueeze(0) # Shape (1,)
    
    true_indices = np.where(y_true_data[sample_idx] == 1)[0]
    if len(true_indices) == 0:
        print("No defects in this sample.")
        return

    # !!! CRITICAL FIX FOR CUDNN RNN BACKWARD ERROR !!!
    # CUDNN optimizations don't support computing input gradients in eval mode.
    # We disable CUDNN for this block, forcing PyTorch to use the native implementation.
    with cudnn.flags(enabled=False):
        logits = model(input_tensor, length_tensor)
        
        fig, axes = plt.subplots(len(true_indices), 1, figsize=(10, 4 * len(true_indices)), squeeze=False)
        
        for i, class_idx in enumerate(true_indices):
            if input_tensor.grad is not None:
                input_tensor.grad.zero_()
            
            score = logits[0, class_idx]
            
            # Now backward() will work
            score.backward(retain_graph=True)
            
            saliency = input_tensor.grad.abs().squeeze(0).cpu().numpy()
            
            # Visualization logic
            ax = axes[i, 0]
            input_series = input_tensor.detach().cpu().squeeze(0).numpy()
            
            ax.plot(input_series[:, 0], label='Sensor 0', alpha=0.6)
            ax.plot(input_series[:, 1], label='Sensor 1', alpha=0.6)
            ax.plot(input_series[:, 2], label='Sensor 2', alpha=0.6)
            
            # Use actual length to clip visualization (ignore padding in plot)
            actual_len = length_tensor.item()
            
            saliency_map = saliency.max(axis=1) 
            saliency_norm = (saliency_map - saliency_map.min()) / (saliency_map.max() - saliency_map.min() + 1e-8)
            saliency_img = saliency_norm.reshape(1, -1)
            
            # Show heatmap only up to actual length
            ax.imshow(saliency_img, aspect='auto', cmap='Reds', alpha=0.5, 
                      extent=[0, max_len, ax.get_ylim()[0], ax.get_ylim()[1]])
            
            ax.set_title(f"Explanation for {target_names[class_idx]}")
            ax.legend(loc='upper right')
            ax.set_xlim(0, actual_len) # Clean up plot to only show real data

        plt.tight_layout()
        plt.show()

print("\nGenerating Explanation for a sample with defects...")
y_test_np = y_test.cpu().numpy()
defect_indices = np.where(np.sum(y_test_np, axis=1) >= 2)[0]

if len(defect_indices) > 0:
    # Pass the Lengths tensor to the explanation function
    explain_prediction(defect_indices[0], X_test, y_test_np, len_test)
else:
    print("No multi-defect samples found in test set to explain.")