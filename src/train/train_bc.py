import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agent.state_encoder import ObservationEncoder
from agent.action_mask import MAX_ACTION_SPACE
from train.train_ppo import ActorCritic, STATE_DIM, ACTION_DIM, HIDDEN_DIM

def train_bc(dataset_path="data/replays_dataset.pt", epochs=50, batch_size=32, lr=1e-3):
    if not os.path.exists(dataset_path):
        print(f"Dataset {dataset_path} not found.")
        return

    data = torch.load(dataset_path, weights_only=True)
    states = data["states"]
    masks = data["masks"]
    actions = data["actions"]
    
    print(f"Loaded {len(states)} transitions from {dataset_path}")
    
    dataset = TensorDataset(states, masks, actions)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")
    
    model = ActorCritic(STATE_DIM, ACTION_DIM, HIDDEN_DIM).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch_states, batch_masks, batch_actions in dataloader:
            batch_states = batch_states.to(device)
            batch_masks = batch_masks.to(device)
            batch_actions = batch_actions.to(device)
            
            optimizer.zero_grad()
            
            logits, _ = model(batch_states)
            
            # Mask out invalid actions by setting their logits to a large negative number
            # Using -1e9 instead of -inf to avoid NaN in softmax/cross_entropy
            masked_logits = logits.clone()
            masked_logits[~batch_masks] = -1e9
            
            loss = F.cross_entropy(masked_logits, batch_actions)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * batch_states.size(0)
            
            preds = torch.argmax(masked_logits, dim=1)
            correct += (preds == batch_actions).sum().item()
            total += batch_states.size(0)
            
        avg_loss = total_loss / total
        accuracy = correct / total * 100
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:03d}/{epochs:03d} | Loss: {avg_loss:.4f} | Acc: {accuracy:.2f}%")
            
    os.makedirs("models", exist_ok=True)
    save_path = "models/bc_model.pth"
    torch.save(model.state_dict(), save_path)
    print(f"Saved behavioral cloning model to {save_path}")

if __name__ == "__main__":
    train_bc()
