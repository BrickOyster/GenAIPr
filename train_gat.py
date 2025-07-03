import os
import argparse
import torch
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
from models.gat_model import GATLayoutPredictor
from utils.metrics import layout_loss, calculate_iou
from utils.visualize_layout import save_visualization
from datetime import datetime

class VisualGenomeDataset(torch.utils.data.Dataset):
    def __init__(self, data_path, device='cuda' if torch.cuda.is_available() else 'cpu', 
                 train_ratio=0.75, shuffle=True, seed=42):
        self.device = device
        full_data = torch.load(data_path, weights_only=False)
        
        # Shuffle if needed
        if shuffle:
            g = torch.Generator()
            g.manual_seed(seed)
            indices = torch.randperm(len(full_data), generator=g)
            full_data = [full_data[i] for i in indices]
        
        # Split the data
        split_idx = int(len(full_data) * train_ratio)
        self.data = full_data[:split_idx]
        
        print(f"Dataset split: {train_ratio} train | {1 - train_ratio} test")
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        graph = item['graph']
        
        # Ensure valid node features (must be integers for embedding)
        if not hasattr(graph, 'x') or graph.x is None:
            # Create sequential IDs for nodes
            graph.x = torch.arange(graph.num_nodes, dtype=torch.long).unsqueeze(1)
        else:
            # Convert existing features to long integers
            graph.x = graph.x.long()
        
        # Ensure valid edge attributes (must be 2D)
        if not hasattr(graph, 'edge_attr') or graph.edge_attr is None:
            graph.edge_attr = torch.ones(graph.edge_index.size(1), 1, dtype=torch.float)
        elif graph.edge_attr.dim() == 1:
            graph.edge_attr = graph.edge_attr.unsqueeze(-1).float()
            
        return graph.to(self.device), item['layout'].float().to(self.device), item['objects']

def save_checkpoint(model, optimizer, epoch, loss, path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, path)

def train(args):
    device = torch.device(args.device)
    print(f"\nStarting training on {device}")
    
    # Setup output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join("checkpoints", f"gat_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "pngs"), exist_ok=True)
    print(f"Checkpoints saved to: {os.path.abspath(output_dir)}")
    
    # Load dataset without batching
    try:
        print(f"Loading dataset from {args.data_path}")
        dataset = VisualGenomeDataset(args.data_path, device)
        print(f"Loaded {len(dataset)} samples")
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        return

    # Initialize model
    model = GATLayoutPredictor(
        num_node_features=args.num_node_features,
        num_edge_features=args.num_edge_features,
        hidden_dim=args.hidden_dim,
        heads=args.num_heads,
        num_layers=args.num_layers
    ).to(device)
    
    # Training loop with epoch-level progress bar
    best_loss = float('inf')
    dataset_size = len(dataset)

    optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.02, total_iters=dataset_size*2)
    if args.checkpoint_dir:
        checkpoint_path = os.path.join(args.checkpoint_dir, "best_model.pt")
        if os.path.exists(checkpoint_path):
            print(f"Loading checkpoint from {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            best_loss = checkpoint['loss']
            print(f"Resumed training from epoch {checkpoint['epoch']} with loss {best_loss:.4f}")
        else:
            print(f"No checkpoint found at {checkpoint_path}, starting fresh.")
    with tqdm(range(args.epochs*dataset_size), desc="Training...", unit="samples") as pbar:
        for epoch in range(args.epochs):
            model.train()
            epoch_loss = 0
            avg_loss = 0
            epoch_samples = 0

            # Process each sample individually
            for graph, layout, objects in dataset:
                optimizer.zero_grad()
                pred_layout = model(graph)
                loss = layout_loss(pred_layout, layout) + 0.6*torch.sum((1 - calculate_iou(pred_layout, layout)))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                
                epoch_loss += loss.item()

                epoch_samples += 1
                pbar.update(1)
                if epoch_samples % 100 == 0:
                    avg_loss = epoch_loss / epoch_samples if epoch_samples > 0 else 0
                    pbar.set_postfix(loss=f"{avg_loss:.4f}")

                if epoch_samples % (dataset_size // 4) == 0:
                    avg_loss = epoch_loss / epoch_samples if epoch_samples > 0 else 0
                    save_visualization(
                        objects, 
                        pred_layout.cpu(), 
                        layout.cpu(),
                        os.path.join(output_dir, f"pngs/epoch_{epoch+1}_sample_{epoch_samples}.png"),
                        f"{avg_loss:.4f}"
                    )
                        
            avg_loss = epoch_loss / len(dataset)
            
            # Save checkpoints
            if (epoch + 1) % args.checkpoint_interval == 0:
                checkpoint_path = os.path.join(output_dir, f"epoch_{epoch+1}.pt")
                save_checkpoint(model, optimizer, epoch, avg_loss, checkpoint_path)
                
                if avg_loss < best_loss:
                    best_loss = avg_loss
                    best_path = os.path.join(output_dir, "best_model.pt")
                    save_checkpoint(model, optimizer, epoch, avg_loss, best_path)
    
    # Final save
    final_path = os.path.join(output_dir, "final_model.pt")
    save_checkpoint(model, optimizer, args.epochs, avg_loss, final_path)
    
    print(f"\nTraining complete!")
    print(f"Best loss: {best_loss:.4f}")
    print(f"Final loss: {avg_loss:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train GAT Layout Predictor')
    
    # Data parameters
    parser.add_argument('--data-path', type=str, required=True,
                      help='Path to preprocessed dataset .pt file')
    parser.add_argument('--checkpoint_dir', type=str, default=None,
                        help='Directory to resume training from a checkpoint')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=50,
                      help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                      help='Learning rate')
    parser.add_argument('--checkpoint_interval', type=int, default=5,
                      help='Save checkpoint every N epochs')
    
    # Model parameters
    parser.add_argument('--num_node_features', type=int, default=1000,
                      help='Dimension of node features')
    parser.add_argument('--num_edge_features', type=int, default=1,
                      help='Dimension of edge features')
    parser.add_argument('--hidden_dim', type=int, default=128,
                      help='Hidden dimension size')
    parser.add_argument('--num_heads', type=int, default=4,
                      help='Number of attention heads')
    parser.add_argument('--num_layers', type=int, default=3,
                      help='Number of GAT layers')
    
    # System
    parser.add_argument('--device', type=str, 
                      default='cuda' if torch.cuda.is_available() else 'cpu',
                      help='Device to train on (cuda/cpu)')
    parser.add_argument('--seed', type=int, default=42,
                      help='Random seed')
    
    args = parser.parse_args()
    
    # Set random seeds
    torch.manual_seed(args.seed)
    if args.device == 'cuda':
        torch.cuda.manual_seed_all(args.seed)
    
    train(args)