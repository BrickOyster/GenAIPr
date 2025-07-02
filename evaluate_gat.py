import os
import argparse
import random
import torch
from tqdm import tqdm
from utils.metrics import calculate_iou, node_edge_accuracy, layout_loss
from utils.visualize_layout import save_visualization

class Evaluator:
    def __init__(self, model, checkpoint_path, data_path, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.model = model.to(device)
        self.load_model(checkpoint_path)
        self.load_data(data_path)

    def load_data(self, data_path):
        """Load and split data (75% train, 25% test)"""
        full_data = torch.load(data_path, weights_only=False)
        split_idx = int(len(full_data) * 0.75)
        self.test_data = full_data[split_idx:]
        print(f"Loaded {len(self.test_data)} test samples")
    
    def load_model(self, checkpoint_path):
        """Load model weights from checkpoint"""
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded model weights from {checkpoint_path}")

    def evaluate(self, save_dir=None, num_visualizations=5):
        """Run evaluation with proper tensor shape handling"""
        self.model.eval()
        metrics = {
            'iou': [],
            'ged': [],
            'node_acc': [],
            'edge_acc': []
        }
        
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        
        with torch.no_grad():
            visualize = random.sample(range(len(self.test_data)), num_visualizations)
            for i, item in enumerate(tqdm(self.test_data, desc="Evaluating")):
                graph = item['graph'].to(self.device)
                layout = item['layout'].to(self.device)
                objects = item['objects']

                # Get predictions and match shapes
                pred_layout = self.model(graph)
                if pred_layout.size(0) != layout.size(0):
                    pred_layout = pred_layout[:layout.size(0)]  # Truncate to match
                
                # Calculate metrics
                metrics['iou'].append(calculate_iou(pred_layout, layout))
                
                if 'gt_graph' in item:
                    node_acc, edge_acc = node_edge_accuracy(graph, item['gt_graph'].to(self.device))
                    metrics['node_acc'].append(node_acc)
                    metrics['edge_acc'].append(edge_acc)
                
                # Visualizations
                if save_dir and i in visualize:
                    save_visualization(
                        objects, 
                        pred_layout.cpu(), 
                        layout.cpu(),
                        os.path.join(save_dir, f"sample_{i}.png")
                    )
        
        # Average metrics properly
        avg_metrics = {
            'iou': torch.cat(metrics['iou']).mean().item() if metrics['iou'] else 0,
            'node_acc': sum(metrics['node_acc'])/len(metrics['node_acc']) if metrics['node_acc'] else 0,
            'edge_acc': sum(metrics['edge_acc'])/len(metrics['edge_acc']) if metrics['edge_acc'] else 0
        }
        return avg_metrics

if __name__ == "__main__":
    from models.gat_model import GATLayoutPredictor
    
    parser = argparse.ArgumentParser(description='Evaluate GAT Layout Predictor')
    
    parser.add_argument('--data_path', type=str, required=True,
                      help='Path to the test dataset')
    parser.add_argument('--checkpoint_path', type=str, required=True,
                      help='Path to the pre-trained GAT model checkpoint')
    parser.add_argument('--output_dir', type=str, default=None,
                      help='Directory to save evaluation results and visualizations')
    parser.add_argument('--num_node_features', type=int, default=1000,
                      help='Dimension of node features')
    parser.add_argument('--num_edge_features', type=int, default=1,
                      help='Dimension of edge features')
    parser.add_argument('--train_ratio', type=float, default=0.75,
                      help='Ratio of training data to total data')
    
    args = parser.parse_args()

    if not args.output_dir:
        output_dir = os.path.join('evaluation_results/', args.checkpoint_path.split('/')[-2], args.checkpoint_path.split('/')[-1].replace('.pt', '_eval'))
    else:
        output_dir = args.output_dir
    print(f"Output directory: {output_dir}")

    # Example usage
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = GATLayoutPredictor(
        num_node_features=args.num_node_features,
        num_edge_features=args.num_edge_features
    ).to(device)

    # "data/visual_genome/processed/vg_processed.pt"
    evaluator = Evaluator(model, args.checkpoint_path, args.data_path, device)
    metrics = evaluator.evaluate(save_dir=output_dir)

    print("\n📊 Evaluation Results:")
    print(f"Mean IoU: {metrics['iou']:.4f}")
    print(f"Node Accuracy: {metrics['node_acc']:.2%}")
    print(f"Edge Accuracy: {metrics['edge_acc']:.2%}")