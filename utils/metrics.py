import numpy as np
import networkx as nx
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
from torch_geometric.utils import to_networkx
from itertools import permutations

def layout_loss(pred, target, eps=1e-8):
    """Composite loss for layout prediction with numerical stability"""
    target = torch.clamp(target, 0.01, 0.99)
    # 1. Coordinate loss (Smooth L1 for robustness)
    coord_loss = F.smooth_l1_loss(pred[:, :2], target[:, :2])
    
    # 2. Size loss (log-space with safeguards)
    # Add small epsilon to prevent log(0) and ensure positive values
    pred_size = torch.clamp(pred[:, 2:], min=eps)
    target_size = torch.clamp(target[:, 2:], min=eps)
    size_loss = F.mse_loss(torch.log(pred_size), torch.log(target_size))
    
    total_loss = coord_loss + 0.7*size_loss
    
    # Check for NaN before returning
    if torch.isnan(total_loss).any():
        print("Warning: NaN detected in loss!")
        print("Pred:", pred)
        print("Target:", target)
        # You might want to return a fallback loss here
    
    return total_loss

def calculate_iou(bboxes1, bboxes2):
    """
    Calculate IoU between two sets of bounding boxes
    Args:
        bboxes1: Tensor of shape [N, 4] (x, y, w, h)
        bboxes2: Tensor of shape [N, 4] (x, y, w, h)
    Returns:
        iou: Tensor of shape [N]
    """
    # Ensure correct shape
    if bboxes1.dim() == 1:
        bboxes1 = bboxes1.unsqueeze(0)
    if bboxes2.dim() == 1:
        bboxes2 = bboxes2.unsqueeze(0)
    
    # Extract coordinates
    x1 = bboxes1[..., 0]
    y1 = bboxes1[..., 1]
    w1 = bboxes1[..., 2]
    h1 = bboxes1[..., 3]
    
    x2 = bboxes2[..., 0]
    y2 = bboxes2[..., 1]
    w2 = bboxes2[..., 2]
    h2 = bboxes2[..., 3]
    
    # Calculate intersection
    x_left = torch.max(x1, x2)
    y_top = torch.max(y1, y2)
    x_right = torch.min(x1 + w1, x2 + w2)
    y_bottom = torch.min(y1 + h1, y2 + h2)
    
    intersection = torch.clamp(x_right - x_left, min=0) * torch.clamp(y_bottom - y_top, min=0)
    
    # Calculate union
    area1 = w1 * h1
    area2 = w2 * h2
    union = area1 + area2 - intersection
    
    # Avoid division by zero
    return intersection / (union + 1e-6)

def node_edge_accuracy(pred_graph, gt_graph):
    """
    Calculate node and edge classification accuracy
    Returns: (node_accuracy, edge_accuracy)
    """
    # Node accuracy
    node_acc = 0.0
    if hasattr(gt_graph, 'x') and hasattr(pred_graph, 'x'):
        if gt_graph.x is not None and pred_graph.x is not None:
            node_acc = accuracy_score(
                gt_graph.x.cpu().numpy(),
                pred_graph.x.cpu().numpy()
            )
    
    # Edge accuracy
    edge_acc = 0.0
    if hasattr(gt_graph, 'edge_attr') and hasattr(pred_graph, 'edge_attr'):
        if gt_graph.edge_attr is not None and pred_graph.edge_attr is not None:
            edge_acc = accuracy_score(
                gt_graph.edge_attr.cpu().numpy(),
                pred_graph.edge_attr.cpu().numpy()
            )
    
    return node_acc, edge_acc

def clip_similarity(text, rendered_layout):
    """
    Calculate CLIP similarity between text and rendered layout
    Note: Requires CLIP model to be loaded separately
    """
    # Implementation would go here
    raise NotImplementedError("CLIP similarity needs separate implementation")

def layout_diversity(layouts):
    """
    Calculate diversity score for multiple layouts of the same prompt
    """
    if len(layouts) < 2:
        return 0.0
    
    pairwise_dists = []
    for a, b in permutations(layouts, 2):
        pairwise_dists.append(np.linalg.norm(a - b))
    
    return np.mean(pairwise_dists)