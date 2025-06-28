import os
import argparse
from datetime import datetime
from tqdm import tqdm
from dotmap import DotMap

import torch
from torch_geometric.data import Data

from models.gat_model import GATLayoutPredictor
from models.text_encoder import BERTTextEncoder
from utils.scene_graph_from_text import extract_scene_graph

import matplotlib
import matplotlib.pyplot as plt

def get_current_time():
    now = datetime.now().strftime("%m-%d-%H%M%S")
    return now

def main(args=None):
    """config"""
    config = DotMap()
    config.update(vars(args))
    config.device = f"cuda:{args.gpu}"
    
    # --- Setup device ---
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # --- Samples
    sample = {
    }

    # --- Init encoder & model ---
    encoder = BERTTextEncoder(device=device)
    model = GATLayoutPredictor().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.MSELoss()

    # --- Parse scene graph ---
    nodes, edges = extract_scene_graph(sample["text"])
    node_features = {n: encoder.encode(n).squeeze(0) for n in nodes}

    # --- Build graph ---
    node_idx = {n: i for i, n in enumerate(nodes)}
    edge_index = []
    for src, tgt, _ in edges:
        if src in node_idx and tgt in node_idx:
            edge_index.append([node_idx[src], node_idx[tgt]])
    if not edge_index:
        edge_index = [[0, 0]]  # dummy self-loop
    edge_index = torch.tensor(edge_index, dtype=torch.long, device=device).t().contiguous()
    x = torch.stack([node_features[n] for n in nodes]).to(device)
    y_true = torch.tensor(sample["layout"][:len(nodes)], dtype=torch.float, device=device)

    # --- Checkpoint setup ---
    start_time = get_current_time()
    ckpt_dir = f"gat-{start_time}"
    os.makedirs(ckpt_dir, exist_ok=True)

    # --- Training loop ---
    step = 0
    losses = []
    with tqdm(initial=step, total=config.train_num_steps) as pbar:
        while step < config.train_num_steps:
            if step % config.log_interval == 0:
                # Save checkpoint
                plt.plot(losses)
                plt.savefig(f"{config.ckpt_dir}/loss.png")
                plt.close()
                ckpt_path = os.path.join(ckpt_dir, f"checkpoint_{step}.pt")
                torch.save({
                    'step': step,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss.item(),
                }, ckpt_path)

            model.train()
            optimizer.zero_grad()
            y_pred = model(x, edge_index)
            loss = criterion(y_pred, y_true)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

            # Progress bar
            step += 1
            pbar.update(1)
            pbar.set_postfix(loss=loss.item())
        pbar.close()

    print("Training complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--num_steps", type=int, default=10000, help="Number of training steps")
    parser.add_argument("--ckpt_dir", type=str, default="gat_checkpoints", help="Directory to save checkpoints")
    parser.add_argument("--log_interval", type=int, default=200)
    parser.add_argument("--train_num_steps", type=int, default=10000, help="Total number of training steps")
    args = parser.parse_args()
    main(args)