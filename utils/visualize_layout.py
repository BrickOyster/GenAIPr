import matplotlib.pyplot as plt
import matplotlib.patches as patches

def save_visualization(objects, pred_boxes, gt_boxes, save_path, loss=""):
        """Save comparison visualization with size validation"""
        import matplotlib.pyplot as plt
        
        # Ensure equal number of boxes
        num_boxes = min(len(pred_boxes), len(gt_boxes), len(objects))
        pred_boxes = pred_boxes[:num_boxes]
        gt_boxes = gt_boxes[:num_boxes]
        objects = objects[:num_boxes]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
        
        # Predicted layout
        for obj, box in zip(objects, pred_boxes):
            x, y, w, h = box.tolist()
            rect = plt.Rectangle((x, y), w, h, fill=False, edgecolor='r', linewidth=2)
            ax1.add_patch(rect)
            ax1.text(x, y, obj, bbox=dict(facecolor='white', alpha=0.5))
        ax1.set_title("Predicted Layout loss=" + loss)
        ax1.set_xlim(0, 1)
        ax1.set_ylim(0, 1)
        
        # Ground truth layout
        for obj, box in zip(objects, gt_boxes):
            x, y, w, h = box.tolist()
            rect = plt.Rectangle((x, y), w, h, fill=False, edgecolor='g', linewidth=2)
            ax2.add_patch(rect)
            ax2.text(x, y, obj, bbox=dict(facecolor='white', alpha=0.5))
        ax2.set_title("Ground Truth Layout")
        ax2.set_xlim(0, 1)
        ax2.set_ylim(0, 1)
        
        plt.savefig(save_path)
        plt.close()