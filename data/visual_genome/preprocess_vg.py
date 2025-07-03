import json
import os
import torch
from torch_geometric.data import Data
from tqdm import tqdm  # for progress bar

def preprocess_visual_genome():
    """
    Preprocesses the Visual Genome dataset to generate graph-based representations for each image.
    This function loads Visual Genome image metadata, object annotations, and relationship annotations from JSON files,
    filters and validates the data, constructs graph structures suitable for PyTorch Geometric (PyG), and normalizes
    bounding box coordinates. Each processed image is represented as a dictionary containing the graph, object layouts,
    object names, object IDs, image ID, and image dimensions.
    Returns:
        List[dict]: A list of dictionaries, each containing:
            - 'graph': PyG Data object with node features, edge indices, and edge attributes.
            - 'layout': Tensor of normalized bounding box coordinates for valid objects.
            - 'objects': List of object names (first name per object).
            - 'object_ids': List of object IDs corresponding to valid objects.
            - 'image_id': The image ID.
            - 'width': Image width.
            - 'height': Image height.
    Prints:
        The number of images skipped due to missing or invalid data.
    """
    # Load all data first
    with open('raw/image_data.json') as f:
        image_data = json.load(f)
    with open('raw/objects.json') as f:
        all_objects = json.load(f)
    with open('raw/relationships.json') as f:
        all_relationships = json.load(f)
    
    # Create image_id to data mapping
    image_objects = {img['image_id']: img for img in all_objects}
    image_relationships = {img['image_id']: img for img in all_relationships}
    
    processed_data = []
    skipped_images = 0
    
    for img_meta in tqdm(image_data, desc="Processing Images"):
        img_id = img_meta['image_id']
        
        # Skip if no objects or relationships
        if img_id not in image_objects or img_id not in image_relationships:
            skipped_images += 1
            continue
            
        img_objects = image_objects[img_id]['objects']
        img_rels = image_relationships[img_id]['relationships']
        
        # Create object mapping with validation
        obj_id_to_idx = {}
        valid_objects = []
        
        for idx, obj in enumerate(img_objects):
            if not obj['names']:  # Skip objects with no names
                continue
            obj_id_to_idx[obj['object_id']] = idx
            valid_objects.append(obj)
        
        # Process relationships (only between valid objects)
        edge_index = []
        edge_attr = []
        rel_types = set()
        
        for rel in img_rels:
            try:
                subj_id = rel['subject']['object_id']
                obj_id = rel['object']['object_id']
                
                if subj_id in obj_id_to_idx and obj_id in obj_id_to_idx:
                    rel_type = rel['predicate'].lower().strip()
                    rel_types.add(rel_type)
                    edge_index.append([obj_id_to_idx[subj_id], obj_id_to_idx[obj_id]])
                    edge_attr.append(rel_type)
            except KeyError:
                continue
        
        if not edge_index:  # Skip images with no valid relationships
            skipped_images += 1
            continue
            
        # Create numerical edge attributes
        rel_type_to_idx = {t: i for i, t in enumerate(sorted(rel_types))}
        edge_attr = [rel_type_to_idx[t] for t in edge_attr]
        
        # Create PyG Data object
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attr, dtype=torch.long)
        
        # Get bounding boxes (only for valid objects)
        boxes = torch.tensor(
            [[obj['x'], obj['y'], obj['w'], obj['h']] 
            for obj in valid_objects],
            dtype=torch.float
        )
        
        # Normalize coordinates
        boxes[:, 0] /= img_meta['width']   # x
        boxes[:, 1] /= img_meta['height']  # y
        boxes[:, 2] /= img_meta['width']   # w
        boxes[:, 3] /= img_meta['height']  # h
        
        processed_data.append({
            'graph': Data(
                x=torch.arange(len(valid_objects)),  # Using indices as placeholder features
                edge_index=edge_index,
                edge_attr=edge_attr
            ),
            'layout': boxes,
            'objects': [obj['names'][0] for obj in valid_objects],
            'object_ids': [obj['object_id'] for obj in valid_objects],
            'image_id': img_id,
            'width': img_meta['width'],
            'height': img_meta['height']
        })
    
    print(f"Skipped {skipped_images} images due to missing/invalid data")
    return processed_data

if __name__ == "__main__":
    processed = preprocess_visual_genome()
    os.makedirs("preprocessed", exist_ok=True)
    torch.save(processed, os.path.join("preprocessed/vg_processed.pt"))