"""
Making a mini COCO dataset from the full COCO dataset with balanced classes
"""

import json
from pathlib import Path
from collections import defaultdict
import random

def make_coco_mini_balanced(ann_path: Path, out_path: Path, images_per_class: int = 250,
                            random_seed: int = 42,):
    """
    Create a mini COCO dataset with balanced classes.
    
    Args:
        ann_path: Path to original COCO annotation file
        out_path: Path to save mini COCO annotation file
        images_per_class: Number of images per category (default 250)
        random_seed: Random seed for reproducibility
    """
    random.seed(random_seed)
    
    # Load original annotations
    ann = json.loads(ann_path.read_text())
    images = ann["images"]
    annotations = ann["annotations"]
    categories = ann["categories"]
    
    # Create mappings for easier lookup
    image_dict = {img["id"]: img for img in images}
    category_dict = {cat["id"]: cat for cat in categories}
    
    # Group annotations by image_id and category_id
    image_annotations = defaultdict(lambda: defaultdict(list))
    image_categories = defaultdict(set) # image_id -> set of category_ids
    
    for ann in annotations:
        img_id = ann["image_id"]
        cat_id = ann["category_id"]
        image_annotations[img_id][cat_id].append(ann)
        image_categories[img_id].add(cat_id)
    
    # Count images per category
    category_image_counts = defaultdict(set) # category_id -> set of image_ids
    
    for img_id, cats in image_categories.items():
        for cat_id in cats:
            category_image_counts[cat_id].add(img_id)
    
    # Print initial statistics
    print("Original dataset statistics:")
    print(f"Total images: {len(images)}")
    print(f"Total categories: {len(categories)}")
    for cat_id, img_ids in sorted(category_image_counts.items()):
        cat_name = category_dict[cat_id]["name"]
        print(f"  Category {cat_id} ({cat_name}): {len(img_ids)} images")
    
    # Select balanced set of images
    selected_image_ids = set()
    
    # For each category, select images_per_class images
    for cat_id in sorted(category_dict.keys()):
        cat_name = category_dict[cat_id]["name"]
        available_images = list(category_image_counts[cat_id])
        
        if len(available_images) < images_per_class:
            print(f"Warning: Category {cat_id} ({cat_name}) has only {len(available_images)} images, "
                  f"requested {images_per_class}. Using all available.")
            selected_for_cat = available_images
        else:
            # Randomly select images for this category
            selected_for_cat = random.sample(available_images, images_per_class)
        
        selected_image_ids.update(selected_for_cat)
        print(f"Selected {len(selected_for_cat)} images for category {cat_id} ({cat_name})")
    
    # Convert to list and shuffle
    selected_image_ids = list(selected_image_ids)
    random.shuffle(selected_image_ids)
    
    # Limit to exactly images_per_class * num_categories if we have more
    max_images = images_per_class * len(categories)
    if len(selected_image_ids) > max_images:
        print(f"Note: Selected {len(selected_image_ids)} unique images, "
              f"keeping first {max_images} to maintain balance.")
        selected_image_ids = selected_image_ids[:max_images]
    
    # Create mini dataset
    mini_images = [image_dict[img_id] for img_id in selected_image_ids]
    
    # Collect all annotations for selected images
    mini_annotations = []
    for img_id in selected_image_ids:
        for cat_id, anns in image_annotations[img_id].items():
            mini_annotations.extend(anns)
    
    # Create final mini COCO dataset
    mini = {
        "info": ann.get("info", {}),
        "licenses": ann.get("licenses", []),
        "images": mini_images,
        "annotations": mini_annotations,
        "categories": categories,
    }
    
    # Save to file
    out_path.write_text(json.dumps(mini))
    
    # Print final statistics
    print("\nMini dataset statistics:")
    print(f"Saved mini COCO to {out_path}")
    print(f"Total images: {len(mini_images)}")
    print(f"Total annotations: {len(mini_annotations)}")
    
    # Count images per category in mini dataset
    mini_category_counts = defaultdict(set)
    for ann in mini_annotations:
        mini_category_counts[ann["category_id"]].add(ann["image_id"])
    
    print("\nImages per category in mini dataset:")
    for cat_id, img_ids in sorted(mini_category_counts.items()):
        cat_name = category_dict[cat_id]["name"]
        print(f"  Category {cat_id} ({cat_name}): {len(img_ids)} images")
    
    # Calculate overlap statistics
    total_images = len(selected_image_ids)
    print(f"\nTotal unique images selected: {total_images}")
    print(f"Target: {images_per_class} images per category × {len(categories)} categories = "
          f"{images_per_class * len(categories)} images")


if __name__ == "__main__":
    root = Path("data/coco/annotations")
    ann_path = root / "instances_train2017.json"
    out_path = root / "instances_train2017_mini.json"

    # 800 images per class × 80 classes = 64,000 images
    make_coco_mini_balanced(
        ann_path=ann_path,
        out_path=out_path,
        images_per_class=800,
        random_seed=42
    )