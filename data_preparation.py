import os
import argparse
import random
import shutil
from PIL import Image, ImageEnhance, ImageOps
import pillow_heif

# Register the HEIF opener with Pillow
pillow_heif.register_heif_opener()

def create_augmentations(image_path):
    """
    Creates a list of 5 augmented images from a single image path.
    The first image is the resized original, and the others are distorted.
    """
    try:
        img = Image.open(image_path).convert("RGB")
    except IOError:
        print(f"Error opening {image_path}. Skipping.")
        return []

    # 1. Just resize the original
    base_image = img.resize((256, 256))
    images = [base_image]

    # 2. Rotate
    rotated_image = base_image.rotate(random.uniform(-10, 10), resample=Image.BICUBIC, expand=False)
    images.append(rotated_image)

    # 3. Adjust Brightness
    enhancer = ImageEnhance.Brightness(base_image)
    bright_image = enhancer.enhance(random.uniform(0.8, 1.2))
    images.append(bright_image)
    
    # 4. Adjust Contrast
    enhancer = ImageEnhance.Contrast(base_image)
    contrast_image = enhancer.enhance(random.uniform(0.8, 1.2))
    images.append(contrast_image)

    # 5. Horizontal Flip
    flipped_image = ImageOps.mirror(base_image)
    images.append(flipped_image)

    return images


def process_images(source_dir, dest_dir, file_list):
    """
    Processes a list of image files from a source directory, creates augmentations,
    and saves them to a destination directory.
    """
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    for i, filename in enumerate(file_list):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.heic')):
            image_path = os.path.join(source_dir, filename)
            augmented_images = create_augmentations(image_path)
            
            base_name, _ = os.path.splitext(filename)
            
            for j, aug_img in enumerate(augmented_images):
                new_filename = f"{base_name}_aug_{j}.png"
                aug_img.save(os.path.join(dest_dir, new_filename))
        
        # Simple progress indicator
        print(f"\rProcessing {os.path.basename(source_dir)}: {i+1}/{len(file_list)}", end="")
    print()


def main(input_dir, output_dir, train_split):
    """
    Main function to orchestrate the image processing and file organization.
    """
    # Define source directories
    source_good_dir = os.path.join(input_dir, 'good')
    source_defect_dir = os.path.join(input_dir, 'defect')

    # Define destination directories
    dest_train_good_dir = os.path.join(output_dir, 'pump_house', 'train', 'good')
    dest_test_good_dir = os.path.join(output_dir, 'pump_house', 'test', 'good')
    dest_test_defect_dir = os.path.join(output_dir, 'pump_house', 'test', 'defect')
    
    # Clean up output directory if it exists
    if os.path.exists(output_dir):
        print(f"Removing existing output directory: {output_dir}")
        shutil.rmtree(output_dir)

    # Create directories
    for path in [dest_train_good_dir, dest_test_good_dir, dest_test_defect_dir]:
        os.makedirs(path, exist_ok=True)
        
    # --- Process 'good' images and split them ---
    good_images = [f for f in os.listdir(source_good_dir) if not f.startswith('.')]
    random.shuffle(good_images)
    
    split_index = int(len(good_images) * train_split)
    train_files = good_images[:split_index]
    test_files = good_images[split_index:]
    
    print(f"Found {len(good_images)} 'good' images. Splitting into {len(train_files)} training and {len(test_files)} testing images.")
    process_images(source_good_dir, dest_train_good_dir, train_files)
    process_images(source_good_dir, dest_test_good_dir, test_files)
    
    # --- Process 'defect' images ---
    defect_images = [f for f in os.listdir(source_defect_dir) if not f.startswith('.')]
    print(f"Found {len(defect_images)} 'defect' images. Moving to test/defect.")
    process_images(source_defect_dir, dest_test_defect_dir, defect_images)

    print("\nData preparation complete!")
    print(f"Processed images are located in: {os.path.join(output_dir, 'pump_house')}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Prepare and augment image data for anomaly detection training.")
    parser.add_argument('--input-dir', type=str, default='unprocessed', help='Directory containing the raw "good" and "defect" image folders.')
    parser.add_argument('--output-dir', type=str, default='my_dataset', help='Directory where the structured dataset will be saved.')
    parser.add_argument('--train-split', type=float, default=0.8, help='The proportion of "good" images to use for the training set (0.0 to 1.0).')
    
    args = parser.parse_args()
    
    main(args.input_dir, args.output_dir, args.train_split) 