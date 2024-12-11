import webdataset as wds
from PIL import Image
from numpy import asarray
from pathlib import Path
import json

# Directory paths pointing to respective datasets.
FMOW_SENTINEL_PATH = "./data/fmow-sentinel"
FMOW_RGB_PATH = "./data/fmow-rgb"

def main():
    # Initialize a Webdataset file.
    sink = wds.TarWriter("dest.tar")

    # Initialize Path objects to the data folders.
    fmow_sentinel = Path(FMOW_SENTINEL_PATH)
    fmow_rgb = Path(FMOW_RGB_PATH)

    # Initialize Path objects to the training folders.
    fmow_sentinel_train = fmow_sentinel / "train"
    fmow_rgb_train_meta = fmow_rgb / "train"
    fmow_rgb_train_images = fmow_rgb / "fmow-rgb-images" / "train"

    # Iterate through each FMOW class.
    # ./fmow-rgb/train/{class}
    fmow_rgb_train_meta_class_paths = [x for x in fmow_rgb_train_meta.iterdir() if x.is_dir()]
    for fmow_rgb_train_meta_class_path in fmow_rgb_train_meta_class_paths:
        
        # Iterate through each instance of the class.
        # ./fmow-rgb/train/{class}/{class}_{instance_id}
        fmow_rgb_train_meta_instance_paths = [x for x in fmow_rgb_train_meta_class_path.iterdir() if x.is_dir()]
        for fmow_rgb_train_meta_instance_path in fmow_rgb_train_meta_instance_paths:

            # Iterate through each label of each instance.
            # ./fmow-rgb/train/{class}/{class}_{instance_id}/{class}_{instance_id}_{label_id}_{rgb/msrgb}.json
            fmow_rgb_train_meta_label_paths = [x for x in fmow_rgb_train_meta_instance_path.iterdir() if x.is_dir()]
            for fmow_rgb_train_meta_label_path in fmow_rgb_train_meta_label_paths:

                # Extract relevant data from file paths.
                path_parts = fmow_rgb_train_meta_label_path.parts
                cls_folder, instance_folder, metadata_label_file = path_parts[-3:]
                cls_name, instance_id, label_id = metadata_label_file.split("_")[:3]
                image_label_file = f"{metadata_label_file.split(".")[0]}.jpg"

                # Navigate to corresponding image file.
                # Convert it into a (h, w, c) numpy array.
                fmow_rgb_train_image_label_path = fmow_rgb_train_images / cls_folder / instance_folder / image_label_file
                fmow_rgb_train_image = Image.open(str(fmow_rgb_train_image_label_path))
                fmow_rgb_train_image_np = asarray(fmow_rgb_train_image)

                # Parse JSON from metadata file.
                with fmow_rgb_train_meta_label_path.open() as json_file:
                    metadata = json.load(json_file)
                
                # Write a training example to the dataset.
                sink.write({
                    "__key__": f"fmow-{cls_name}-{instance_id}",
                    "output.cls": int(label_id),
                    "input.npy": fmow_rgb_train_image_np,
                    "metadata.json": metadata,
                    "multispectral.npy": None
                })

    # Close the Webdataset file.
    sink.close()

if __name__ == "__main__":
    main()