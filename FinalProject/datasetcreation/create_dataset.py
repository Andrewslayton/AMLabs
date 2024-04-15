import os
import shutil
from sklearn.model_selection import train_test_split

def create_dataset(source_folder, output_folder, train_size=0.6, val_size=0.25, test_size=0.15):
    classes = [d for d in os.listdir(source_folder) if os.path.isdir(os.path.join(source_folder, d))]
    assert abs((train_size + val_size + test_size) - 1.0) < 1e-5, "The split ratios must sum to 1!"
    for split in ['train', 'val', 'test']:
        for class_name in classes:
            os.makedirs(os.path.join(output_folder, split, class_name), exist_ok=True)
    for class_name in classes:
        class_dir = os.path.join(source_folder, class_name)
        images = os.listdir(class_dir)
        
        train_files, test_files = train_test_split(images, test_size=(val_size + test_size), random_state=42)
        val_files, test_files = train_test_split(test_files, test_size=test_size / (val_size + test_size), random_state=42)

        def copy_files(files, split_name):
            for file in files:
                src_file = os.path.join(class_dir, file)
                dst_file = os.path.join(output_folder, split_name, class_name, file)
                shutil.copy(src_file, dst_file)
        copy_files(train_files, 'train')
        copy_files(val_files, 'val')
        copy_files(test_files, 'test')
        print(f"Processed {class_name}: {len(train_files)} train, {len(val_files)} val, {len(test_files)} test files.")

if __name__ == "__main__":
    project_root = os.path.dirname(os.path.abspath(__file__)) 
    source_dir = os.path.join(project_root, '..', 'datasets')
    output_dir = os.path.join(project_root, '..', 'datasetcreation', 'data_split')

    create_dataset(source_dir, output_dir)
