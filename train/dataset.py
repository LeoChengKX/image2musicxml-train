import os
import random
import shutil
import pandas as pd


dataset_path = "..\\train_dataset"

def prepare_dataset(split=0.9):
    # Perform a the splitting. 
    files_list = os.listdir(os.path.join(dataset_path, "raw"))
    
    print(f"Dataset size: {int(len(files_list))}.")
    # A split on train and validation set
    train_size = int(len(files_list) * split)
    print(f"Train split: {train_size}. ")
    print(f"Test split: {int(len(files_list)) - train_size}. ")
    train_sample = random.sample(files_list, train_size)

    train_path = os.path.join(dataset_path, "train", "image")
    if os.path.exists(train_path):
        shutil.rmtree(train_path)
    os.mkdir(train_path)

    test_path = os.path.join(dataset_path, "test", "image")
    if os.path.exists(test_path):
        shutil.rmtree(test_path)
    os.mkdir(test_path)
    
    metadata = open(os.path.join(dataset_path, "metadata.txt"), 'w')
    xml_path = os.path.join(dataset_path, "xml")
    for file in files_list:
        if file in train_sample:
            shutil.copy(os.path.join(dataset_path, "raw", file), train_path)
        else:
            shutil.copy(os.path.join(dataset_path, "raw", file), test_path)
        
        # Generate the metadata file
        xml_name = os.path.join(xml_path, file + ".xml")
        if os.path.exists(xml_name):
            with open(xml_name, 'r') as f:
                metadata.write(file + "\t" + f.read() + "\n")
        

if __name__ == '__main__':
    prepare_dataset()
