import os

def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        print('Creating directory ' + directory)
        os.makedirs(directory) 
