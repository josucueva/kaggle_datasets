import os
import glob
import shutil
import pandas as pd


def has_single_csv(path):
    """check if directory contains exactly one csv file"""
    archivos_csv = glob.glob(os.path.join(path, "*.csv"))
    return len(archivos_csv) == 1, archivos_csv


def get_csv_path(path):
    """get the csv file path from a directory"""
    is_single, archivos = has_single_csv(path)
    if is_single:
        return archivos[0]
    return None


def read_dataset_sample(csv_path, nrows=5000):
    """read a sample of the dataset"""
    return pd.read_csv(csv_path, nrows=nrows)


def copy_dataset(source, destination):
    """copy dataset directory to destination"""
    if os.path.exists(destination):
        shutil.rmtree(destination)
    shutil.copytree(source, destination)


def ensure_dir(path):
    """create directory if it doesn't exist"""
    os.makedirs(path, exist_ok=True)
