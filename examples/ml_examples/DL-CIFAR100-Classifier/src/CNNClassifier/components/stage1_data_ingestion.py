import os
# import urllib.request as request
import requests
import wget
import tarfile
import pickle
import pandas as pd

# from zipfile import ZipFile
# from pathlib import Path
# from tqdm import tqdm
from collections import defaultdict
from CNNClassifier import logger
from CNNClassifier.entity import DataIngestionConfig
from CNNClassifier.utils.utilities import *


class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def download_file(self):
        if not os.path.exists(self.config.local_data_file):
            logger.info("trying to download file")
            download_url = self.config.source_url
            outfile_path = self.config.local_data_file
            try:
                response = requests.get(download_url, stream=True)
                if response.status_code == 200:
                    with open(outfile_path, 'wb') as f:
                        f.write(response.raw.read())
            except:
                wget.download(download_url, out=outfile_path)
            logger.info(f"Downloaded {outfile_path} file successfully!")
        else:
            # logger.info("file already exists")
            logger.info(f"File already exists of the size: {get_size(Path(self.config.local_data_file))}")
            return

    def unzip_targzfile(self):
        targzfile = self.config.local_data_file
        outfolder = self.config.unzip_dir
        # open file
        with tarfile.open(targzfile) as f:
            logger.info(f.getnames())
            # extract files
            f.extractall(outfolder)
        return

    @staticmethod
    def unpickle(file):
        with open(file, 'rb') as fo:
            dict = pickle.load(fo, encoding='latin1')
        return dict

    def get_metadata(self):
        train_set = self.unpickle(os.path.join(self.config.unzip_dir, self.config.trainset_file))
        meta_data = self.unpickle(os.path.join(self.config.unzip_dir, self.config.meta_file))
        # create a data records
        file_names = train_set['filenames']
        fine_labels = train_set['fine_labels']
        coarse_labels = train_set['coarse_labels']
        coarse_names = meta_data['coarse_label_names']
        fine_names = meta_data['fine_label_names']
        images = train_set['data']
        n_images = len(images)
        images = images.reshape(n_images, 3, 32, 32).transpose(0, 2, 3, 1)
        # create a dictionary
        image_dict = defaultdict(list)
        for i in range(n_images):
            img = images[i]
            image_dict['file_name'].append(file_names[i])
            image_dict['fine_labels'].append(fine_labels[i])
            image_dict['coarse_labels'].append(coarse_labels[i])
            image_dict['fine_label_names'].append(fine_names[fine_labels[i]])
            image_dict['coarse_label_names'].append(coarse_names[coarse_labels[i]])
            image_dict['image_height'].append(img.shape[0])
            image_dict['image_width'].append(img.shape[1])
            image_dict['image_channel'].append(img.shape[2])
            image_dict['min_pixel'].append(img.min())
            image_dict['max_pixel'].append(img.max())
        # dict to dataframe
        df = pd.DataFrame.from_dict(image_dict)
        # save the result for EDA analysis
        # df.to_csv(os.path.join(self.config.root_dir, self.config.metadata), index=False)
        df.to_csv(self.config.metadata, index=False)
        return
