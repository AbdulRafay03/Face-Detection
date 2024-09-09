import os
import shutil
from logger import setup_logging
from sklearn.model_selection import train_test_split


logger = setup_logging('application.log')

class inital_preprocessing:

    def __init__(self, folderpath):
        self.folderpath = folderpath
        self.__check_folder_exists()
        
        self.output_path = folderpath + "_Split"
        self.split(self.output_path)

    #check if the folder path exists
    def __check_folder_exists(self):
        try:
            if not os.path.exists(self.folderpath):
                raise FileNotFoundError(f"Folder '{self.folderpath}' does not exist.")
            else:
                logger.debug("Dataset Folder Found")
        except x as e:
            logger.exception("Folder Not Found" , exc_info=True)

    #split the data into train test and validation sets and save them in a new folder
    def split(self,output_path , train_size = 0.7, test_size = 0.15, val_size = 0.15 ):
        try:
            if train_size + val_size + test_size != 1.0: #sum of ratios should be 1.0
                raise ValueError("Ratios must sum to 1.0")
        except ValueError as e:
              logger.exception("Ratios dont sum-up to 1.0" , exc_info=True)

        try:
            classes = os.listdir(self.folderpath) #get the classes from the orignal folder
        except NotADirectoryError as e:
            logger.exception("Not a directory", exc_info=True)
            return 
        
        for cls in classes:
            cls_dir = os.path.join(self.folderpath, cls)

            try:
                images = [img for img in os.listdir(cls_dir) if os.path.isfile(os.path.join(cls_dir, img))]
            except NotADirectoryError as e:
                logger.exception(f"Error accessing directory: {cls_dir}", exc_info=True)
                continue
            
            #spliting the dataset
            train_images, temp = train_test_split(images, train_size=train_size)
            val_images, test_images = train_test_split(temp, test_size=(test_size / (test_size + val_size)))
            
            self.__copy_images(cls_dir, train_images, os.path.join(output_path, 'train', cls))
            logger.info(f"{cls} train set  done")
            self.__copy_images(cls_dir, val_images, os.path.join(output_path, 'val', cls))
            logger.info(f"{cls} validation set  done")
            self.__copy_images(cls_dir, test_images, os.path.join(output_path, 'test', cls))
            logger.info(f"{cls} test set  done")

    def __copy_images(self,src_dir, images, dst_dir):
        if not os.path.exists(dst_dir):
            os.makedirs(dst_dir)
        for img in images:
            src_path = os.path.join(src_dir, img)
            dst_path = os.path.join(dst_dir, img)
            shutil.copy(src_path, dst_path)
        
       




