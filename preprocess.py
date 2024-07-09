import os
import cv2
import shutil
import torch
from torchvision import datasets, transforms
from logger import setup_logging

logger = setup_logging('application.log')

class preprocess:

    DataLoaders = []

    def __init__(self , input_path,output_path,batchSize, processing_required):
        self.input = input_path
        self.output = output_path
        self.batch_size = batchSize

        self.__check_folder_exists(self.input)
        self.__check_folder_exists(self.output)

        self.target_size = (224, 224) # Define the target size for images

        if processing_required:
            for i in ['train' , 'test' , 'val']:
                out = self.__processing(i)
                logger.info(f"{i} processed")
                self.DataLoaders.append(self.createLoader(out))
                logger.info(f"{i} Loader created")
        else:
            for i in ['train' , 'test' , 'val']:
                out = os.path.join(output_path , i)
                self.DataLoaders.append(self.createLoader(out))
                logger.info(f"{i} Loader created")



    #check if the folder path exists
    def __check_folder_exists(self,path):
        try:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Folder '{path}' does not exist.")
            else:
                logger.debug("Input Folder Found")
        except FileNotFoundError as e:
            logger.exception("Input Folder Not Found" , exc_info=True)


# Process images and save them to the output directory

    def __processing(self,set_name):

        set_path = os.path.join(self.input, set_name)
        output_path = os.path.join(self.output, set_name) # Define the output path for processed data
        logger.info(output_path)
        os.makedirs(output_path, exist_ok=True) # Create the output directory if it doesn't exist

        try:
            classes = os.listdir(set_path) # Get the classes from the folder
        except NotADirectoryError as e:
            logger.exception("Not a directory", exc_info=True)
            return 

        for class_name in classes:
            input_class_path = os.path.join(set_path, class_name)
            output_class_path = os.path.join(output_path, class_name.lower())
            os.makedirs(output_class_path, exist_ok=True)
            for filename in os.listdir(input_class_path):
                image_path = os.path.join(input_class_path, filename)
                output_image_path = os.path.join(output_class_path, filename)
                image = cv2.imread(image_path)
                denoised_image = cv2.GaussianBlur(image, (5, 5), 0)
                resized_denoised_image = cv2.resize(denoised_image, self.target_size)
                cv2.imwrite(output_image_path, resized_denoised_image)

        return output_path
        

    def createLoader(self,input_path):
        
        data_dir = input_path

        # Define the transforms for data augmentation and normalization
        data_transforms = transforms.Compose([
            transforms.RandomRotation(20),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # Load the dataset using ImageFolder and apply the defined transforms
        dataset = datasets.ImageFolder(root=data_dir, transform=data_transforms)        

        # Create a DataLoader for the training dataset
        loader = torch.utils.data.DataLoader(dataset, batch_size= self.batch_size, shuffle=True)

        return loader




