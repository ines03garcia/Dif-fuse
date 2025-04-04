import pandas as pd
import os
import torch
import torch.utils.data as data
import imageio


class VinDrMammoDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            dataset_root_folder_filepath,
            df_path,
            transform = None,
            only_positive = False,
            only_negative = False

    ):
        super(VinDrMammoDataset, self).__init__()
        self.dataset_root_folder_filepath = dataset_root_folder_filepath
        self.df_path = df_path
        self.only_positive = only_positive
        self.only_negative = only_negative
        self.transform = transform

        self.meta_data_data_frame = pd.read_csv(
            self.df_path, encoding="ISO-8859-1"
        )
        if self.only_positive:
            # Positive - benign and malignant tumours
            self.meta_data_data_frame = self.meta_data_data_frame[self.meta_data_data_frame['breast_birads']!='BI-RADS 1']
        if self.only_negative:
            self.meta_data_data_frame = self.meta_data_data_frame[self.meta_data_data_frame['breast_birads']=='BI-RADS 1']

        self.sample_idx_to_scan_path_and_label = []

        self.sample_idx_to_scan_path_and_label = [
            (row["image_id"], row["patient_id"], row["breast_birads"])
            for idx, row in self.meta_data_data_frame.iterrows()
        ]


    def __len__(self):

        return len(self.sample_idx_to_scan_path_and_label)

    def __getitem__(self, item):
        # x_path (image_id) and y_sample (breast_birads)
        image_file, patient_id, y_sample = self.sample_idx_to_scan_path_and_label[item]

        raw_image = []
        im = os.path.join(self.dataset_root_folder_filepath, patient_id, image_file)
        image = imageio.imread(im)
        image = torch.tensor(image, dtype=torch.float32)
        
        if self.transform is not None:
            image = self.transform(image)
        
        image = image / torch.max(image)
        raw_image.append(image)
        
        # Stack to maintain consistent shape (adds a channel dimension)
        im = torch.stack(raw_image)  # [1, H, W]

        # return image
        return im