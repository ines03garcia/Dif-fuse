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
            only_negative = True

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
        image_file, _, y_sample = self.sample_idx_to_scan_path_and_label[item]

        im = os.path.join(self.dataset_root_folder_filepath, image_file)
        image = imageio.imread(im)
        image = torch.tensor(image, dtype=torch.float32)
        
        if self.transform is not None:
            image = self.transform(image)

        image = image / torch.max(image)
        image = image.unsqueeze(0)  # [1, H, W]
        # torch.Size([1, 1520, 912])


        # ATENCAO - estou a dar return do tensor
        return image, y_sample