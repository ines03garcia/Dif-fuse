image_file, patient_id, y_sample = self.sample_idx_to_scan_path_and_label[item]

    im_path = os.path.join(self.dataset_root_folder_filepath, patient_id, image_file)
    image = imageio.imread(im_path)
    image = torch.tensor(image, dtype=torch.float32)

    if self.transform is not None:
        image = self.transform(image)

    max_val = torch.max(image)
    if max_val > 0:
        image = image / max_val

    image = image.unsqueeze(0)  # Add channel dimension