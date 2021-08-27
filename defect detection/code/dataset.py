# -*- coding: utf-8 -*-
"""
Created on Tue Aug 17 10:23:42 2021

@author: 零
"""

class dataset_defectDetection(Dataset):
    def __init__(self, path, transform=None):
        super(dataset_defectDetection, self).__init__()
        self.path = path
        self.component = ["data", "label"]
        self.transform = transform
        self.path_data = os.path.join(self.path, self.component[0])
        self.path_label = os.path.join(self.path, self.component[1])
        dataList, _, _ = os.walk(self.path_data)
        self.data = [os.path.join(self.path_data, data_name) for data_name in dataList[-1]]
        self.label = [os.path.join(self.path_label, label_name) for label_name in os.listdir(self.path_label)]
        
    def __len__(self):
        return len(self.label)
    
    def __getitem__(self, item):
        # 根据索引生成标签
        path_label = self.label[item]
        with open(path_label, "r", encoding="utf8") as fp:
            label = json.load(fp)
            label_mask = np.zeros((label["imageHeight"], label["imageWidth"]), np.uint8)
            for i in range(len(label["shapes"])):
                label_points = np.array(label["shapes"][i]["points"], np.int32)
                label_mask = cv2.drawContours(label_mask, [label_points], -1, 255, -1)
        
        # 根据标签查找图片(该数据集图片总共1104，带有标签的图片只有394)
        path_image = os.path.join(self.path_data, os.path.splitext(os.path.split(path_label)[-1])[0] + ".jpg")
        image = Image.open(path_image)
        
        # 数据处理
        if self.transform is not None:
            random.seed(7)
            image = self.transform(image) 
            random.seed(7)
            label = self.transform(Image.fromarray(np.uint8(label_mask)))
            label[label >= 0.5] = 1.
            label[label < 0.5] = 0.
        label = torch.cat([label, (1-label)], dim=0)
        
        # 返回图像和标签
        return image, label
    
if __name__ == "__main__":

    path = r"C:\Users\风\Desktop\表面缺陷检测\BSData-main\BSData-main"
    input_size = (256, 512)
    batch_size = 2
    shuffle = True
    num_workers = 0
    pin_memory = True
    drop_last = True

    transform = transforms.Compose([
            transforms.Resize(input_size),
            transforms.ToTensor()
    ])

    image_datasets = dataset_defectDetection(path, transform=transform)
    torch.manual_seed(7) # 设置随机种子以便dataset分割的结果可重复
    image_datasets_split = random_split(image_datasets, [len(image_datasets)//5, len(image_datasets)-len(image_datasets)//5])

    loader = {
        "train": DataLoader(image_datasets_split[1], batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory, drop_last=drop_last),
        "eval": DataLoader(image_datasets_split[0], batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory, drop_last=drop_last), 
    }