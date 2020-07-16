import os
import time

import numpy as np
import pandas
import torch.utils.data as data
from PIL import Image


# 继承自torch.utils.data.Dataset
class pascal3d(data.Dataset):
    """
        Construct a Pascal Dataset.
        Inputs:
            csv_path    path containing instance data
            augment     boolean for flipping images
    """

    def __init__(self, csv_path, dataset_root=None, im_size=227, transform=None, just_easy=False, num_classes=12):

        start_time = time.time()

        # Load instance data from csv-file
        # 读取csv_path中的数据集信息返回为四个列表
        # return image_paths, bboxes, obj_class, viewpoints
        im_paths, bbox, obj_cls, vp_labels = self.csv_to_instances(csv_path)
        print("csv file length: ", len(im_paths))

        # dataset parameters
        self.root = dataset_root
        self.loader = self.pil_loader

        # 将csv数据存储为成员变量:
        self.im_paths = im_paths
        self.bbox = bbox
        self.obj_cls = obj_cls
        self.vp_labels = vp_labels

        self.flip = [False] * len(im_paths)

        # 将数据集其他信息储存为成员变量
        self.im_size = im_size
        self.num_classes = num_classes
        self.num_instances = len(self.im_paths)

        # 必须要有transform
        # 取train_transform,test_transform其中一个
        assert transform != None
        self.transform = transform

        # Set weights for loss
        # 统计数据集中各个类别的数量,
        # histogram返回一个长度为2的tuple,第一个元素为直方图的值,第二个元素为直方图的横坐标刻度
        class_hist = np.histogram(obj_cls, list(range(0, self.num_classes + 1)))[0]
        mean_class_size = np.mean(class_hist)
        # 根据数据集中各类数据占比来定义各类的损失权重，占比越高，权重越小
        self.loss_weights = mean_class_size / class_hist

        # Print out dataset stats
        print("Dataset loaded in ", time.time() - start_time, " secs.")
        print("Dataset size: ", self.num_instances)

    # All subclasses should overwrite :meth:`__getitem__`, supporting fetching a data sample for a given key.
    def __getitem__(self, index):
        """
            Args:
            index (int): Index
            Returns:
            tuple: (image, target) where target is class_index of the target class.
        """

        # Load and transform image
        # 提取dataset[index]
        if self.root == None:
            im_path = self.im_paths[index]
        else:
            im_path = os.path.join(self.root, self.im_paths[index])

        # 按照输入的index返回ground truth
        bbox = self.bbox[index]
        obj_cls = self.obj_cls[index]
        view = self.vp_labels[index]
        flip = self.flip[index]

        # Transform labels
        # 由int转换为float
        azim, elev, tilt = (view + 360.) % 360.

        # Load and transform image
        # 使用成员函数loader()读取图片
        # self.loader = self.pil_loader
        img = self.loader(im_path, bbox=bbox, flip=flip)
        # 使用成员函数transform()转换图片
        # self.transform = transform
        if self.transform is not None:
            img = self.transform(img)

        # construct unique key for statistics -- only need to generate imid and year
        # _bb:框的四个值转换为字符串模式
        _bb = str(bbox[0]) + '-' + str(bbox[1]) + '-' + str(bbox[2]) + '-' + str(bbox[3])
        # key_uid:每个图片一个独特的id
        key_uid = self.im_paths[index] + '_' + _bb + '_objc' + str(obj_cls) + '_kpc' + str(0)

        return img, azim, elev, tilt, obj_cls, -1, -1, key_uid

    # Subclasses could also optionally overwrite:meth:`__len__`, which is expected to return the size of the dataset
    def __len__(self):
        return self.num_instances

    """
        Loads images and applies the following transformations
            1. convert all images to RGB
            2. crop images using bbox (if provided)
            3. resize using LANCZOS to rescale_size
            4. convert from RGB to BGR
            5. (? not done now) convert from HWC to CHW
            6. (optional) flip image

        TODO: once this works, convert to a relative path, which will matter for
              synthetic data dataset class size.
    """

    def pil_loader(self, path, bbox=None, flip=False):
        # open path as file to avoid ResourceWarning
        # link: (https://github.com/python-pillow/Pillow/issues/835)
        with open(path, 'rb') as f:
            # from PIL import Image
            with Image.open(f) as img:
                # 创建一个img的副本以防止更改原文件
                img = img.convert('RGB')

                # Convert to BGR from RGB
                # 将img的rgb通道分解为三个单通道
                r, g, b = img.split()
                # 将分解的三个通道安装bgr的顺序重组
                img = Image.merge("RGB", (b, g, r))
                # 按照bbox裁剪
                img = img.crop(box=bbox)

                # verify that imresize uses LANCZOS
                # 使用pascal3d类定义的im_size调整图片大小
                img = img.resize((self.im_size, self.im_size), Image.LANCZOS)

                # flip image
                if flip:
                    img = img.transpose(Image.FLIP_LEFT_RIGHT)

                return img

    # im_paths, bbox, obj_cls, vp_labels = self.csv_to_instances(csv_path)
    def csv_to_instances(self, csv_path):
        df = pandas.read_csv(csv_path, sep=',')
        data = df.values
        # 将读取的数据分割(im_paths, bbox, obj_cls, vp_labels)
        data_split = np.split(data, [0, 1, 5, 6, 9], axis=1)
        # 删除掉第一个元素[]
        del (data_split[0])

        image_paths = np.squeeze(data_split[0]).tolist()
        bboxes = data_split[1].tolist()
        obj_class = np.squeeze(data_split[2]).tolist()
        viewpoints = np.array(data_split[3].tolist())

        return image_paths, bboxes, obj_class, viewpoints
