import os

import numpy as np
import cv2
from PIL import Image
from pycocotools.coco import COCO
import transforms as T
import torch

def get_bbox_from_binary_mask(mask):
    # 查找掩码中为1的位置
    indices = torch.nonzero(mask)

    if len(indices) == 0:
        # 如果掩码中没有为1的位置，则返回空边界框
        return torch.tensor([0.,0.,0.,0.])

    # 提取掩码中为1的位置的最小和最大坐标
    min_x = torch.min(indices[:, 1])
    min_y = torch.min(indices[:, 0])
    max_x = torch.max(indices[:, 1])
    max_y = torch.max(indices[:, 0])

    # 构建边界框 [x_min, y_min, x_max, y_max]
    bbox = torch.tensor([min_x, min_y, max_x, max_y])

    return bbox.numpy()

# # 示例用法
# segmentation = torch.tensor([[0, 0, 0, 0],
#                              [0, 1, 1, 0],
#                              [0, 1, 1, 0],
#                              [0, 0, 0, 0]])
#
# bbox = generate_bbox_from_segmentation(segmentation)
# print(bbox)

def seg2mask(seg,h,w):
    gt = np.zeros((h, w), dtype=np.uint8)
    seg = np.array(seg).reshape(-1, 2)  # [n_points, 2]
    mask = cv2.fillPoly(gt, seg.astype(np.int32)[np.newaxis, :, :], 1)
    return mask


def get_item(idx,coco):
    imgId = idx+1
    image = coco.loadImgs([imgId])[0]
    h, w = image['height'], image['width']
    annId = coco.getAnnIds(imgIds=imgId, iscrowd=False)
    anns = coco.loadAnns(annId)  # 获取所有注释信息(所有区域的分割)
    labels = []
    bboxs = []
    masks = []
    areas = []
    iscrowds = []
    for ann in anns:
        category = ann["category_id"]
        seg = ann['segmentation']
        bbox = ann['bbox']
        #对coco数据集要处理下,因为模型要的是两个坐标，不是坐标加宽高
        bbox[2] = bbox[0]+bbox[2]
        bbox[3] = bbox[1]+bbox[3]
        area = ann['area']
        iscrowd = ann["iscrowd"]
        mask = seg2mask(seg, h, w)
        mask = torch.from_numpy(mask)
        labels.append(category)
        bboxs.append(bbox)
        masks.append(mask)
        areas.append(area)
        iscrowds.append(iscrowd)
    return labels,bboxs,masks,areas,iscrowds,h,w


class landUseDataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms=None):
        self.root = root
        self.transforms = transforms
        self.imgs = list(sorted(os.listdir(os.path.join(root, "images"))))
        self.annotation_file = './mydata/annotations/labels.json'
        self.coco = COCO(self.annotation_file)
        self.imgIds = self.coco.getImgIds()  # 图像ID列表
    def __getitem__(self, idx):
        # load images ad masks
        img_path = os.path.join(self.root, "images", self.imgs[idx])
        # mask_path = os.path.join(self.root, "PedMasks", self.masks[idx])
        img = Image.open(img_path).convert("RGB")

        labels,bboxs,masks,areas,iscrowds,h,w = get_item(idx,self.coco)
        target = {}
        target["boxes"] = torch.tensor(bboxs,dtype=torch.float32)
        target["labels"] = torch.tensor(labels,dtype=torch.int64)
        target["masks"] = torch.tensor(masks,dtype=torch.uint8)
        target["image_id"] = torch.tensor([idx],dtype=torch.int64)
        target["area"] = torch.tensor(areas,dtype=torch.float32)
        target["iscrowd"] = torch.tensor(iscrowds,dtype=torch.int64)

        # if self.transforms is not None:
        #     img, target = self.transforms(img, target)
        # img = torch.tensor(np.array(img))
        # img = img.reshape(-1,h,w)
        # img = img/255.
        t = T.ToTensor()
        img,target = t(np.array(img),target)


        return img, target

    def __len__(self):
        return len(self.imgs)

class google_dataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms=None):
        self.root = root
        self.imgs = list(sorted(os.listdir(os.path.join(root, "images"))))
        self.labels_txt = list(sorted(os.listdir(os.path.join(root, "labelTxt"))))
        #using v1.5 有16种+1背景
        self.names = ['background','plane',
                      'ship', 'storage-tank',
                      'baseball-diamond', 'tennis-court',
                      'basketball-court', 'ground-track-field',
                      'harbor', 'bridge', 'large-vehicle', 'small-vehicle',
                      'helicopter','roundabout', 'soccer-ball-field',
                      'swimming-pool' , 'container-crane']
    def __getitem__(self, idx):
        # load images ad masks
        img_path = os.path.join(self.root, "images", self.imgs[idx])
        label_path = os.path.join(self.root, "labelTxt", self.labels_txt[idx])
        # mask_path = os.path.join(self.root, "PedMasks", self.masks[idx])
        img = Image.open(img_path).convert("RGB")
        h,w  = img.height,img.width
        labels = []
        bboxs = []
        masks = []
        areas = []
        iscrowds = []
        segments = []
        # label_names = []
        with open(label_path, 'r') as file:
            lines = file.readlines()
            for line in lines[2:]:
                ls = line.strip().split(' ')
                segment = ls[:8]
                segment= [int(t) for t in segment]
                label_name = ls[8:9][0]
                iscrowd = ls[9:][0]
                label = self.names.index(label_name)
                mask = seg2mask(segment,h,w)
                bbox = get_bbox_from_binary_mask(torch.as_tensor(mask))
                area = (bbox[3] - bbox[1]) * (bbox[2] - bbox[0])
                areas.append(area)
                bboxs.append(bbox)
                masks.append(mask)
                # 将字段添加到相应的列表中
                segments.append(segment)
                # label_names.append(label_name)
                iscrowds.append(int(0))
                labels.append(label)

        target = {}
        target["boxes"] = torch.as_tensor(bboxs,dtype=torch.float32)
        target["labels"] = torch.as_tensor(labels,dtype=torch.int64)
        target["masks"] = torch.as_tensor(masks,dtype=torch.uint8)
        # target["masks"] = torch.as_tensor(masks)
        target["image_id"] = torch.as_tensor([idx],dtype=torch.int64)
        target["area"] = torch.as_tensor(areas,dtype=torch.float32)
        target["iscrowd"] = torch.as_tensor(iscrowds,dtype=torch.int64)

        t = T.ToTensor()
        img,target = t(np.array(img),target)
        return img, target

    def __len__(self):
        return len(self.imgs)

if __name__ == '__main__':
    dataset = google_dataset('mydata/object_segment/val')
    # print(dataset[0][0])
    ###扫描坏数据！！！
    # for idx in range(len(dataset)):
    #     boxes = dataset[idx][1]['boxes']
    #     if not(len(boxes.shape) == 2 and boxes.shape[-1] == 4):
    #         print(idx) #113
    print(dataset[113][1]['boxes'])
# if __name__ == '__main__':
#     names = ['plane',
#              'ship', 'storage tank',
#              'baseball diamond', 'tennis court',
#              'basketball court', 'ground track field',
#              'harbor', 'bridge', 'large-vehicle', 'small-vehicle',
#              'helicopter', 'roundabout', 'soccer ball field',
#              'swimming pool', 'container crane']
#     label_path = 'mydata/object_segment/val/labelTxt/P0003.txt'
#     img_path = 'mydata/object_segment/val/images/images/P0003.png'
#     img = Image.open(img_path).convert("RGB")
#     h, w = img.height, img.width
#     labels = []
#     bboxs = []
#     masks = []
#     areas = []
#     iscrowds = []
#     segments = []
#
#     with open(label_path, 'r') as file:
#         lines = file.readlines()
#         for line in lines[2:]:
#             ls = line.strip().split(' ')
#             segment = ls[:8]
#             segment = [int(t) for t in segment]
#             label_name = ls[8:9][0]
#             iscrowd = ls[9:][0]
#             label = names.index(label_name)
#             mask = seg2mask(segment, h, w)
#             mask = torch.tensor(mask)
#             bbox = get_bbox_from_binary_mask(mask)
#             masks.append(mask)
#             # 将字段添加到相应的列表中
#             segments.append(segment)
#             # label_names.append(label_name)
#             iscrowds.append(int(iscrowd))
#             labels.append(label)
    # names = [ 'background','plane',
    #          'ship', 'storage tank',
    #          'baseball diamond', 'tennis court',
    #          'basketball court', 'ground track field',
    #          'harbor', 'bridge', 'large vehicle', 'small vehicle',
    #          'helicopter', 'roundabout', 'soccer ball field',
    #          'swimming pool', 'container crane']
    # label_name = 'plane'
    # print(names.index(label_name))





#
# if __name__ == '__main__':
#     annotation_file = './mydata/annotations/labels.json'
#     coco = COCO(annotation_file)
#     # catIds = coco.getCatIds()  # 类别ID列表
#     imgIds = coco.getImgIds()  # 图像ID列表
#     # labels = []
#     # bboxs = []
#     # masks = []
#     # areas = []
#     # iscrowd = []
#     for imgId in imgIds:
#         #以下是对一张照片的信息获取
#         image = coco.loadImgs([imgId])[0]
#         h, w = image['height'], image['width']
#         annId = coco.getAnnIds(imgIds=imgId, iscrowd=False)
#         anns =  coco.loadAnns(annId)   # 获取所有注释信息(所有区域的分割)
#         labels = []
#         bboxs = []
#         masks = []
#         areas = []
#         iscrowds = []
#         for ann in anns:
#             category = ann["category_id"]
#             seg = ann['segmentation']
#             bbox = ann['bbox']
#             area = ann['area']
#             iscrowd = ann["iscrowd"]
#             mask = sge2mask(seg,h,w)
#             labels.append(category)
#             bboxs.append(bbox)
#             masks.append(mask)
#             areas.append(area)
#             iscrowds.append(iscrowd)







# if __name__ == '__main__':
#     seg = np.array(
#         [[116.39999184016978,
#           213.75421133231242,
#           105.70473916023103,
#           227.12327718223585,
#           104.36783257523868,
#           266.78483920367535,
#           127.09524452010851,
#           273.4693721286371]]
#     )
#     mask = sge2mask(seg,h=362,w=582)
#     # print(type(mask))
#     mask = Image.fromarray(mask)
#     mask = mask.convert("P")
#     # 定义调色板
#     palette = [
#         0, 0, 0,  # 黑色背景
#         255, 0, 0,  # 索引1为红色
#         255, 255, 0,  # 索引2为黄色
#         255, 153, 0,  # 索引3为橙色
#     ]
#
#     # 应用调色板
#     mask.putpalette(palette)
#     mask.s

