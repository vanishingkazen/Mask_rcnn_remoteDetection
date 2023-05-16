import os
from PIL import Image

def delete_large_files(folder_path, threshold):
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(root, file)
            file_size = os.path.getsize(file_path)
            if file_size > threshold:
                os.remove(file_path)
                print(f"Deleted file: {file_path}")


def mydelete():
    root = 'B:\\DOTA\\data\\train'
    img_root = os.path.join(root, "images")
    label_txts_root = os.path.join(root,"labelTxt")
    imgs = list(sorted(os.listdir(img_root)))
    label_txts = list(sorted(os.listdir(label_txts_root)))
    for idx,img_file_name in enumerate(imgs):
        img_path = os.path.join(img_root, img_file_name)
        img = Image.open(img_path).convert("RGB")
        h, w = img.height, img.width
        if h>1200 or w > 1200:
            label_path = os.path.join(label_txts_root,label_txts[idx])
            os.remove(img_path)
            os.remove(label_path)
            print(img_path,label_path)


if __name__ == '__main__':
    mydelete()