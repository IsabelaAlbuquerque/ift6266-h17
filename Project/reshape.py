import os, sys
import glob
import cPickle as pkl
import numpy as np
import PIL.Image as Image
from skimage.transform import resize

def reshape_train():
    train_data_path = "inpainting/train2014"
    save_dir = "inpainting/useful/train/complete/"

    preserve_ratio = True
    image_size = (64, 64)
    imgs = glob.glob(train_data_path+"/*.jpg")
    for i, img_path in enumerate(imgs):
        img = Image.open(img_path)
        print i, len(imgs), img_path

        if img.size[0] != image_size[0] or img.size[1] != image_size[1] :
            if not preserve_ratio:
                img = img.resize((image_size), Image.ANTIALIAS)
            else:
                ### Resize based on the smallest dimension
                scale = image_size[0] / float(np.min(img.size))
                new_size = (int(np.floor(scale * img.size[0]))+1, int(np.floor(scale * img.size[1])+1))
                img = img.resize((new_size), Image.ANTIALIAS)

                ### Crop the 64/64 center
                tocrop = np.array(img)
                center = (int(np.floor(tocrop.shape[0] / 2.)), int(np.floor(tocrop.shape[1] / 2.)))
                print tocrop.shape, center, (center[0]-32,center[0]+32), (center[1]-32,center[1]+32)
                if len(tocrop.shape) == 3:
                    tocrop = tocrop[center[0]-32:center[0]+32, center[1] - 32:center[1]+32, :]
                else:
                    tocrop = tocrop[center[0]-32:center[0]+32, center[1] - 32:center[1]+32]
                img = Image.fromarray(tocrop)

        img.save(save_dir + os.path.basename(img_path))


def reshape_valid():
    valid_data_path = "inpainting/val2014"
    save_dir = "inpainting/useful/valid/complete/"

    preserve_ratio = True
    image_size = (64, 64)

    imgs = glob.glob(valid_data_path+"/*.jpg")


    for i, img_path in enumerate(imgs):
        img = Image.open(img_path)
        print i, len(imgs), img_path

        if img.size[0] != image_size[0] or img.size[1] != image_size[1] :
            if not preserve_ratio:
                img = img.resize((image_size), Image.ANTIALIAS)
            else:
                ### Resize based on the smallest dimension
                scale = image_size[0] / float(np.min(img.size))
                new_size = (int(np.floor(scale * img.size[0]))+1, int(np.floor(scale * img.size[1])+1))
                img = img.resize((new_size), Image.ANTIALIAS)

                ### Crop the 64/64 center
                tocrop = np.array(img)
                center = (int(np.floor(tocrop.shape[0] / 2.)), int(np.floor(tocrop.shape[1] / 2.)))
                print tocrop.shape, center, (center[0]-32,center[0]+32), (center[1]-32,center[1]+32)
                if len(tocrop.shape) == 3:
                    tocrop = tocrop[center[0]-32:center[0]+32, center[1] - 32:center[1]+32, :]
                else:
                    tocrop = tocrop[center[0]-32:center[0]+32, center[1] - 32:center[1]+32]
                img = Image.fromarray(tocrop)

        img.save(save_dir + os.path.basename(img_path))        

if __name__ == '__main__':
    #reshape_train()
    reshape_valid()        