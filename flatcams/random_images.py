import glob
import os
import random
import shutil

def get_1_rand_img (data_path):
    src_path = os.path.join(data_path, 'fc_captures')
    dst_path = os.path.join(data_path, 'rice_rand_87')
    if not os.path.exists(dst_path):
        os.makedirs(dst_path)
    root_dir = [x for x in os.walk(src_path)]
    for c, sub_dir in enumerate(root_dir[1:]):
        class_dir = sub_dir[0]
        class_name = os.path.split(class_dir)[1]
        random_img = random.sample(glob.glob(os.path.join(class_dir, '*.png')), 1)
        dst_img_name = f"{class_name}_{random_img[0].split('/')[-1]}"
        dst_img_path = os.path.join(dst_path, dst_img_name)
        os.system(f'cp {random_img[0]} {dst_img_path}')
    print('created random image from each dirs')


if __name__ == '__main__':
    data_path = '/scratch/s571b087/project/Lensless_Imaging/rice_face'
    get_1_rand_img(data_path)
