import os

data_path = '/home/s571b087/lensless/project/rice_face'


def split_train_Test(data_path):
    root_dir = [x for x in os.walk(data_path, 'demosaiced_measurement')]
    root_new = os.path.join(data_path, 'flatcam_split_dataset')
    test = os.path.join(root_new, 'test')
    train = os.path.join(root_new, 'train')
    test_image = []
    train_image = []
    extension = '.png'
    for i in range(1, 8):
        train_image.append(f'00{i}{extension}')
    for i in range(8, 10):
        test_image.append(f'00{i}{extension}')
    test_image.append(f'0{10}{extension}')
    for i in range(11, 100):
        if 0 < i % 10 <= 7:
            train_image.append(f'0{i}{extension}')
        else:
            test_image.append(f'0{i}{extension}')
    test_image.append(f'{100}{extension}')
    for i in range(101, 221):
        if 0 < i % 10 <= 7:
            train_image.append(f'{i}{extension}')
        else:
            test_image.append(f'{i}{extension}')
    for i in range(221, 275):
        if 0 < ((i - 220) % 6) <= 4:
            train_image.append(f'{i}{extension}')
        else:
            test_image.append(f'{i}{extension}')

    for sub_dir in root_dir[0][1]:
        test_dir = os.path.join(test, sub_dir)
        train_dir = os.path.join(train, sub_dir)
        if not os.path.isdir(test_dir):
            os.makedirs(test_dir)
        if not os.path.isdir(train_dir):
            os.makedirs(train_dir)

    for c, sub_dir in enumerate(root_dir[1:]):
        class_dir = sub_dir[0]
        for image_name in train_dir:
            class_name = os.path.split(class_dir)[1]
            source_im = os.path.join(class_dir, image_name)
            dest_im = os.path.join(train_dir, class_name, image_name)
            os.system(f'cp {source_im} {dest_im}')
        for image_name in test_dir:
            class_name = os.path.split(class_dir)[1]
            source_im = os.path.join(class_dir, image_name)
            dest_im = os.path.join(test_dir, class_name, image_name)
            os.system(f'cp {source_im} {dest_im}')


split_train_Test(data_path)
