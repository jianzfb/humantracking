import os
from shutil import copyfile

# You only need to change this line to your dataset download path
download_path = '/workspace/dataset/four_public'

if not os.path.isdir(download_path):
    print('please change the download_path')

save_path = '/workspace/dataset/reid-four-public/pytorch/train'
if not os.path.isdir(save_path):
    os.makedirs(save_path)

#---------------------------------------
#train_val
train_path = download_path
train_all_save_path = save_path

for name in open('/workspace/dataset/four_public/train_list.txt'):
    name = name.strip()
    image_name = name.split(' ')[0]
    ID = name.split(' ')[1]

    src_path = os.path.join(train_path, image_name)
    dst_all_path = train_all_save_path + '/' + ID
    if not os.path.isdir(dst_all_path):
        os.mkdir(dst_all_path)
    copyfile(src_path, dst_all_path + '/' + os.path.basename(image_name))
