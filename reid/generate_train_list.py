import os

print('running...')
images = os.listdir('/workspace/dataset/zhangjiagang_test_data')
images.sort()

with open('test_list.txt', 'w') as f:
    for img in images:
        pid = int(img.split('_')[0])
        f.write(img + ' ' + str(pid) + '\n')
print('Done')

