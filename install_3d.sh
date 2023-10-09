conda create -n pytorch3d python=3.9
conda activate pytorch3d
conda install pytorch=1.13.0 torchvision pytorch-cuda=11.6 -c pytorch -c nvidia
conda install -c fvcore -c iopath -c conda-forge fvcore iopath
conda install pytorch3d -c pytorch3d
pip3 install cython
pip3 install -r requirements.txt
pip3 install --upgrade numpy==1.23.0

mkdir /usr/lib/dri/
ln -s /usr/lib/x86_64-linux-gnu/dri/swrast_dri.so /usr/lib/dri/swrast_dri.so
rm /miniconda3/envs/pytorch3d/bin/../lib/libstdc++.so.6
ln /usr/lib/x86_64-linux-gnu/libstdc++.so.6 /miniconda3/envs/pytorch3d/bin/../lib/libstdc++.so.6

apt update
apt-get install -y libosmesa6-dev freeglut3-dev 