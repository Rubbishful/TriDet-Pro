"""下载 THUMOS14 I3D 特征 + 标注（~12GB）"""
import gdown
import tarfile
import os

url = 'https://drive.google.com/file/d/1zt2eoldshf99vJMDuu8jqxda55dCyhZP/view'
tar_path = os.path.join(os.path.dirname(__file__), 'thumos.tar.gz')

print("正在下载 THUMOS14 数据集 (~12GB)，请耐心等待...")
gdown.download(url, tar_path, quiet=False, fuzzy=True)

print("正在解压...")
with tarfile.open(tar_path, 'r:gz') as tar:
    tar.extractall(os.path.dirname(__file__))

print("解压完成！现在可以删除 thumos.tar.gz 节省空间")
print("数据位置: ./data/thumos/")
