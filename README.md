# Surface-Defect-Detection
A semantic segmentation network for surface defect detection based on BSData dataset

# 基于BSData数据集训练的语义分割网络
## A semantic segmentation network for surface defect detection based on BSData dataset
## BSData数据集介绍
该数据集是一个关于金属丝杆表面缺陷的数据集，该数据集包含1104个3通道图像，以及394张带点蚀缺陷注释的图像，标注为json文件格式。官方数据模板如下：
<br>
<div align=center>
<img src="https://github.com/looog-e/surface-defect-detection/blob/main/ImageFolder/demo.png" width = "700" height = "400" alt="" align=center />
</div>
<br>

具体介绍见此[链接](https://github.com/2Obe/BSData)，该数据集相对应的论文中也介绍了作者自己所使用的分割模型，作者使用的是一个Mask R-CNN，并在该数据集上实现了31.6%的mIOU。其分割结果如下：
<br>
<div align=center>
<img src="https://github.com/looog-e/surface-defect-detection/blob/main/ImageFolder/3.PNG" width = "700" height = "400" alt="" align=center />
</div>
<br>
本课题研究的是一个基于UNet架构的分割网络，暂且称该网络为ZNet，其结构图如下：
<br>
<div align=center>
<img src="https://github.com/looog-e/surface-defect-detection/blob/main/ImageFolder/%E5%9B%BE%E7%89%871.png" width = "1000" height = "600" alt="" align=center />
</div>
