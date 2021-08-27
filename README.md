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

## ZNet网络介绍
本课题研究的是一个基于UNet架构的分割网络，暂且称该网络为ZNet，其结构图如下：
<br>
<div align=center>
<img src="https://github.com/looog-e/surface-defect-detection/blob/main/ImageFolder/%E5%9B%BE%E7%89%871.png" width = "1000" height = "600" alt="" align=center />
</div>

该网络相较于UNet的主要差别在于三点，其一是在网络的开头使用Scope_Conv结构，该结构的主要目的是将感受野大小不同的特征图concat在一起，以实现不同尺度的特征的检测精度提升，其次是使用了attention机制来实现不同channel之间的权重关系调节，最后一点是该网络使用了Res_Conv层，该模块主要是承担网络中编码部分的卷积工作，进行特征提取。该网络是通过Adam优化器来进行优化的，目前最优结果mIOU可达46.45%。

## 识别结果展示及说明
通过语义分割网络成功实现检测的结果如下：
<br>
<div align=center>
<img src="https://github.com/looog-e/surface-defect-detection/blob/main/ImageFolder/4.png" width = "300" height = "150" alt="" align=center /><img src="https://github.com/looog-e/surface-defect-detection/blob/main/ImageFolder/5.png" width = "300" height = "150" alt="" align=center /><img src="https://github.com/looog-e/surface-defect-detection/blob/main/ImageFolder/6.png" width = "300" height = "150" alt="" align=center />
</div>
<br>
<div align=center>
<img src="https://github.com/looog-e/surface-defect-detection/blob/main/ImageFolder/7.png" width = "300" height = "150" alt="" align=center /><img src="https://github.com/looog-e/surface-defect-detection/blob/main/ImageFolder/8.png" width = "300" height = "150" alt="" align=center /><img src="https://github.com/looog-e/surface-defect-detection/blob/main/ImageFolder/9.png" width = "300" height = "150" alt="" align=center />
</div>

由此可见实现的分割效果还是较为准确，但是该网络和官方提供的Mask R-CNN存在一样的问题，即对小目标很难准确检测到，而对一些细长的缺陷检测效果较差，且对于复杂情况很难分辨缺陷和非缺陷部分，这些问题都还有待提升。（该网络的主要优势是网络结构较小且检测精度还令人满意）
