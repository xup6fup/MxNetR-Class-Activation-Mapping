# MxNetR for the Class Activation Mapping (CAM)

This project propose a simple example to expose the implicit attention of Convolutional Neural Networks on the image. The paper is published at [CVPR'16](https://arxiv.org/pdf/1512.04150.pdf). The basic idea is based on global average pooling (GAP) layer in the last of network. The framework of the Class Activation Mapping (CAM) is as below:

<img src="image/F1.jpg" width="835" height="392" alt="F1"/>

As illustrated in above figure, global average pooling (GAP) outputs the spatial average of the feature map of each unit at the last convolutional layer. A weighted sum of these values is used to generate the final output. Similarly, we compute a weighted sum of the feature maps of the last convolutional layer to obtain our class activation maps. We describe this more formally below for the case of softmax. The same technique can be applied to regression and other losses.

Fortunately, some of popular networks such as DenseNet, SqueezeNet, ResNet already use global average pooling (GAP) at the end, so we can directly use pre-trained model to generate the class activation mapping (CAM) without any modification. Here is a sample script to generate class activation mapping (CAM) by a pre-trained DenseNet-169. 

Note: The pre-trained DenseNet-169 parameters by MxNet can be download from [here](https://drive.google.com/open?id=1rcLiIeyXiSYU10Ce-1UpqO3sNalaIZ5M), and this is contributed by [bruinxiong/densenet.mxnet](https://github.com/bruinxiong/densenet.mxnet). After download, you can put this file in the 'model' directory.


The example output is shown as following:

<img src="image/F2.jpg" width="835" height="626" alt="F2"/>


