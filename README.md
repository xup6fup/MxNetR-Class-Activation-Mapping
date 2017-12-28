# MxNetR for the Class Activation Mapping (CAM)

## Background

This project propose a simple example to expose the implicit attention of Convolutional Neural Networks on the image. The paper is published at [CVPR'16](https://arxiv.org/pdf/1512.04150.pdf). The basic idea is based on global average pooling (GAP) layer in the last of network. The framework of the Class Activation Mapping (CAM) is as below:

<img src="image/F1.jpg" width="835" height="392" alt="F1"/>

As illustrated in above figure, global average pooling (GAP) outputs the spatial average of the feature map of each unit at the last convolutional layer. A weighted sum of these values is used to generate the final output. Similarly, we compute a weighted sum of the feature maps of the last convolutional layer to obtain our class activation maps. We describe this more formally below for the case of softmax. The same technique can be applied to regression and other losses.

Fortunately, some of popular networks such as DenseNet, SqueezeNet, ResNet already use global average pooling (GAP) at the end, so we can directly use pre-trained model to generate the class activation mapping (CAM) without any modification. Here is a sample script to generate class activation mapping (CAM) by a pre-trained DenseNet-169. 

Note: If you cannot find the 'densenet-imagenet-169-0-0125.params' file in 'model' directory, you can download it from [here](https://drive.google.com/open?id=1rcLiIeyXiSYU10Ce-1UpqO3sNalaIZ5M). This pre-trained parameter is contributed by [bruinxiong/densenet.mxnet](https://github.com/bruinxiong/densenet.mxnet). After download, you can put this file in the 'model' directory.

## Requirements

- [R](https://www.r-project.org/) & [Rstudio (not necessary)](https://www.rstudio.com/)

- [MxNetR](https://mxnet.incubator.apache.org/)

You can install the package directly in the R console.

```r
cran <- getOption("repos")
cran["dmlc"] <- "https://s3-us-west-2.amazonaws.com/apache-mxnet/R/CRAN/"
options(repos = cran)
install.packages("mxnet")
```

- [EBImage](https://bioconductor.org/packages/release/bioc/html/EBImage.html) 

You can install the package directly in the R console.

```r
## try http:// if https:// URLs are not supported
source("https://bioconductor.org/biocLite.R")
biocLite("EBImage")
```

- [magrittr](https://cran.r-project.org/web/packages/magrittr/index.html) & [imager](https://cran.r-project.org/web/packages/imager/index.html)

You can install the package directly in the R console.

```r
install.packages(c('magrittr', 'imager'))
```

## Start to try

The example code can be found from ['code/Class-Activation-Mapping.R'](https://github.com/xup6fup/Class-Activation-Mapping-CAM-in-MxNet/blob/master/code/Class-Activation-Mapping.R). The output is shown as following, and the integration function was been coded in ['code/Single fucntion.R'](https://github.com/xup6fup/Class-Activation-Mapping-CAM-in-MxNet/blob/master/code/Single fucntion.R).

<img src="image/F2.jpg" width="835" height="626" alt="F2"/>

## Other pre-trained MxNet Network

[MXNet Model Zoo](https://mxnet.incubator.apache.org/model_zoo/) provides number of pre-trained model by imagenet. You can download them to modify this script. Among **Inception v3 w/BatchNorm**, **ResidualNet152**, **ResNext101-64x4d** can be directly used to generate the class activation mapping (CAM).
