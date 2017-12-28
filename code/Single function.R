library(mxnet)

#Load pre-trained model & image

Dense_model = mx.model.load('model/densenet-imagenet-169-0', 125)

img_path = paste0('image/1.jpg')
img <- load.image(img_path)

#Function

CAM = function (img) {
  
  require(imager)
  require(EBImage)  
  require(magrittr)
  
  #Resized the image
  
  resized_img <- resize(img, 224, 224)
  X = array(resized_img, dim = c(224, 224, 3, 1)) * 256
  
  #Visualization
  
  par(mai = rep(0, 4), mfrow = c(1, 2))
  plot(resized_img, axes = FALSE)
  img <- resized_img %>% grayscale()
  plot(img, axes = FALSE)
  
  #Extract the feature map from DenseNet
  
  all_layers = Dense_model$symbol$get.internals()
  relu1_output = which(all_layers$outputs == 'relu1_output') %>% all_layers$get.output()
  softmax_output = which(all_layers$outputs == 'softmax_output') %>% all_layers$get.output()
  
  #Note-1: 'tail(all_layers$outputs, 20)' can be used to understand the last few layers of your network.
  #Note-2: 'mx.symbol.infer.shape(relu1_output, data = c(224, 224, 3, 1))$out.shapes' can be used to understand
  #        output shape of the intrested layer.
  
  out = mx.symbol.Group(c(relu1_output, softmax_output))
  executor = mx.simple.bind(symbol = out, data = c(224, 224, 3, 1), ctx = mx.cpu())
  
  mx.exec.update.arg.arrays(executor, Dense_model$arg.params, match.name = TRUE)
  mx.exec.update.aux.arrays(executor, Dense_model$aux.params, match.name = TRUE)
  mx.exec.update.arg.arrays(executor, list(data = mx.nd.array(X)), match.name = TRUE)
  mx.exec.forward(executor, is.train = FALSE)
  
  #Get the weights in final fully connected layer
  
  FC_weight = as.array(Dense_model$arg.params$fc1_weight)
  
  #Get the feature maps from last convolution layer
  
  feature = as.array(executor$ref.outputs$relu1_output)
  
  #Get the prediction output
  
  pred_pos = which.max(as.array(executor$ref.outputs$softmax_output))
  label_names = readLines('http://data.dmlc.ml/mxnet/models/imagenet/synset.txt')
  print(label_names[pred_pos])
  
  #Weithged sum of feature map: the core of class activation mapping (CAM)
  
  for (i in 1:1664) {
    if (i == 1) {
      CAM_ = feature[,,i,] * FC_weight[i, pred_pos] 
    } else {
      CAM_ = CAM_ + feature[,,i,] * FC_weight[i, pred_pos] 
    }
  }
  
  #Standardization from 0 to 1
  
  CAM_ = (CAM_ - min(CAM_))/(max(CAM_) - min(CAM_)) 
  
  #Define the color
  
  cols <-  colorRampPalette(c("#000099", "#00FEFF", "#45FE4F", 
                              "#FCFF00", "#FF9400", "#FF3100"))(256)
  
  #Enlarge the class activation mapping (7*7 to 224*224) and fuzzification
  
  w = makeBrush(size = 7, shape = 'gaussian', sigma = 2)
  
  CAM = EBImage::resize(CAM_, 224, 224) %>% filter2(., w)
  CAM = round(CAM*255)+1 %>% as.integer()
  FINAL_CAM = cols[CAM] %>% paste0(., "80") %>% matrix(., 224, 224, byrow = TRUE) %>% .[224:1,] %>% as.raster()
  
  #Visualization
  
  plot(FINAL_CAM, add = TRUE)
  
  #Show the prediction output
  
  obj = label_names[pred_pos]
  legend('bottomright', paste0(substr(obj, 11, nchar(obj)), ' (prob = ', round(as.array(executor$ref.outputs$softmax_output)[pred_pos], 3), ')'), bg = 'gray90')
  
}

#Use this function!

CAM(img)
