# Water-depth-estimation
This repository includes code for the paper: VGG16、ResNet、GoogleNet、EfficientNet、ViT、SE-ResNet、CBAM-ResNet、CA-ResNet.
All of these models in the repository can assess the level of flood images.
## CA-ResNet
This study tried to add Coordinate Attention (CA) into ResNet, and proposed CA-ResNet to acquire the global submerged water depth.The proposed CA-ResNet method was compared with VGGNet, ResNet, GoogleNe, and EfficientNet, the CA-ResNet model exhibits superior performance.In order to evaluate the effect of adding different attention mechanisms to ResNet. This study investigate the effect of attention modules on the pre-trained model ResNet by incorporating two different attention mechanisms, SENet, and CBAM, into the ResNet network and performing them in the same experimental context and setting. 
![image](https://github.com/powerchen16/water-depth-estimation/assets/155006365/4db1a347-ea66-4aa6-a42b-975b31e00553)
## Dataset
We crawled images of floods from social media for the Henan rainstorm 7.20 event as inputs to the model.Here are two example images.
![4661756159070330-1](https://github.com/powerchen16/water-depth-estimation/assets/155006365/478fd696-ae01-4f35-b2f5-23cb43a4282a)
![4662216433337658-1](https://github.com/powerchen16/water-depth-estimation/assets/155006365/3065b191-90c4-4ca5-aeda-8216bfb70162)
## Setup
All code was developed and tested on Nvidia RTX3060 in the following environment:
python3.6、pytorch、matplotlib、numpy、cuda>=8.0、cudnn>=5.0
## Get Started
The folder contains the code for the training or testing of each model and the Grad-CAM++ visualisation. Please contact the authors if you need the datasets.
