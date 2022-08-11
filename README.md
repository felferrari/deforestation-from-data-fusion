# Fusing Sentinel-1 and Sentinel-2 images for deforestation detection in the Brazilian Amazon under diverse cloud conditions

Increasing deforestation rates in the Brazilian Amazon highlight the importance of early warning systems for deforestation monitoring. 
These systems usually employ cloud-free optical images, which are difficult to obtain in tropical environments. In this letter, we proposed strategies based on
convolutional neural networks (CNNs) to fuse Sentinel-2 (Optical) and Sentinel-1 (SAR) to detect deforestation under diverse 
cloud conditions. We showed that optical-SAR fusion improved deforestation detection in low, medium and high cloud conditions.
The joint fusion strategy that combines the feature maps from encoders trained with optical and SAR images separately provided
the best results. Fusing optical and SAR images with CNNs is an alternative to detect deforestation, especially for regions with a
high probability of the presence of clouds throughout the year.

## Models Schema

All implemented models were based on modified ResUnet:
![base model](https://user-images.githubusercontent.com/9152265/184145633-00e0d05f-484a-46cc-9ec9-cd3f8fe60070.png)

The Single Modal Models, which employs only one modal (Optical or SAR):
![single modal models](https://user-images.githubusercontent.com/9152265/184146306-4d28944a-1ff5-4c5e-bdb8-36887888da7c.png)

