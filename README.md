# Fusing Sentinel-1 and Sentinel-2 images for deforestation detection in the Brazilian Amazon under diverse cloud conditions

Most current early warning systems for deforestation rely on cloud-free optical images, which are difficult to obtain in tropical regions. The fusion of optical and SAR images is an attractive alternative in these cases. Although less discriminative in cloudless regions, SAR data are nearly unaltered by clouds, allowing better discrimination in cloudy areas than the optical counterpart. This letter proposes solutions that seek the best combination between the two modalities for each pixel as a function of the surrounding cloud cover to maximize classification accuracy. We compared early, joint, and late fusion variants of Fully Convolutional Networks (FCN)  to detect deforestation in the Amazon rainforest from Sentinel 1 and Sentinel 2 data. Experiments conducted to compare the architecture variants showed that optical-SAR fusion might outperform the single-modal variants for deforestation detection on pixels affected by any cloud cover level. In particular, the joint fusion approach outperformed the single modal counterparts under all cloud cover scenarios.

# Code description

01a_Evaluate_Images.ipynb - Evaluation of available images in Google Earth Engine.
