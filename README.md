This code takes a model's weights for deeplabV3+ trained on city scape dataset and fine tunes it on AMP's EVT footage.

Necessary linkage of the path's weights and the folder locations for respective input and output files are needed, but set up is minimal within the main method.

current model weights: https://drive.google.com/file/d/1HXUEMMD85b746F2CVlpJ4ICecv6q_Ebz/view?usp=sharing


Training Data: 
https://drive.google.com/file/d/1Hz-Hlyr5NFwk2YSwtRh5L75vWTK1r3k3/view?usp=sharing

or 

https://www.dropbox.com/scl/fo/i4k86qqiuzy33aawug287/AJZbFCPIvT521xW8lekSDr8/best_deeplabv3plus_mobilenet_cityscapes_os16.pth?rlkey=fgm7rgpeankeh9394j492sxif&st=oc0bpoyx&dl=0

git clone of https://github.com/VainF/DeepLabV3Plus-Pytorch.git is recommended for model initialization. 

Segment_metafolder parses the folder of EVT raw image frames, runs inference with the model and stores segmented images and label .pt files into a folder denoted as fuu{i}.

The training method matches the original folders with the respective temporary folder fuu{i} and performs batch training off of passed in parameters of pytorch loss function and optimizers

#For Inference of frame:
  Simply pass in model and frame path into inference(), it will return an image similiar to this: ![image](https://github.com/user-attachments/assets/30c19974-16a1-4267-9e3e-4883c6860a2c)

  Along with pixel coords of centerline.


