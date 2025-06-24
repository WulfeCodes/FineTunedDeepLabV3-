This code takes a model's weights for deeplabV3+ trained on city scape dataset and fine tunes it on KSU EVT footage.

Necessary linkage of the path's weights and the folder locations for respective input and output files are needed, but set up is minimal within the main method.

current model weights: https://drive.google.com/file/d/1HXUEMMD85b746F2CVlpJ4ICecv6q_Ebz/view?usp=sharing
Training Data: 
https://drive.google.com/file/d/1Hz-Hlyr5NFwk2YSwtRh5L75vWTK1r3k3/view?usp=sharing

git clone of https://github.com/VainF/DeepLabV3Plus-Pytorch.git is recommended for model initialization. 

Segment_metafolder parses the folder of EVT raw image frames, runs inference with the model and stores segmented images and label .pt files into a folder denoted as fuu{i}.

The training method matches the original folders with the respective temporary folder fuu{i} and performs batch training off of passed in parameters of pytorch loss function and optimizers



