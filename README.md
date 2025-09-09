This code takes a model's weights for deeplabV3+ trained on city scape dataset and fine tunes it on AMP's EVT footage.

Necessary linkage of the path's weights and the folder locations for respective input and output files are needed, but set up is minimal within the main method.

Training Data: 
https://drive.google.com/file/d/1Hz-Hlyr5NFwk2YSwtRh5L75vWTK1r3k3/view?usp=sharing

current model weights: https://drive.google.com/file/d/1WJeUss2ko5mjIykvDbLMbn8uyhRXuON-/view?usp=sharing

or Quantized ONNX: 
https://drive.google.com/file/d/1OnO1lCKWBXHQDGAcTA4uYjCLubIDR2kK/view?usp=sharing

git clone of https://github.com/VainF/DeepLabV3Plus-Pytorch.git is recommended for model initialization along with a requirements.txt install on a clean venv. 

Segment_metafolder parses the folder of EVT raw image frames, runs inference with the model and stores segmented images and label .pt files into a folder denoted as fuu{i}.

The training method matches the original folders with the respective temporary folder fuu{i} and performs batch training off of passed in parameters of pytorch loss function and optimizers

#For Inference of frame:
  Simply pass in model and frame path into inference(), it will return an image similiar to this: ![image](https://github.com/user-attachments/assets/30c19974-16a1-4267-9e3e-4883c6860a2c)

  Along with pixel coords of centerline.

## CV info document:
https://docs.google.com/document/d/1oydoEg7ShT8s-GqMCGngmE2rg-mTC_cPRtpS6nkWkDc/edit?usp=sharing

## Non Linear MPC info document:
https://kennesawedu-my.sharepoint.com/:w:/r/personal/vwulfek1_students_kennesaw_edu/Documents/Non-Linear%20Model%20Predictive%20Control%20for%20Autonomous%20Go-Karts.docx?d=w4c3b30fee2d44488954b32006cc48bd6&csf=1&web=1&e=BNkHsO

