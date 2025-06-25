import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from network import modeling
import cv2 as cv
import os
import subprocess
import glob
import torchvision.transforms.functional as F

#dependencies: git clone https://github.com/VainF/DeepLabV3Plus-Pytorch.git

    #cityscape eval input size: 512x512, 513x513, 768x768,1024x2048, and 16:9{ds shapes:1280x720}
    #standard shapes are Batch,RGB,H,W
def create_smart_preprocessor(target_width=768):
    """
    Keep 16:9 aspect ratio, scale by width
    1280x720 → 768x432 (width x height in image terms)
    Results in tensor shape: [3, 432, 768] (C, H, W)
    """
    
    def smart_resize(img):
        original_width, original_height = img.size  # PIL gives (width, height)
        aspect_ratio = original_height / original_width  # h/w ratio
        
        new_width = target_width  # 768
        new_height = int(target_width * aspect_ratio)  # 768 * (720/1280) = 432
        
        # print(f"Resizing {original_width}x{original_height} → {new_width}x{new_height}")
        # print(f"Will create tensor shape: [3, {new_height}, {new_width}]")
        return img.resize((new_width, new_height), Image.BILINEAR)
    
    transforms_list = [
        transforms.Lambda(smart_resize),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ]
    
    return transforms.Compose(transforms_list)


def preprocess_image(img_path):
    """Complete preprocessing pipeline"""
    img = Image.open(img_path)
    # print(f"Original image: {img.size}")
    
    preprocessor = create_smart_preprocessor(target_width=768)
    processed_tensor = preprocessor(img)
    
    # print(f"Processed tensor: {processed_tensor.shape}")
    
    # Add batch dimension for model input
    batch_tensor = processed_tensor.unsqueeze(0)
    # print(f"Model input shape: {batch_tensor.shape}")
    
    return batch_tensor, img  # Return both for visualization later

def training_inference(model,batch_tensor,threshold=.75):
    try:
        with torch.no_grad():
            op_tensor=model(batch_tensor)
            # print("successful training inference:",op_tensor.shape)
            #training inference needs a batch dimension
            op_tensor=op_tensor.squeeze(0)
            probs = torch.softmax(op_tensor,dim=0)
            confidence, _labels= torch.max(probs, dim=0)
            mask = confidence > threshold
            pseudo_labels = torch.argmax(probs,axis=0)
            final_labels = torch.where(mask,pseudo_labels,-1)
            #shapes Class_num,H,W to Class_num
            #each pixel gets an arg max from the batch dim of classes
            #-1 for low confidence regions
            return final_labels
        
    except Exception as e:
        print(f"error at inference with {e}")

def train(model,criterion,optimizer,img_src_meta_folder,trainFolder,device='cpu'):

    total_loss = 0

    for i, (img_folder,folder) in enumerate(zip(os.listdir(img_src_meta_folder),os.listdir(trainFolder))):


            if i == 0:
                continue

            if folder == None:
                break
            
            curr_img_folder_path = os.path.join(img_src_meta_folder,img_folder)
            images = [img for img in glob.glob(os.path.join(curr_img_folder_path, "*.jpg"))]
            images.sort()

            folder_path = os.path.join(trainFolder,folder)
            tensors = [tns for tns in glob.glob(os.path.join(folder_path, "*.pt"))]
            tensors.sort()


            if len(tensors)!=len(images):
                print("warning may be a folder mismatch")
                print(tensors,images)
                print(len(tensors),len(images))
                continue


            for z,(image_,label_file) in enumerate(zip(images,tensors)):

                    if not tensors:
                        print("No images found")

                    #only train on 5th images
                    if z > len(images)-5 or z%5!=0:
                        continue
                    
                    labels = torch.load(label_file)
                    nxt_labels = torch.load(tensors[z+1])

                    input_tensor, original_img = preprocess_image(image_)

                    nxt_input_tensor, nxt_img = preprocess_image(images[z+1])

                    print(f"pre batch+1 input tensor shape: {input_tensor.shape}, label shape: {labels.shape}")

                    input_tensor = input_tensor.to(device)
                    nxt_input_tensor = nxt_input_tensor.to(device)
                    labels = labels.unsqueeze(0).to(device).long()
                    nxt_labels = nxt_labels.unsqueeze(0).to(device).long()

                    batch_inputs = torch.cat([input_tensor,nxt_input_tensor],dim=0)
                    batch_labels = torch.cat([labels,nxt_labels],dim=0)

                    if (batch_inputs.shape != batch_labels.shape):
                        print("WARNING, LOADING ERROR MIGHT OCCUR")
                        print("shape shi",batch_inputs.shape, batch_labels.shape)

                    optimizer.zero_grad()
                    op=model(batch_inputs)
                    loss = criterion(op,batch_labels)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()

            print(f"Epoch [{i+1}], Loss: {total_loss:.4f}")

def inference(model,batch_tensor):

    img_path = "C:/Project/DL_HW/dataset/images_backup_aks_race/2025-06-06-17-06-33/2025-06-06-17-06-40.028664.jpg"

    batch_tensor, orig_image = preprocess_image(img_path)

    #road = ID 7

    try:
        with torch.no_grad():
            op_tensor=model(batch_tensor)
            print("successful inference")
            op_tensor=op_tensor.squeeze(0)
            labels = torch.argmax(op_tensor,dim=0)

            #each pixel gets an arg max from the batch dim of classes
            binary_mask = (labels == 7).float()

            binary_segmented_image = color_binary_mask(orig_image,labels)
            
            #TODO experimentation of kernel sizes 
            kernel = np.ones((5,5), np.uint8)  # 5x5 square kernel

            closed = cv.morphologyEx(binary_mask, cv.MORPH_CLOSE, kernel)

            #skeletonize or custom centerline approach

            return binary_segmented_image,labels
        
    except Exception as e:
        print(f"error at inference with {e}")

def color_binary_mask(image, binary_mask, color=(0, 255, 0), alpha=0.5):
    """Overlay binary mask on image with specified color"""
    colored_image = image.copy()
    colored_image[binary_mask == 1] = colored_image[binary_mask == 1] * (1 - alpha) + np.array(color) * alpha
    return colored_image.astype(np.uint8)

def get_cityscapes_colors():
    """Standard Cityscapes color palette for 19 classes"""
    return [
        [128, 64, 128],   # road
        [244, 35, 232],   # sidewalk  
        [70, 70, 70],     # building
        [102, 102, 156],  # wall
        [190, 153, 153],  # fence
        [153, 153, 153],  # pole
        [250, 170, 30],   # traffic light
        [220, 220, 0],    # traffic sign
        [107, 142, 35],   # vegetation
        [152, 251, 152],  # terrain
        [70, 130, 180],   # sky
        [220, 20, 60],    # person
        [255, 0, 0],      # rider
        [0, 0, 142],      # car
        [0, 0, 70],       # truck
        [0, 60, 100],     # bus
        [0, 80, 100],     # train
        [0, 0, 230],      # motorcycle
        [119, 11, 32],    # bicycle
    ]

def colorize_segmentation(predictions):
    """Convert class indices to RGB colors"""
    colors = get_cityscapes_colors()
    h, w = predictions.shape
    colored = np.zeros((h, w, 3), dtype=np.uint8)
    
    for class_id in range(len(colors)):
        mask = predictions == class_id
        colored[mask] = colors[class_id]
    
    return colored

def create_overlay_visualization(original_img, pred_map, alpha=0.6, save_path=None):
    """
    Create overlay of segmentation on original image
    alpha: transparency of segmentation (0=invisible, 1=opaque)
    """
    
    # Convert predictions to colors
    colored_seg = colorize_segmentation(pred_map)
    
    # Make sure original image is numpy array
    if isinstance(original_img, Image.Image):
        original_np = np.array(original_img)
    else:
        original_np = original_img
    
    # Resize original to match prediction size if needed
    if original_np.shape[:2] != pred_map.shape:
        original_pil = Image.fromarray(original_np)
        original_pil = original_pil.resize((pred_map.shape[1], pred_map.shape[0]))
        original_np = np.array(original_pil)
    
    # Create overlay: blend original and segmentation
    overlay = (1 - alpha) * original_np + alpha * colored_seg
    overlay = overlay.astype(np.uint8)
    
    # Visualization
    # fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    
    # # Original
    # axes[0].imshow(original_np)
    # axes[0].set_title('Original Image')
    # axes[0].axis('off')
    
    # # Pure segmentation
    # axes[1].imshow(colored_seg) 
    # axes[1].set_title('Segmentation')
    # axes[1].axis('off')
    
    # # Overlay
    # axes[2].imshow(overlay)
    # axes[2].set_title(f'Overlay (α={alpha})')
    # axes[2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    # plt.show()
    
    return colored_seg

def segment_metafolder(model,image_folder, output_video_path, temp_frame_path, fps=30):
    
    for i,folder in enumerate(os.listdir(image_folder)):

        folder_path = os.path.join(image_folder,folder)
        op_video_path = os.path.join(output_video_path,f"{i}.mp4")
        temp_frames_path_made=os.path.join(temp_frame_path,f"fuu{i}")
        os.makedirs(temp_frames_path_made,exist_ok=True)
        print(f"parsing: {folder_path}, outputting vid to {output_video_path}, temp to: {temp_frames_path_made}")

        images = [img for img in glob.glob(os.path.join(folder_path, "*.jpg"))]
        images.sort()

        if not images:
            print("No images found")

        first_image_path = images[0]
        frame = cv.imread(first_image_path)
        height, width, layers = frame.shape
        size = (width, height)

        ffmpeg_input_pattern = os.path.join(temp_frames_path_made, "frame_%06d.jpg")  
        ffmpeg_command = [
            'ffmpeg',
            '-y',
            '-r', str(fps),
            '-i', ffmpeg_input_pattern,
            '-c:v', 'libx264',
            '-pix_fmt', 'yuv420p',
            '-crf', '23',
            '-preset', 'medium',
            '-vf', f'scale={width}:{height}', # Added this for explicit resolution
            op_video_path
        ]

        print(f"\nExecuting FFmpeg command to create video: {' '.join(ffmpeg_command)}")
        print("Capturing FFmpeg output for debugging...")
        for i,image_path in enumerate(images):

            temper_frame_path = os.path.join(temp_frames_path_made, f"frame_{i:06d}.jpg")
            input_tensor, original_img = preprocess_image(image_path)
            labels=training_inference(model,input_tensor) 
            #TODO training inference function that outputs min_prob segmented tensor, store this along with images
            if labels is None:
                print("error parsing image")
                continue
            segmented_image = create_overlay_visualization(original_img,labels)
            cv.imwrite(temper_frame_path, segmented_image)
            label_path=os.path.join(temp_frames_path_made,f"labels{i:06d}.pt")
            torch.save(labels,label_path)
        print(f"video created at {output_video_path}")

        try:
            # This is the line that executes ffmpeg.exe
            ffmpeg_process = subprocess.run(ffmpeg_command + ['-loglevel', 'verbose'], check=False, capture_output=True, text=True)

            if ffmpeg_process.returncode != 0:
                print(f"\nFATAL ERROR: FFmpeg command failed with return code {ffmpeg_process.returncode}.")
                print(f"Stderr: \n{ffmpeg_process.stderr}") # <--- This will print FFmpeg's error messages
                print(f"Stdout: \n{ffmpeg_process.stdout}") # <--- This will print FFmpeg's general output
                print("--- End FFmpeg Error Output ---")
            else:
                print(f"\nSUCCESS: FFmpeg command completed with return code {ffmpeg_process.returncode}.")
                print(f"FFmpeg Stdout (might contain warnings):\n{ffmpeg_process.stdout}")
                if ffmpeg_process.stderr: # Sometimes warnings appear in stderr even on success
                    print(f"FFmpeg Stderr (might contain warnings):\n{ffmpeg_process.stderr}")
                print("--- End FFmpeg Output ---")

        except Exception as e:
            print(f"\nAn unexpected error occurred during FFmpeg execution: {e}")

        print("successfully performed img inference")

def main():


    img_path = "C:/Project/DL_HW/dataset/images_backup_aks_race"
    meta_folder_path = "C:/Project/DL_HW/dataset/vids"
    MODEL_NAME='deeplabv3plus_resnet101'
    NUM_CLASSES = 19
    OUTPUT_STRIDE = 8
    PATH_TO_PTH = "C:/Project/SemanticTester/DeepLabV3Plus-Pytorch/best_deeplabv3plus_resnet101_cityscapes_os16.pth"


    model = modeling.__dict__[MODEL_NAME](num_classes=NUM_CLASSES, output_stride=OUTPUT_STRIDE)
    model.load_state_dict(torch.load( PATH_TO_PTH, map_location='cpu',weights_only=False)['model_state'])
    print("entering meta mode")

    # frame = cv.imread("C:/Project/DL_HW/dataset/images_backup_aks_race/2025-06-06-16-14-24/2025-06-06-16-14-30.782489.jpg")
    # height, width, layers = frame.shape
    # size = (width, height)


    # op_video_path = os.path.join(f"{meta_folder_path}","2.mp4")
    # ffmpeg_input_pattern = os.path.join("C:/Project/DL_HW/dataset/fuu2", "frame_%06d.jpg")  
    # ffmpeg_command = [
    #     'ffmpeg',
    #     '-y',
    #     '-r', str(30),
    #     '-i', ffmpeg_input_pattern,
    #     '-c:v', 'libx264',
    #     '-pix_fmt', 'yuv420p',
    #     '-crf', '23',
    #     '-preset', 'medium',
    #     '-vf', f'scale={width}:{height}', # Added this for explicit resolution
    #     op_video_path
    # ]



    # try:
    #     # This is the line that executes ffmpeg.exe
    #     ffmpeg_process = subprocess.run(ffmpeg_command + ['-loglevel', 'verbose'], check=False, capture_output=True, text=True)

    #     if ffmpeg_process.returncode != 0:
    #         print(f"\nFATAL ERROR: FFmpeg command failed with return code {ffmpeg_process.returncode}.")
    #         print(f"Stderr: \n{ffmpeg_process.stderr}") # <--- This will print FFmpeg's error messages
    #         print(f"Stdout: \n{ffmpeg_process.stdout}") # <--- This will print FFmpeg's general output
    #         print("--- End FFmpeg Error Output ---")
    #     else:
    #         print(f"\nSUCCESS: FFmpeg command completed with return code {ffmpeg_process.returncode}.")
    #         print(f"FFmpeg Stdout (might contain warnings):\n{ffmpeg_process.stdout}")
    #         if ffmpeg_process.stderr: # Sometimes warnings appear in stderr even on success
    #             print(f"FFmpeg Stderr (might contain warnings):\n{ffmpeg_process.stderr}")
    #         print("--- End FFmpeg Output ---")

    # except Exception as e:
    #     print(f"\nAn unexpected error occurred during FFmpeg execution: {e}")

    # print("successfully performed img inference")
    

    ##annotation function##
    # model.eval()
    # segment_metafolder(model,img_path,meta_folder_path,"C:/Project/DL_HW/dataset")
    ##annotation function##

    ##train function##
    model.train()
    criterion = torch.nn.CrossEntropyLoss(ignore_index=-1)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    train(model,criterion,optimizer,img_path,"C:/Project/DL_HW/dataset/trainFrames")
    ##train function##

    # model.eval()
    # op=inference(model,input_tensor)   

    # create_overlay_visualization(original_img,op)

    print("successful homes")
if __name__ == "__main__":
    main()