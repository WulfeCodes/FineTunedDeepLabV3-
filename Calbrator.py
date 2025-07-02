import torch
import cv2
import numpy as np
from depthai import Device
import depthai as dai
import os
class CharucoBoard:
    def __init__(self,nX=6, nY = 4, squareSize=4,markerSize =2.9):
        self.nX = nX
        self.nY = nY
        self.squareSize = squareSize
        self.markerSize=markerSize
        self.dictionary = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)
        self.board = cv2.aruco.CharucoBoard_create(
            nX, nY, squareSize, markerSize, self.dictionary) 
        self.all_corners = []
        self.all_ids = []
        
        # Create output directory
        self.output_dir = os.path.join("./calibration_images")
        self.output_dir.mkdir(exist_ok=True)    

class LuxCam:
    def __init__(self):
        self.pipeline = dai.Pipeline()
        self.cam = self.pipeline.create(dai.node.ColorCamera)
        self.cam.setPreviewSize(1280, 720)
        self.cam.setInterleaved(False)
        self.out = self.pipeline.create(dai.node.XLinkOut)
        self.out.setStreamName("rgb")
        self.cam.preview.link(self.out.input)

    def calibrateDatHoe(self,CharucoBoard):
        with dai.Device(self.pipeline) as device:
            q = device.getOutputQueue("rgb")
            
            while True:
                frame = q.get().getCvFrame()
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                # Detect markers
                corners, ids, _ = cv2.aruco.detectMarkers(gray, CharucoBoard.dictionary)
                
                if ids is not None:
                    # Get charuco corners
                    ret, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
                        corners, ids, gray, CharucoBoard.board)
                    
                    if ret > 6:  # Need at least 6 corners
                        cv2.aruco.drawDetectedCornersCharuco(frame, charuco_corners, charuco_ids)
                        cv2.putText(frame, f"Corners: {ret}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
                
                cv2.putText(frame, f"Images: {len(CharucoBoard.all_corners)}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
                cv2.imshow("Calibration", frame)
                print("to continue with image captire press space, to quit type q")
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord(' ') and 'charuco_corners' in locals() and ret > 6:
                    CharucoBoard.all_corners.append(charuco_corners)
                    CharucoBoard.all_ids.append(charuco_ids)
                    print(f"Captured {len(CharucoBoard.all_corners)} images")
                    
                elif key == ord('q') and len(CharucoBoard.all_corners) > 10:
                    break            


def main():
    cBoard = CharucoBoard()
    Camera = LuxCam()
    Camera.calibrateDatHoe(cBoard)
    pass

if __name__=='__main__':
    main()