import numpy as np
from PIL import Image
import cv2
from deepface import DeepFace
from ..detectors.dlib_resnet_wrapper import *
import torch
from torchvision import transforms
import math
from .filter_functions import *
from . import faceBlendCommon as fbc

def filter_frame(frame: np.array, filter_name: str, model: torch.nn.Module = None, transforms: transforms = None, points2Prev = None, img2GrayPrev = None):
    # frame_image
    frame_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    width = frame_image.width

    # prepare filter
    filters, multi_filter_runtime = load_filter(filter_name=filter_name)

    # detect faces
    resp_objs = DeepFace.extract_faces(img_path=frame, detector_backend="opencv", enforce_detection=False)
    if resp_objs is not None:
        for resp_obj in resp_objs:
            # deal with extract_faces
            if resp_obj["facial_area"]["w"] == width:
                break
            box = resp_obj["facial_area"]

            # prepare input image: crop & transform
            input_image = frame_image.crop(box=(box["x"], box["y"], box["x"] + box["w"], box["y"] + box["h"]))
            landmarks = get_landmarks(input_image, model, transforms)
            landmarks = landmarks + np.array([box["x"], box["y"]])

            # get points2 ~ points on face
            landmarks = np.vstack([landmarks, np.array([landmarks[0][0], int(box["y"])])])
            landmarks = np.vstack([landmarks, np.array([landmarks[16][0], int(box["y"])])])
            points2 = landmarks.tolist()

            ################ Optical Flow and Stabilization Code #####################
            img2Gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            if points2Prev is None and img2GrayPrev is None:
                points2Prev = np.array(points2, np.float32)
                img2GrayPrev = np.copy(img2Gray)

            lk_params = dict(winSize=(101, 101), maxLevel=15,
                            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.001))
            points2Next, st, err = cv2.calcOpticalFlowPyrLK(img2GrayPrev, img2Gray, points2Prev,
                                                            np.array(points2, np.float32),
                                                            **lk_params)

            # Final landmark points are a weighted average of detected landmarks and tracked landmarks

            for k in range(0, len(points2)):
                d = cv2.norm(np.array(points2[k]) - points2Next[k])
                alpha = math.exp(-d * d / 50)
                points2[k] = (1 - alpha) * np.array(points2[k]) + alpha * points2Next[k]
                points2[k] = fbc.constrainPoint(points2[k], frame.shape[1], frame.shape[0])
                points2[k] = (int(points2[k][0]), int(points2[k][1]))

            # Update variables for next pass
            points2Prev = np.array(points2, np.float32)
            img2GrayPrev = img2Gray
            ################ End of Optical Flow and Stabilization Code ###############
            
            # applying filter
            for idx, filter in enumerate(filters):

                filter_runtime = multi_filter_runtime[idx]
                img1 = filter_runtime['img']
                points1 = filter_runtime['points']
                img1_alpha = filter_runtime['img_a']

                if filter["morph"]:
                    hull1 = filter_runtime['hull']
                    hullIndex = filter_runtime['hullIndex']
                    dt = filter_runtime['dt']

                    # create copy of frame
                    warped_img = np.copy(frame)

                    # Find convex hull
                    hull2 = []
                    for i in range(0, len(hullIndex)):
                        hull2.append(points2[hullIndex[i][0]])

                    mask1 = np.zeros((warped_img.shape[0], warped_img.shape[1]), dtype=np.float32)
                    mask1 = cv2.merge((mask1, mask1, mask1))
                    img1_alpha_mask = cv2.merge((img1_alpha, img1_alpha, img1_alpha))

                    # Warp the delaunay triangles
                    for i in range(0, len(dt)):
                        t1 = []
                        t2 = []

                        for j in range(0, 3):
                            t1.append(hull1[dt[i][j]])
                            t2.append(hull2[dt[i][j]])

                        fbc.warpTriangle(img1, warped_img, t1, t2)
                        fbc.warpTriangle(img1_alpha_mask, mask1, t1, t2)

                    # Blur the mask before blending
                    mask1 = cv2.GaussianBlur(mask1, (3, 3), 10)

                    mask2 = (255.0, 255.0, 255.0) - mask1

                    # Perform alpha blending of the two images
                    temp1 = np.multiply(warped_img, (mask1 * (1.0 / 255)))
                    temp2 = np.multiply(frame, (mask2 * (1.0 / 255)))
                    output = temp1 + temp2
                else:
                    dst_points = [points2[int(list(points1.keys())[0])], points2[int(list(points1.keys())[1])]]
                    tform = fbc.similarityTransform(list(points1.values()), dst_points)

                    # Apply similarity transform to input image
                    trans_img = cv2.warpAffine(img1, tform, (frame.shape[1], frame.shape[0]))
                    trans_alpha = cv2.warpAffine(img1_alpha, tform, (frame.shape[1], frame.shape[0]))
                    mask1 = cv2.merge((trans_alpha, trans_alpha, trans_alpha))

                    # Blur the mask before blending
                    mask1 = cv2.GaussianBlur(mask1, (3, 3), 10)

                    mask2 = (255.0, 255.0, 255.0) - mask1

                    # Perform alpha blending of the two images
                    temp1 = np.multiply(trans_img, (mask1 * (1.0 / 255)))
                    temp2 = np.multiply(frame, (mask2 * (1.0 / 255)))
                    output = temp1 + temp2
                
                frame = np.array(output, dtype=np.uint8)
    
    return frame, points2Prev, img2GrayPrev

if __name__ == "__main__":
    # prepare video capture
    source = "apply_filter/test/data/videos/IMG_1843.MOV"
    cap = cv2.VideoCapture(source)
    cap_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    # create the output video file
    if source != 0:
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(filename='annotated_video.mp4', fourcc=fourcc, fps=fps, frameSize=frame_size)

    # create model & transform
    model, simple_transform = load_model()
    
    # loop through each frame
    isFirstFrame = True
    while(cap.isOpened()):
        # common
        ret, frame = cap.read()
        if ret == False:
            break
        if source == 0:
            frame = cv2.flip(frame, 1)

        # apply filter
        frame = filter_frame(frame, "squid_game_front_man", model, simple_transform)
        
        if source == 0:
            # fps
            # cur_time = time.time()
            # print(1 / (cur_time - prev_time))
            # prev_time = cur_time

            # show frame
            cv2.imshow("Filter App", frame)

            # handle keypress
            keypressed = cv2.waitKey(1) & 0xFF
            if keypressed == 27:
                break
            elif keypressed == ord('f'):
                try:
                    filters, multi_filter_runtime = load_filter(next(iter_filter_keys))
                except:
                    iter_filter_keys = iter(filters_config.keys())
                    filters, multi_filter_runtime = load_filter(next(iter_filter_keys))
        else:
            out.write(frame)
    
    # save & free resource
    cap.release()
    if source != 0:
        out.release()
    cv2.destroyAllWindows()