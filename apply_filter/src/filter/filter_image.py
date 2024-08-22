from deepface import DeepFace
from apply_filter.src.filter.filter_functions import *
import numpy as np
from PIL import Image
from apply_filter.src.detectors.dlib_resnet_wrapper import *

def filter_image(image: np.array, filter_name = "squid_game_front_man")->Image:
    """Apply filter on faces in image"""

    # handle image
    width = image.width
    frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    # prepare filter
    filters, multi_filter_runtime = load_filter(filter_name=filter_name)

    # detect faces
    resp_objs = DeepFace.extract_faces(img_path=frame, enforce_detection=False)

    if resp_objs is not None:
        for resp_obj in resp_objs:
            # deal with extract_faces
            if resp_obj["facial_area"]["w"] == width:
                break
            box = resp_obj["facial_area"]

            # get landmarks
            input_image = image.crop(box=(box["x"], box["y"], box["x"] + box["w"], box["y"] + box["h"]))
            landmarks = get_landmarks(input_image)
            landmarks = landmarks + np.array([box["x"], box["y"]])

            # get points2 ~ points on face
            landmarks = np.vstack([landmarks, np.array([landmarks[0][0], int(box["y"])])])
            landmarks = np.vstack([landmarks, np.array([landmarks[16][0], int(box["y"])])])
            points2 = landmarks.tolist()
            
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
                
                frame = output = np.uint8(output)

    rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(np.uint8(rgb_image))    

    return pil_image

if __name__ == "__main__":

    image = Image.open("apply_filter/test/data/images/example_1.png")
    image = filter_image(image=image)
    image.save("apply_filter/test/output/example_1.png")
