from PIL import Image
from apply_filter.src.filter import filter_functions, filter_image_with_yolo, filter_video, filter_image
from apply_filter.src.functions import load_image, load_pil_image
import cv2
from apply_filter.src.detectors.dlib_resnet_wrapper import load_model

def apply_filter_on_image(image, filter_name = "squid_game_front_man", output_path = None)->Image:
    print("here")
    image = load_pil_image(image)
    image = filter_image.filter_image(image, filter_name)
    
    if output_path is None:
        return image
    else:
        image.save(output_path)
        
def apply_filter_on_video(source, filter_name = "squid_game_front_man", output_path = "output_video.mp4")->None:
    # prepare video capture
    cap = cv2.VideoCapture(source)
    cap_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    # create the output video file
    if source != 0:
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, frame_size)

    # create model & transform
    model, simple_transform = load_model()

    # prepare for filter
    if filter_name is None:
        iter_filter_keys = iter(filter_functions.filters_config.keys())
        filter_name = next(iter_filter_keys)
    
    # for optical flow
    points2Prev = None
    img2GrayPrev = None


    # loop through each frame
    while(cap.isOpened()):
        # common
        ret, frame = cap.read()
        if ret == False:
            break
        if source == 0:
            frame = cv2.flip(frame, 1)

        # apply filter
        frame, points2Prev, img2GrayPrev = filter_video.filter_frame(frame, filter_name, model, simple_transform, points2Prev, img2GrayPrev)

        if source == 0:
            # # fps
            # cur_time = time.time()
            # print(1 / (cur_time - prev_time))
            # prev_time = cur_time

            # show frame
            cv2.imshow("Filter app", frame)

            # handle keypress
            keypressed = cv2.waitKey(1) & 0xFF
            if keypressed == 27:
                break
            elif keypressed == ord('f'):
                try:
                    filter_name = next(iter_filter_keys)
                except:
                    iter_filter_keys = iter(filter_functions.filters_config.keys())
                    filter_name = next(iter_filter_keys)
        else:
            out.write(frame) 
    
    # save & free resource
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
    return output_path