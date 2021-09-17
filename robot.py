#!/usr/bin/env python3
# Pen Picker main file
# RKS, Sept 15th

# Project imports
# None
# Python imports
import cv2 as cv
import math
import numpy as np
# 3rd-party imports
import pyrealsense2 as rs

## Vision component
def configure_camera():
    """
    Configure the camera to locate the purple pen
    """
    #Create a pipeline and config object. 
    #These are used to configure camera
    #See pyrealsense API for more info
    print("Configuring Camera")
    pipeline = rs.pipeline()
    config = rs.config()

    pipeline_wrapper = rs.pipeline_wrapper(pipeline)
    pipeline_profile = config.resolve(pipeline_wrapper)
    device = pipeline_profile.get_device()
    device_product_line = str(device.get_info(rs.camera_info.product_line))
    print(f"Found device: {device_product_line}")

    sensors_str = "\nFound sensors: \n"
    found_stereo = False
    found_rgb = False

    #Gather sensors; check for stereo and rgb camera
    for sensor in device.sensors:
        sensor_name = sensor.get_info(rs.camera_info.name)
        if sensor_name == "Stereo Module":
            found_stereo = True
        elif sensor_name == "RGB Camera":
            found_rgb = True

        sensors_str += sensor_name + '\n'
    
    print(sensors_str)

    if found_stereo and found_rgb:
        print("Found required cameras!")
    else:
        raise RuntimeError("Could not find required cameras.\n"
            f"Found Stereo Camera: {found_stereo}\n"
            f"Found RGB Camera: {found_rgb}\n")

    #Enable streams
    depth_stream_width = 640
    depth_stream_height = 480
    depth_framerate = 30
    
    color_stream_width = 640
    color_stream_height = 480
    color_framerate = 30

    #Enable depth
    config.enable_stream(rs.stream.depth, 
        depth_stream_width, 
        depth_stream_height, 
        rs.format.z16,
        depth_framerate)
    #Enable color
    config.enable_stream(rs.stream.color,
        color_stream_width,
        color_stream_height,
        rs.format.bgr8,
        color_framerate)

    return pipeline, config


def apply_color_filter(image, hue=122):
    """
    Takes in an image and returns a filter image with only
    that color in it. 
    """
    #Apply color filter
    lower_hsv = np.array([hue-10, 100, 100])
    higher_hsv = np.array([hue+10, 255, 255])
    hsv_image = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    #Create and dilate mask
    mask = cv.inRange(hsv_image, lower_hsv, higher_hsv)
    dilation_elem = cv.getStructuringElement(cv.MORPH_RECT, (4,4))
    mask = cv.dilate(mask, dilation_elem)
    #Create filtered image
    filter_image = cv.bitwise_and(image, image, mask=mask)

    return filter_image, mask


def apply_threshold_filter(image, thres=85):
    """
    Apply Grayscale filter to image
    """
    gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    filter_image, mask = cv.threshold(gray_image, thres, 255, cv.THRESH_BINARY_INV)
    filter_image = cv.bitwise_and(image, image, mask=mask)
    return filter_image, mask

# TODO: Implement a color-difference function
# def apply_nn_policy(image, target_BGR):
#     """
#     Apply a makeshift nearest_neighbor filter.
#     The filter finds the closest color to the target color and
#     then applies a color filter. 
#     """
#     min_dist = float('inf') #Super High Number
#     best_py = None
#     best_px = None
#     for py in range(image.shape[0]):
#         for px in range(image.shape[1]):
#             if all(image[py, px] == 0):
#                 print("BLACK")
#                 continue
#             blue_diff = (image[py, px, 0] - target_BGR[0])**2
#             green_diff = (image[py, px, 1] - target_BGR[1])**2
#             red_diff = (image[py, px, 2] - target_BGR[2])**2
#             dist = math.sqrt(blue_diff + green_diff + red_diff)
#             if (dist < min_dist):
#                 best_py = py
#                 best_px = px
#                 min_dist = dist
    
#     best_match = np.uint8([[image[best_py, best_px]]])
#     best_match_HSV = cv.cvtColor(best_match, cv.COLOR_BGR2HSV)
#     filter_image, mask = apply_color_filter(image, best_match_HSV[0][0][0])
#     return filter_image, mask


def draw_centroid(image, cen_x, cen_y, draw_len=5):
    """
    Draws a red square on image to represent centroid

    Arguments:
        2D np.arrary: image - original image
        int: cen_x - x pixel coordinate of centroid
        int: cen_y - y pixel coordinate of centroid
        int: draw_len - pixel side length of drawn sqaure

    Returns:
        If cen_x & cen_y are not None, return image with square
        Else, return original image
    """
    if cen_x is None or cen_y is None:
        return image

    image_width = image.shape[1]
    image_length = image.shape[0]

    lower_x_bound = max(0, cen_x - draw_len)
    upper_x_bound = min(image_width, cen_x + draw_len)

    lower_y_bound = max(0, cen_y - draw_len)
    upper_y_bound = min(image_length, cen_y + draw_len)

    image[lower_y_bound:upper_y_bound, lower_x_bound:upper_x_bound] = [0,0,255]
    return image


def stream_camera(pipeline, config):
    """
    Stream from the camera given a pyrealsense.config 
    """
    # Start streaming
    profile = pipeline.start(config)

    #Get depth sensor (Stereo Module)
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    print(f"Depth Scale: {depth_scale}")

    #Ignore any object .5 meters away
    clip_dist_m = .5 # 1/2 meter
    clip_dist_px = clip_dist_m / depth_scale

    # Create align object, makes alignment of two streams
    align_to = rs.stream.color
    align = rs.align(align_to)

    #Streaming loop
    try:
        while True:
            # Get frames from pipeline [Blocking function]
            frames = pipeline.wait_for_frames()

            # Align the frames
            aligned_frames = align.process(frames)

            # Get aligned frames
            aligned_depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()

            # Check both frames are good
            if not aligned_depth_frame or not color_frame:
                continue

            #Grab data from depth frame and color frame
            depth_image = np.asanyarray(aligned_depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())
            
            #Set target color 
            northwestern_purple_BGR = [132, 42, 78]

            #Filter data from color frame
            thres_img, thres_mask = apply_threshold_filter(color_image)
            erosion_elem = cv.getStructuringElement(cv.MORPH_RECT, (4,4))
            thres_mask = cv.erode(thres_mask, erosion_elem)
                       
            #Find and draw contours on image
            cons, _ = cv.findContours(thres_mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
            cv.drawContours(thres_img, cons, 0, (0,255,0), 3)
            try:
                cnt = cons[0]
                M = cv.moments(cnt)
                cen_x = int(M['m10']/M['m00'])
                cen_y = int(M['m01']/M['m00'])
            except:
                # If no contours can be found, 
                # Do not define a centroid
                cen_x = None
                cen_y = None
            
            # #draw centroid if it exist
            filter_image = draw_centroid(thres_img, cen_x, cen_y)

            #Display feeds (for debugging)
            cv.imshow('Standard RGB', color_image)
            cv.imshow('Threshold Mask', thres_mask)
            cv.imshow('Thres Filtered', thres_img)
            #cv.imshow('HSV Mask', hsv_mask)
            #cv.imshow('HSV Filtered', hsv_img)
            key = cv.waitKey(1)
            if key == 27:
                cv.destroyAllWindows()
                break
    finally:
        pipeline.stop()


if __name__ == '__main__':
    pipeline, config = configure_camera()
    stream_camera(pipeline, config)

