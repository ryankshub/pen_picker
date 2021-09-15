#!/usr/bin/env python3
# Pen Picker main file
# RKS, Sept 15th

# Project imports
# None
# Python imports
import cv2 as cv
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

            depth_image = np.asanyarray(aligned_depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())
            #grayscale_image = cv.cvtColor(color_image, cv.COLOR_BGR2GRAY)

            light_purple_hsv = np.array([122, 100, 100])
            dark_purple_hsv = np.array([142, 255, 255])

            mask = cv.inRange(hsv_image, light_purple_hsv, dark_purple_hsv)

            filter_image = cv.bitwise_and(color_image, color_image, mask=mask)


            cv.imshow('Standard RGB', color_image)
            #cv.imshow('Grayscale', grayscale_image)
            cv.imshow('Mask', mask)
            cv.imshow('Filtered', filter_image)
            key = cv.waitKey(1)
            if key == 27:
                cv.destroyAllWindows()
                break
    finally:
        pipeline.stop()


if __name__ == '__main__':
    pipeline, config = configure_camera()
    stream_camera(pipeline, config)

