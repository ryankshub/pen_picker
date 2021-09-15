#!/usr/bin/env python3
# Pen Picker main file
# RKS, Sept 15th

# Project imports
# None
# Python imports
import cv2
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

    return config

def stream_camera(config):
    """
    Stream from the camera given a pyrealsense.config 
    """
    # Start streaming
    profile = pipeline.start(config)


if __name__ == '__main__':
    configure_camera()

