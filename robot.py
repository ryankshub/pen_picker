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


if __name__ == '__main__':
    configure_camera()

