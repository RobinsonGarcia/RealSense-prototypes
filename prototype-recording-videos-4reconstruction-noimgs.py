# pyrealsense2 is required.
# Please see instructions in https://github.com/IntelRealSense/librealsense/tree/master/wrappers/python
import pyrealsense2 as rs
import numpy as np
import cv2
import argparse
from os import makedirs
from os.path import exists, join
import shutil
import json
from enum import IntEnum
import time
import os

try:
    # Python 2 compatible
    input = raw_input
except NameError:
    pass

"""misc"""
def check_make_dir(root):    
    if not os.path.exists(root):
            os.mkdir(root)
    pass

def get_next_idx(root):

    idx = np.array([int(folder) for folder in os.listdir(root) if os.path.isdir(os.path.join(root,folder))])

    if len(idx)==0:
        newbag = "1"
    else:
        newbag = str(np.sort(idx)[-1]+1)

    return newbag


class Preset(IntEnum):
    Custom = 0
    Default = 1
    Hand = 2
    HighAccuracy = 3
    HighDensity = 4
    MediumDensity = 5


def make_clean_folder(path_folder):
    if not exists(path_folder):
        makedirs(path_folder)
    else:
        user_input = input("%s not empty. Overwrite? (y/n) : " % path_folder)
        if user_input.lower() == 'y':
            shutil.rmtree(path_folder)
            makedirs(path_folder)
        else:
            exit()


def save_intrinsic_as_json(filename, frame):
    intrinsics = frame.profile.as_video_stream_profile().intrinsics
    with open(filename, 'w') as outfile:
        obj = json.dump(
            {
                'width':
                    intrinsics.width,
                'height':
                    intrinsics.height,
                'intrinsic_matrix': [
                    intrinsics.fx, 0, 0, 0, intrinsics.fy, 0, intrinsics.ppx,
                    intrinsics.ppy, 1
                ]
            },
            outfile,
            indent=4)


if __name__ == "__main__":



    class args:
        output_folder = 'recorded_bag_videos4reconst'
        record_rosbag = True
        record_imgs = False
        playback_rosbag = False

    root = 'recorded_bag_videos'
    check_make_dir(args.output_folder)
    newbag = get_next_idx(args.output_folder)


    path_output = os.path.join(args.output_folder,newbag)
    path_depth = join(path_output, "depth")
    path_color = join(path_output,"color")
    make_clean_folder(path_output)
    if args.record_imgs:
        
        make_clean_folder(path_depth)
        make_clean_folder(path_color)

    path_bag = join(path_output, newbag+".bag")

    """
    if args.record_rosbag:
        if exists(path_bag):
            user_input = input("%s exists. Overwrite? (y/n) : " % path_bag)
            if user_input.lower() == 'n':
                exit()
    """

    # Set advanced mode
    DS5_product_ids = ["0AD1", "0AD2", "0AD3", "0AD4", "0AD5", "0AF6", "0AFE", "0AFF", "0B00", "0B01", "0B03", "0B07","0B3A"]
    def find_device_that_supports_advanced_mode() :
        ctx = rs.context()
        ds5_dev = rs.device()
        devices = ctx.query_devices();
        for dev in devices:
            if dev.supports(rs.camera_info.product_id) and str(dev.get_info(rs.camera_info.product_id)) in DS5_product_ids:
                if dev.supports(rs.camera_info.name):
                    print("Found device that supports advanced mode:", dev.get_info(rs.camera_info.name))
                return dev
        raise Exception("No device that supports advanced mode was found")
        
    dev = find_device_that_supports_advanced_mode()
    advnc_mode = rs.rs400_advanced_mode(dev)
    print("Advanced mode is", "enabled" if advnc_mode.is_enabled() else "disabled")
    
    # Loop until we successfully enable advanced mode
    while not advnc_mode.is_enabled():
        print("Trying to enable advanced mode...")
        advnc_mode.toggle_advanced_mode(True)
        # At this point the device will disconnect and re-connect.
        print("Sleeping for 5 seconds...")
        time.sleep(5)
        # The 'dev' object will become invalid and we need to initialize it again
        dev = find_device_that_supports_advanced_mode()
        advnc_mode = rs.rs400_advanced_mode(dev)
        print("Advanced mode is", "enabled" if advnc_mode.is_enabled() else "disabled")

    # Create a pipeline
    pipeline = rs.pipeline()

    #Create a config and configure the pipeline to stream
    #  different resolutions of color and depth streams
    config = rs.config()

    if args.record_imgs or args.record_rosbag:
        # note: using 640 x 480 depth resolution produces smooth depth boundaries
        #       using rs.format.bgr8 for color image format for OpenCV based image visualization
        config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        if args.record_rosbag:
            config.enable_record_to_file(path_bag)
    if args.playback_rosbag:
        config.enable_device_from_file(path_bag, repeat_playback=True)

    # Start streaming
    profile = pipeline.start(config)
    depth_sensor = profile.get_device().first_depth_sensor()

    # Using preset HighAccuracy for recording
    if args.record_rosbag or args.record_imgs:
        depth_sensor.set_option(rs.option.visual_preset, Preset.HighAccuracy)

    # Getting the depth sensor's depth scale (see rs-align example for explanation)
    depth_scale = depth_sensor.get_depth_scale()

    # We will not display the background of objects more than
    #  clipping_distance_in_meters meters away
    clipping_distance_in_meters = 3  # 3 meter
    clipping_distance = clipping_distance_in_meters / depth_scale

    # Create an align object
    # rs.align allows us to perform alignment of depth frames to others frames
    # The "align_to" is the stream type to which we plan to align depth frames.
    align_to = rs.stream.color
    align = rs.align(align_to)

    # Streaming loop
    frame_count = 0
    try:
        while True:
            # Get frameset of color and depth
            frames = pipeline.wait_for_frames()

            # Align the depth frame to color frame
            aligned_frames = align.process(frames)

            # Get aligned frames
            aligned_depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()

            # Validate that both frames are valid
            if not aligned_depth_frame or not color_frame:
                continue

            depth_image = np.asanyarray(aligned_depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())

            if args.record_imgs:
                if frame_count == 0:
                    save_intrinsic_as_json(
                        join(path_output, "camera_intrinsic.json"),
                        color_frame)
                cv2.imwrite("%s/%06d.png" % \
                        (path_depth, frame_count), depth_image)
                cv2.imwrite("%s/%06d.jpg" % \
                        (path_color, frame_count), color_image)
                print("Saved color + depth image %06d" % frame_count)
                frame_count += 1

            # Remove background - Set pixels further than clipping_distance to grey
            grey_color = 153
            #depth image is 1 channel, color is 3 channels
            depth_image_3d = np.dstack((depth_image, depth_image, depth_image))
            bg_removed = np.where((depth_image_3d > clipping_distance) | \
                    (depth_image_3d <= 0), grey_color, color_image)

            # Render images
            depth_colormap = cv2.applyColorMap(
                cv2.convertScaleAbs(depth_image, alpha=0.09), cv2.COLORMAP_JET)
            images = np.hstack((bg_removed, depth_colormap))
            cv2.namedWindow('Recorder Realsense', cv2.WINDOW_AUTOSIZE)
            cv2.imshow('Recorder Realsense', images)
            key = cv2.waitKey(1)

            # if 'esc' button pressed, escape loop and exit program
            if key == 27:
                cv2.destroyAllWindows()
                break
    finally:
        pipeline.stop()

    time.sleep(5)
