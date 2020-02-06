import pyrealsense2 as rs
import numpy as np
import cv2
import os
import time #added by Robins
import open3d as o3d


#https://github.com/IntelRealSense/librealsense/issues/1231
#https://github.com/IntelRealSense/librealsense/tree/development/examples/measure

"""Modified by Robinson Garcia"""
print(""" 
AUTHOR: Robinson Garcia  \n
DATE: 1/11/2019 \n
COMENTS: \n
used the intelrealsense SDK as a starting point
""")

def enable_advmode():
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

from enum import IntEnum
class Preset(IntEnum):
    Custom = 0
    Default = 1
    Hand = 2
    HighAccuracy = 3
    HighDensity = 4
    MediumDensity = 5
"""end misc"""

from tkinter.filedialog import askopenfilename #added by Robinson


def compute_distance(profile,depth_map,refPt):
    prof = profile.get_stream(rs.stream.depth)
    intrin = prof.as_video_stream_profile().get_intrinsics()

    print("Camera Parameters:\n {}".format(intrin))

    a=np.array(rs.rs2_deproject_pixel_to_point(intrin,refPt[0][:2],depth_map[refPt[0][:2][0],refPt[0][:2][1]]))
    b=np.array(rs.rs2_deproject_pixel_to_point(intrin,refPt[1][:2],depth_map[refPt[1][:2][0],refPt[1][:2][1]]))
    
    D = np.sqrt((a-b).dot(a-b))
    
    d = "distance between dots = {}mm".format(D)
    print(d)
    return D


refPt = []    
def measure_line(color_image,depth_image):
    refPt = []
    
    img = color_image.copy()
    drawing=False

    def click(event,x,y,flags,param):
        global refPt
        
        if event == cv2.EVENT_LBUTTONDOWN:
            refPt.append([y,x,1])
        if event == cv2.EVENT_LBUTTONUP:
            refPt.append([y,x,1])
            cv2.line(img,pt1=(refPt[0][1],refPt[0][0]),pt2=(x,y),color=(255,255,255),thickness=3)
            _x,_y = refPt[0][1]+0.5*(x-refPt[0][1]),refPt[0][0]+0.5*(y-refPt[0][0])

            try:
                d=compute_distance(profile,depth_image,refPt)
                cv2.putText(img,"%.2f"%d, (int(_x)+5,int(_y)+5), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0,0,255))
            except Exception as e:
                print("WARNING:{}".format(e))
            
            refPt=[]
            

    cv2.namedWindow('Measuring data  **prototype developed by Robinson Garcia')
    cv2.setMouseCallback('Measuring data  **prototype developed by Robinson Garcia',click)
    while True:
        cv2.imshow('Measuring data  **prototype developed by Robinson Garcia',img)
        key = cv2.waitKey(1)& 0xFF
        if key==ord("q"):
            break
        if key==ord("s"):
            saveto = os.path.join(path_measures,newbag+"_"+str(count)+".png")
            cv2.imwrite(saveto,img)
            continue
        if key==ord("r"):
            img = color_image.copy()
            continue

    cv2.destroyAllWindows()

if __name__=="__main__":
    FROM_BAG=False
    class args:
        output_folder = 'recorded_bag_videos'

    #root = 'recorded_bag_videos'
    check_make_dir(args.output_folder)
    newbag = get_next_idx(args.output_folder)


    path_output = os.path.join(args.output_folder,newbag)
    check_make_dir(path_output)

    path_bag = os.path.join(path_output, newbag+".bag")
    
    path_measures = os.path.join(path_output,"measures" )
    check_make_dir(path_measures)

    path_ply = os.path.join(path_output,"point_cloud" )
    check_make_dir(path_measures)

    # Configure depth and color streams
    pipeline = rs.pipeline()
    config = rs.config()
    enable_advmode()
    if FROM_BAG:
        filename = askopenfilename()  #added by Robinson
        config.enable_device_from_file(filename)
    else:
        config.enable_record_to_file(path_bag)#added by Robinson
        #config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    # filters
    spatial = rs.spatial_filter()
    spatial.set_option(rs.option.filter_magnitude, 5)
    spatial.set_option(rs.option.filter_smooth_alpha, 1)
    spatial.set_option(rs.option.filter_smooth_delta, 50)
    spatial.set_option(rs.option.holes_fill, 3)

    hole_filling = rs.hole_filling_filter()


    # Start streaming
    profile = pipeline.start(config)
    depth_sensor = profile.get_device().first_depth_sensor()


    # Using preset HighAccuracy for recording
    depth_sensor.set_option(rs.option.visual_preset, Preset.HighAccuracy)

    # get scale and compute clipping distance
    depth_scale = depth_sensor.get_depth_scale()
    clipping_distance_in_meters = 10 # 3 meter
    clipping_distance = clipping_distance_in_meters / depth_scale


    pc = rs.pointcloud()
    count=0

    # create an align object
    align = rs.align(rs.stream.color)
    try:
        while True:

            # Wait for a coherent pair of frames: depth and color
            frames = pipeline.wait_for_frames()

            # Align the depth frame to color frame
            aligned_frames = align.process(frames)

            # Get aligned frames
            aligned_depth_frame = aligned_frames.get_depth_frame()
            aligned_depth_frame=spatial.process(aligned_depth_frame)
            aligned_depth_frame=hole_filling.process(aligned_depth_frame)
            
            
            color_frame = aligned_frames.get_color_frame()

            # Validate that both frames are valid
            if not aligned_depth_frame or not color_frame:
                continue    

            # Convert frames to numpy arrays
            depth_image = np.asanyarray(aligned_depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())
                        
            # Remove background - Set pixels further than clipping_distance to grey
            grey_color = 153
            #depth image is 1 channel, color is 3 channels
            depth_image_3d = np.dstack((depth_image, depth_image, depth_image))
            bg_removed = np.where((depth_image_3d > clipping_distance) | \
                    (depth_image_3d <= 0), grey_color, color_image)

            #colorizer = rs.colorizer()
            # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.09), cv2.COLORMAP_JET)

            # Stack both images horizontally
            images = np.hstack((bg_removed, depth_colormap))
            

            # Show images
            cv2.namedWindow('Capturing data  **prototype developed by Robinson Garcia', cv2.WINDOW_AUTOSIZE)
            cv2.imshow('Capturing data  **prototype developed by Robinson Garcia', images)
            
            key = cv2.waitKey(1)& 0xFF
            
            if key == ord("q"):
                pipeline.stop()
                break
            
            if key == ord('p'):
                measure_line(bg_removed,depth_image)
                
            if key == ord('g'):
                pc.map_to(color_frame)
                points = pc.calculate(aligned_depth_frame)
                saveto = os.path.join(path_ply,newbag+"_"+str(count)+".ply")
                points.export_to_ply(saveto, color_frame)
                pcd = o3d.io.read_point_cloud(saveto)
                o3d.visualization.draw_geometries([pcd])
            count+=1
                

    finally:

        # Stop streaming
        # pipeline.stop()
        try:
            pipeline.stop()
        except:
            print("pipeline stopped")
        time.sleep(5)
        cv2.destroyAllWindows()
 