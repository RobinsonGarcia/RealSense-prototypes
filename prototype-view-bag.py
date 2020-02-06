## License: Apache 2.0. See LICENSE file in root directory.
## Copyright(c) 2015-2017 Intel Corporation. All Rights Reserved.

###############################################
##      Open CV and Numpy integration        ##
###############################################

import pyrealsense2 as rs
import numpy as np
import cv2
import os




"""Modified by Robinson Garcia"""
print(""" 
AUTHOR: Robinson Garcia  \n
DATE: 1/11/2019 \n
COMENTS: \n
used the intelrealsense SDK as a starting point
""")


"""misc"""
def check_make_dir(root):    
    if not os.path.exists(root):
            os.mkdir(root)
    pass

def get_next_idx(root):
    if len(os.listdir(root))==0:
        return "1"
    else:
        idx = np.array([int(file.split(".")[0]) for file in os.listdir(root) if file.split(".")[1]=="bag"])

        newbag = str(np.sort(idx)[-1]+1)
        return newbag
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
            saveto = os.path.join(root,newbag+"_"+str(count)+".png")
            cv2.imwrite(saveto,img)
            continue
        if key==ord("r"):
            img = color_image.copy()
            continue

    cv2.destroyAllWindows()

if __name__=="__main__" :
    FROM_BAG=True



    # Configure depth and color streams
    pipeline = rs.pipeline()
    config = rs.config()

    if FROM_BAG:
        filename = askopenfilename()  #added by Robinson
        config.enable_device_from_file(filename)
    else:
        config.enable_record_to_file(os.path.join(root,newbag +'.bag'))#added by Robinson
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    root = 'viewed_bag_videos'
    check_make_dir(root)
    newbag = get_next_idx(root)


    # Start streaming
    profile = pipeline.start(config)
    pc = rs.pointcloud()
    count=0
    try:
        while True:

            # Wait for a coherent pair of frames: depth and color
            frames = pipeline.wait_for_frames()
            #depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            
            #if not depth_frame or not color_frame:
            #    continue

            # Create alignment primitive with color as its target stream:
            align = rs.align(rs.stream.color)
            frames = align.process(frames)

            # Update color and depth frames:
            colorizer = rs.colorizer()
            aligned_depth_frame = frames.get_depth_frame()
            #colorized_depth = np.asanyarray(colorizer.colorize(aligned_depth_frame).get_data())
            
            depth_image = np.asanyarray(aligned_depth_frame.get_data())
                        
            # Convert images to numpy arrays
            #depth_image = np.asanyarray(depth_frame.get_data())
            color_image = cv2.cvtColor(np.asanyarray(color_frame.get_data()),cv2.COLOR_BGR2RGB)

            # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

            # Stack both images horizontally
            images = np.hstack((color_image, depth_colormap))
            

            # Show images
            cv2.namedWindow('Reading data  **prototype developed by Robinson Garcia', cv2.WINDOW_AUTOSIZE)
            cv2.imshow('Reading data  **prototype developed by Robinson Garcia', images)
            
            key = cv2.waitKey(1)& 0xFF
            
            if key == ord("q"):
                pipeline.stop()
                break
            
            if key == ord('p'):
                measure_line(color_image,depth_image)
                
            if key == ord('g'):
                pc.map_to(color_frame)
                points = pc.calculate(aligned_depth_frame)
                saveto = os.path.join(root,newbag+"_"+str(count)+".ply")
                points.export_to_ply(saveto, color_frame)

            if key == ord('n'):
                filename = askopenfilename() 
                config.enable_device_from_file(filename)
                pipeline.stop()
                profile = pipeline.start(config)
                
            count+=1
                

    finally:

        # Stop streaming
        try:
            pipeline.stop()
        except:
            print("pipeline stopped")
    cv2.destroyAllWindows()