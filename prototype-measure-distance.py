
# coding: utf-8

# In[2]:

from PIL import Image
#import matplotlib.pyplot as plt
import numpy as np
import pyrealsense2 as rs
import cv2
from tkinter.filedialog import askopenfilename

PLOT=False

def compute_distance(profile,refPt):
    prof = profile.get_stream(rs.stream.depth)
    intrin = prof.as_video_stream_profile().get_intrinsics()
    #print("Camera Parameters:\n {}".format(intrin))

    a=np.array(rs.rs2_deproject_pixel_to_point(intrin,refPt[0][:2],depth_map[refPt[0][:2][0],refPt[0][:2][1]]))
    b=np.array(rs.rs2_deproject_pixel_to_point(intrin,refPt[1][:2],depth_map[refPt[1][:2][0],refPt[1][:2][1]]))

    D = np.sqrt((a-b).dot(a-b))
    
    d = "distance between dots = {}mm".format(D)
    print(d)
    return D


if __name__=="__main__":
    filename = askopenfilename() 
    # Setup:
    pipe = rs.pipeline()
    cfg = rs.config()

    cfg.enable_device_from_file(filename)
    profile = pipe.start(cfg)

    # Skip 5 first frames to give the Auto-Exposure time to adjust
    for x in range(75):
        pipe.wait_for_frames()

    # Store next frameset for later processing:
    frameset = pipe.wait_for_frames()
    color_frame = frameset.get_color_frame()
    depth_frame = frameset.get_depth_frame()

    # Cleanup:
    pipe.stop()
    print("Frames Captured")


    # In[31]:


    color = np.asanyarray(color_frame.get_data())
    if  PLOT:
        plt.rcParams["axes.grid"] = False
        plt.rcParams['figure.figsize'] = [12, 6]
        plt.imshow(color)
        Image.fromarray(color).save("images/3.png")
        np.save("images/3.npy",np.array(depth_frame.get_data()))


    # In[32]:

    colorizer = rs.colorizer()
    if PLOT:
        
        colorized_depth = np.asanyarray(colorizer.colorize(depth_frame).get_data())
        plt.imshow(colorized_depth)


    # In[33]:


    # Create alignment primitive with color as its target stream:
    align = rs.align(rs.stream.color)
    frameset = align.process(frameset)

    # Update color and depth frames:
    aligned_depth_frame = frameset.get_depth_frame()
    colorized_depth = np.asanyarray(colorizer.colorize(aligned_depth_frame).get_data())

    # Show the two frames together:
    images = np.hstack((color, colorized_depth))
    if PLOT: plt.imshow(images)

    #keep depth_map array
    depth_map = np.asanyarray(aligned_depth_frame.get_data())

    refPt = []
    img = color.copy()
    def click(event,x,y,flags,param):
        global refPt
        if event == cv2.EVENT_LBUTTONDOWN:
            refPt.append([y,x,1])
        if event == cv2.EVENT_LBUTTONUP:
            refPt.append([y,x,1])
            cv2.line(img,pt1=(refPt[0][1],refPt[0][0]),pt2=(x,y),color=(255,255,255),thickness=3)
            _x,_y = refPt[0][1]+0.5*(x-refPt[0][1]),refPt[0][0]+0.5*(y-refPt[0][0])
            
            try:
                d=compute_distance(profile,refPt)
                cv2.putText(img,"%.2f"%d, (int(_x),int(_y)), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0,0,255))
            except Exception as e:
                print("WARNING:{}".format(e))
            refPt=[]
            
            
            
    cv2.namedWindow("image")
    cv2.setMouseCallback("image",click)
    while True:
        cv2.imshow("image",img)
        key = cv2.waitKey(1)& 0xFF
        if key==ord("c"):
            break
        if key==ord("r"):
            img = color.copy()
    cv2.destroyAllWindows()