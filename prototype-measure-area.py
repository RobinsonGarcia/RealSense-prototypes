
# coding: utf-8
from PIL import Image
import numpy as np
import pyrealsense2 as rs
import cv2
from tkinter.filedialog import askopenfilename
import cv2



def loadRSdata(VIDEO_FILE,SKIP_FRAMES):
    """
    Output:
    RGB image
    Raw depthmap
    Colorized depthmap
    """
    
    """Read data"""
    pipe = rs.pipeline()
    cfg = rs.config()
    cfg.enable_device_from_file(VIDEO_FILE)
    profile = pipe.start(cfg)

    """Skip SKIP_FRAMES first frames to give the Auto-Exposure time to adjust"""
    for x in range(SKIP_FRAMES ):
        pipe.wait_for_frames()

    """Store next frameset for later processing:"""
    frameset = pipe.wait_for_frames()
    color_frame = frameset.get_color_frame()
    depth_frame = frameset.get_depth_frame()

    """Cleanup:"""
    pipe.stop()
    
    """Align depthmap and RGB image"""
    # Get color image
    color = np.asanyarray(color_frame.get_data())

    # Create alignment primitive with color as its target stream:
    align = rs.align(rs.stream.color)
    frameset = align.process(frameset)

    # Update color and depth frames:
    colorizer = rs.colorizer()
    aligned_depth_frame = frameset.get_depth_frame()
    colorized_depth = np.asanyarray(colorizer.colorize(aligned_depth_frame).get_data())

    # Show the two frames together:
    images = np.hstack((color, colorized_depth))

    # keep depth_map array
    depth_map = np.asanyarray(aligned_depth_frame.get_data())

    return color,colorized_depth,depth_map,profile


# # Compute 3D coordinates


def computeXYZ(profile,mask,depth_map,color):
    prof = profile.get_stream(rs.stream.depth)
    intrin = prof.as_video_stream_profile().get_intrinsics()
    #print("Camera Parameters:\n {}".format(intrin))

    w,h = np.argwhere(mask).T
    pts = []
    rgb = []
    for x,y in zip(w,h):
        pts.append(np.array(rs.rs2_deproject_pixel_to_point(intrin,[x,y],depth_map[x,y])))
        rgb.append(color[x,y,:])
    pts = np.array(pts)
    rgb = np.array(rgb)

    a,b= mask.size
    xx,yy = np.meshgrid(np.arange(a),np.arange(b))
    xx = xx.flatten()
    yy = yy.flatten()

    todos_pts = []
    todos_rgb = []
    for y,x in zip(xx,yy):
        todos_pts.append(np.array(rs.rs2_deproject_pixel_to_point(intrin,[x,y],depth_map[x,y])))
        todos_rgb.append(color[x,y,:])
    todos_pts = np.array(todos_pts)
    return todos_pts,pts,todos_rgb,rgb


# # Robust fitting of 3d planes

# In[5]:


def build_data3(p):
    N=p.shape[0]
    dataM=[p[:,0],p[:,1],    p[:,2]]
    return np.array(dataM).T


def RobustPlaneFittingRANSAC(mask,pts):
    w,h = np.argwhere(mask).T
    _w,_h = w,h 
    _pts = pts
    objs = []
    pts3D = []
    pts2D = []
    for j in range(10):
        inliers=0
        for i in range(100):
            N = _pts.shape[0]

            idx = np.random.choice(np.arange(N),1)
            c=_pts[idx].T

            d = np.sum((_pts - c.T)**2,axis=1)

            try:
                idx = np.argsort(d)[:10]
            except:
                break

            XX=build_data3(_pts[idx,:])

            XX_ = (XX - np.mean(XX,axis=0)[np.newaxis,:])
            C = (XX_.T.dot(XX_))/N

            _,_,Vh=np.linalg.svd(C)

            w=Vh[-1,:]

            d = w.dot(np.mean(XX,axis=0))    

            XX = build_data3(_pts)

            e = (XX.dot(w)-d)**2

            idx = e<1000

            if np.sum(inliers) < np.sum(idx):
                inliers=idx
                ee = (XX[inliers,:].dot(w)-d)**2
                best_w=np.append(w,d)

        #print( "obj1")
        #print(np.sum(inliers),_pts.shape)

        objs.append(_pts[inliers,:])
        pts2D.append([_h[inliers],_w[inliers]])
        if np.sum(np.logical_not(inliers))<1000:
            #print("end")
            break

        _pts = _pts[np.logical_not(inliers),:]
        _h = _h[np.logical_not(inliers)]
        _w = _w[np.logical_not(inliers)]
    return pts2D,objs,pts,_h,_w

    


# # Robust Fitting (to do)

def LSfitting(pts2D,objs,pts,_h,_w):
    total_area=0
    for i in range(len(pts2D)):
        data = objs[i]
        Cov=data.T.dot(data)/data.shape[0]

        U,S,Vh = np.linalg.svd(Cov)

        xx,yy=Vh[:2,:].dot((data - np.mean(data,axis=0)[np.newaxis,:]).T)
        h,w = pts2D[i]

        a = np.sqrt(S[0])*Vh[0]
        b = np.sqrt(S[1])*Vh[1]

        c=np.sqrt(np.sum(a**2))
        d=np.sqrt(np.sum(b**2))
        area = 4*c*d/(1000**2)
        total_area+=area
        
        #plt.imshow(color*(np.array(mask)[:,:,np.newaxis]/255).astype(np.uint8))

    return total_area


# # Save point cloud

def get_mask(shape):
    mask = np.zeros((shape))
    ctr = np.array(refPt).reshape((-1,1,2)).astype(np.int32)
    new_ctr = []
    for x,y in np.squeeze(ctr):
        new_ctr.append([y,x])
    ctr = np.array(new_ctr).reshape((-1,1,2)).astype(np.int32)
    mask = cv2.drawContours(mask,ctr,-1,255,-1)
    mask = cv2.fillPoly(mask, pts =[ctr], color=255)
    return Image.fromarray(mask),np.mean(np.squeeze(ctr),axis=0)


if __name__=="__main__":
    PTS=[]
    todos_PTS=[]
    SKIP_FRAMES = 10
    PATH_VIDEOS = 'videos'
    VIDEO_FILE =askopenfilename() 

    color,colorized_depth,depth_map,profile=loadRSdata(VIDEO_FILE,SKIP_FRAMES)


    refPt = []
    img = color.copy()
    H,W,_=img.shape
    drawing=False
    def click(event,x,y,flags,param):
        global refPt
        global drawing
        
        if (event == cv2.EVENT_LBUTTONDOWN)&(not drawing):
            refPt.append([y,x])
            drawing=True
            
        if (event == cv2.EVENT_LBUTTONDOWN)&(drawing):
            cv2.line(img,pt1=(refPt[-1][1],refPt[-1][0]),pt2=(x,y),color=(255,255,255),thickness=3)
            refPt.append([y,x])
    
    cv2.namedWindow("image")
    cv2.setMouseCallback("image",click)
    while True:
        cv2.imshow("image",img)
        key = cv2.waitKey(1)& 0xFF
        if key==ord("c"):
            break
        if key==ord("r"):
            img = color.copy()
            
        if key==ord("a"):
            mask,mu = get_mask((H,W))
            
            todos_pts,pts,todos_rgb,rgb = computeXYZ(profile,mask,depth_map,color)
            PTS.append(pts)
            todos_PTS.append(todos_pts)
            pts2D,objs,pts,_h,_w= RobustPlaneFittingRANSAC(mask,pts)
            area = LSfitting(pts2D,objs,pts,_h,_w)
            cv2.putText(img,"%.2f"%area, (int(mu[0]),int(mu[1])), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0,0,255))
            refPt=[]
            drawing=False
            print(area)
            
            
    cv2.destroyAllWindows()

