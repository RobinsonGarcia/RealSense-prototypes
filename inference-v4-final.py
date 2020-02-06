#!/usr/bin/env python
# coding: utf-8

# In[ ]:
import base64
import json
import tqdm
from PIL import Image
#import matplotlib.pyplot as plt
import numpy as np
#import h5py
import os
import open3d as o3d
import pyrealsense2 as rs
import tensorflow as tf
import time
import cv2
# In[ ]:


#!apt install libusb-1.0
#!pip install open3d-python
#!apt install libgl1-mesa-glx -y


# # Path to videos
#videos/20191018_102048.bag
#75,videos/20191018_101803.bag
#videos/20191018_101621.bag
#"videos/20191018_103123.bag"
# In[ ]:

ENCODING = 'utf-8'
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
    #plt.imshow(images)
    #plt.show()

    # keep depth_map array
    depth_map = np.asanyarray(aligned_depth_frame.get_data())

    H,W,C = color.shape

    strings = {'H':H,'W':W,'C':C,'color':base64.b64encode(color).decode(ENCODING),\
        'depthmap':base64.b64encode(depth_map).decode(ENCODING),\
        'colorized_depth':base64.b64encode(colorized_depth).decode(ENCODING)}

    return color,colorized_depth,depth_map,profile,strings


# # Segmentation

# In[ ]:



config = tf.ConfigProto(
        device_count = {'GPU': 0}
    )

config.gpu_options.allow_growth = True
_FROZEN_GRAPH_PATH = os.path.join('cormodel', 'frozen_graph','frozen_inference_graph.pb')

"""Deeplab Model"""
class DeepLabModel(object):
    """Class to load deeplab model and run inference."""

    INPUT_TENSOR_NAME = 'ImageTensor:0'
    OUTPUT_TENSOR_NAME = 'ResizeBilinear_2:0' #'SemanticPredictions:0'#
    INPUT_SIZE = 513

    def __init__(self, model_dir=_FROZEN_GRAPH_PATH):
        """Creates and loads pretrained deeplab model."""
        self.graph = tf.Graph()

        graph_def = None
        # Extract frozen graph from tar archive.
        model_filename = model_dir
        with tf.gfile.FastGFile(model_filename, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())

        if graph_def is None:
            raise RuntimeError('Cannot find inference graph in tar archive.')

        with self.graph.as_default():
            tf.import_graph_def(graph_def, name='')
        self.sess = tf.Session(graph=self.graph,config=config)

    
        
    def run(self, image):
            """Runs inference on a single image.

            Args:
                image: A PIL.Image object, raw input image.

            Returns:
                resized_image: RGB image resized from original input image.
                seg_map: Segmentation map of `resized_image`.
            """
            width, height = image.size
            resize_ratio = 1.0 #* self.INPUT_SIZE / max(width, height)
            target_size = self.INPUT_SIZE,self.INPUT_SIZE#(int(resize_ratio * width), int(resize_ratio * height))
            resized_image = image.convert('RGB').resize(target_size, Image.ANTIALIAS)
            #print('Image resized')
            start_time = time.time()
            batch_seg_map = self.sess.run(
                self.OUTPUT_TENSOR_NAME,
                feed_dict={self.INPUT_TENSOR_NAME: [np.asarray(resized_image)]})
            #print('Image processing finished')
            #print('Elapsed time : ' + str(time.time() - start_time))
            seg_map = batch_seg_map[0]
            return np.array(resized_image), seg_map


# In[ ]:


"""CRF refinement"""
import pydensecrf.densecrf as dcrf

def _apply_crf(im,un,sxy=80,srgb=30,compat=1):
    W,H,n_classes=un.shape
    d = dcrf.DenseCRF2D(H,W,n_classes)
    U = un.transpose(2,0,1).reshape((n_classes,-1))
    U = U.copy(order='C')
    im = im.copy(order='C')
    d.setUnaryEnergy(U)
    d.addPairwiseBilateral(sxy,srgb,im,compat)
    Q=d.inference(5)
    #print("KL-divergance: {}".format(d.klDivergence(Q)))
    _map = np.argmax(Q,axis=0)
    proba = np.array(Q)
    _map=_map.reshape((W,H))

    return _map


# In[ ]:


def segmentation(model,color,CRF):
    img = Image.fromarray(color)

    im,un = model.run(img)

    mask = np.argmax(un,2)

    #plt.title("w/out CRF")
    #plt.imshow(np.hstack([mask*255,np.mean(im,2)]),cmap="gray")
    #plt.axis("off")
    #plt.show()

    #mask = Image.fromarray(255*mask.astype(np.uint8))
    #mask = mask.resize(img.size)
    if CRF:
        _mask = 1-_apply_crf(im,un)

        #plt.title("w/CRF")
        #plt.imshow(np.hstack([_mask*255,np.mean(im,2)]),cmap="gray")
        #plt.axis("off")
        #plt.show()

        _mask = Image.fromarray(255*_mask.astype(np.uint8))
        mask = _mask.resize(img.size)
    return mask


# # Compute 3D coordinates

# In[ ]:


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

# In[ ]:


def build_data3(p):
    N=p.shape[0]
    dataM=[p[:,0],p[:,1],    p[:,2]]
    return np.array(dataM).T


# In[ ]:


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

# In[ ]:


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
        #plt.title(str(area) + " m2")

        #plt.imshow(color*(np.array(mask)[:,:,np.newaxis]/255).astype(np.uint8))
        #plt.imshow(color)
        #plt.scatter(h,w,c='b',alpha=0.1,s=1)
        #plt.show()

    print("Total area: {}m2".format(total_area))
    return total_area


# # Save point cloud

# In[ ]:


SKIP_FRAMES = 10
PATH_VIDEOS = 'videos'
videos = os.listdir(PATH_VIDEOS)

config = tf.ConfigProto(
        device_count = {'GPU': 0}
    )

config.gpu_options.allow_growth = True
_FROZEN_GRAPH_PATH = os.path.join('cormodel', 'frozen_graph','frozen_inference_graph.pb')

CRF=True

model = DeepLabModel()

for t in tqdm.tqdm(range(len(videos))):
    try:
        VIDEO_FILE = "videos/"+videos[t]

        color,colorized_depth,depth_map,profile,strings=loadRSdata(VIDEO_FILE,SKIP_FRAMES)

        mask = segmentation(model,color,CRF)

        todos_pts,pts,todos_rgb,rgb = computeXYZ(profile,mask,depth_map,color)

        """Show depth map"""
        #plt.imshow(depth_map*mask-(depth_map*mask).max())

        """Show masked corrosion"""
        #plt.imshow(color*(np.array(mask)[:,:,np.newaxis]/255).astype(np.uint8))
        masked_corrosion = base64.b64encode(\
            color*(np.array(mask)[:,:,np.newaxis]/255).astype(np.uint8)).decode(ENCODING)
        strings['masked_corrosion']=masked_corrosion

        pts2D,objs,pts,_h,_w= RobustPlaneFittingRANSAC(mask,pts)
        strings['pts']={'shape':pts.shape,'pts':base64.b64encode(pts).decode(ENCODING),\
            '_h':base64.b64encode(_h).decode(ENCODING),\
            '_w':base64.b64encode(_h).decode(ENCODING)}

        area = LSfitting(pts2D,objs,pts,_h,_w)

        with open(videos[t].split(".")[0]+".json",'w',encoding='utf-8') as f:
            f.write(json.dumps(strings))

        # Pass xyz to Open3D.o3d.geometry.PointCloud and visualize
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np.vstack(objs))
        #pcd.colors = o3d.utility.Vector3dVector(rgb[inliers,:])
        o3d.io.write_point_cloud(videos[t]+"_pc.ply", pcd)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(todos_pts)
        pcd.colors = o3d.utility.Vector3dVector(todos_rgb)
        o3d.io.write_point_cloud(videos[t]+"_total_pc.ply", pcd)
    except Exception as e:
        print(e)
        continue