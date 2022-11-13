from os import environ
environ["OPENCV_IO_ENABLE_OPENEXR"] = "true"

from envmap.environmentmap import EnvironmentMap
import pickle
import cv2
import numpy as np
import matplotlib.pyplot as plt

#img_path = "/Volumes/LaCie/indoorDatasetCalibData/Photometric/Week_19_Sep_2022/14_1#30/2022_09_21_11_26_33/9C4A7281 Panorama_hdr.exr"
img_path = "/Users/christophebolduc/Documents/2022_09_21_11_26_33/9C4A7281 Panorama_hdr.exr"
img_path_correct = "/Users/christophebolduc/Documents/2022_09_21_11_26_33/9C4A7281 Panorama_hdr_correct.exr"

with open("calib5D.pkl", 'rb') as f:
    k, d = pickle.load(f)

#e = EnvironmentMap(img_path, 'fisheye', k=k,d=d,xi=1)
#e_sa = e.convertTo('latlong')
#numpy = e_sa.data.astype(np.float32)
#e_sa = e.solidAngles()

def computeIllumiance(img_path):

    e = EnvironmentMap(img_path, 'fisheye', k=k,d=d,xi=1.0)
    e_sa = e.solidAngles()

    vectors = e.worldCoordinates()
    vectors_x = vectors[0]
    vectors_y = vectors[1]
    vectors_z = vectors[2]
    n = np.array([0,0,-1]) #Front dir -Z

    #Stack 3 direction vectors
    vectors_top = np.stack([vectors_x,vectors_y,vectors_z], axis=-1)

    #Compute cos angle between front dir and each pixel
    vectors_norm = np.linalg.norm(vectors_top, axis = 2)
    vectors_dot = np.dot(vectors_top, n)
    cos = vectors_dot/vectors_norm

    mask = np.where(cos >= 0, 1, 0)
    mask = np.where(np.isnan(e_sa), 0, 1)
    plt.imshow(mask)
    e_sa = np.nan_to_num(e_sa)

    #Compute illuminance
    illumiance_r = np.sum(e_sa * cos * e.data[:,:,0] * mask)
    illumiance_g = np.sum(e_sa * cos * e.data[:,:,1] * mask)
    illumiance_b = np.sum(e_sa * cos * e.data[:,:,2] * mask)
    print(illumiance_r, illumiance_g, illumiance_b)

    plt.colorbar()
    plt.show()

def compareMultCosProjectCos(img_path, img_path_correct):
    e = EnvironmentMap(img_path, 'fisheye', k=k,d=d,xi=1.0)
    e_sa = e.solidAngles()

    vectors = e.worldCoordinates()
    vectors_x = vectors[0]
    vectors_y = vectors[1]
    vectors_z = vectors[2]
    n = np.array([0,0,-1]) #Front dir -Z

    #Stack 3 direction vectors
    vectors_top = np.stack([vectors_x,vectors_y,vectors_z], axis=-1)

    #Compute cos angle between front dir and each pixel
    vectors_norm = np.linalg.norm(vectors_top, axis = 2)
    vectors_dot = np.dot(vectors_top, n)
    cos = vectors_dot/vectors_norm

    mask = np.where(cos >= 0, 1, 0)
    mask = np.where(np.isnan(e_sa), 0, 1)
    
    mean_cos_r = np.sum(e.data[:,:,0] * cos * mask)/np.sum(mask)
    mean_cos_g = np.sum(e.data[:,:,1] * cos * mask)/np.sum(mask)
    mean_cos_b = np.sum(e.data[:,:,2] * cos * mask)/np.sum(mask)

    print("time cos: ", mean_cos_r, mean_cos_g, mean_cos_b)

    img = cv2.imread(img_path_correct)
    mask = np.zeros((img.shape[0],img.shape[1]))
    mask = cv2.circle(mask, (mask.shape[0]//2,mask.shape[1]//2), mask.shape[0]//2, 1, -1)
    mask = mask.astype(np.uint8)

    mean_cos_r = np.sum(img[:,:,0] * mask)/np.sum(mask)
    mean_cos_g = np.sum(img[:,:,1] * mask)/np.sum(mask)
    mean_cos_b = np.sum(img[:,:,2] * mask)/np.sum(mask)

    print("cosine correct: ", mean_cos_r, mean_cos_g, mean_cos_b)

#computeIllumiance(img_path)
compareMultCosProjectCos(img_path,img_path_correct)