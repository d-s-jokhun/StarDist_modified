#!/usr/bin/env python
# coding: utf-8

# In[38]:


import numpy as np
import cv2
import math
# import os
import tifffile


# In[55]:


def Pad_n_Save (save_path, img, ImgSize=None):
    
    if ImgSize==0 or ImgSize==[] or ImgSize==[0,0] or ImgSize==(0,0) or ImgSize==None:
        ImgSize=(img.shape[1],img.shape[0])
    if type(ImgSize)==list:
        ImgSize=tuple(ImgSize)
    assert len(ImgSize)==2, 'ImgSize must be a tuple of 2 integers (W,H) or (0,0) to skip padding'
    assert type(ImgSize[0])==int, 'ImgSize must be a tuple of 2 integers (W,H) or (0,0) to skip padding'
    assert type(ImgSize[1])==int, 'ImgSize must be a tuple of 2 integers (W,H) or (0,0) to skip padding'
    
    W = ImgSize[0]
    H = ImgSize[1]
    W0 = img.shape[1]
    H0 = img.shape[0]

    if (W<W0 or H<H0):
        print (save_path,'was not saved because the original image size exceed the desired one!')
        return
    else:    
        width_to_pad=W-W0
        height_to_pad=H-H0
        width_start, width_end, height_start, height_end = (0,0,0,0)
        if width_to_pad>0:
            width_start = width_to_pad//2
            width_end = width_to_pad - width_start
        if height_to_pad>0:
            height_start = height_to_pad//2
            height_end = height_to_pad - height_start

        if len(img.shape) == 2:
            img = np.pad(img,((height_start,height_end),(width_start,width_end)))
        elif len(img.shape) == 3:
            img = np.pad(img,((height_start,height_end),(width_start,width_end),(0,0)))

        # cv2.imwrite(save_path,img)
        tifffile.imsave(save_path,img,photometric='minisblack', planarconfig='contig')

        # img_0 = img[:,:,0]
        # img_1 = img[:,:,1]
        # img_2 = img[:,:,2]

        # path_0 = os.path.basename(save_path).split('_')
        # path_0.insert(-1, 'Ch0')
        # path_0 = '_'.join(path_0)
        # save_path_0 = os.path.join(os.path.dirname(save_path),path_0)

        # path_1 = os.path.basename(save_path).split('_')
        # path_1.insert(-1, 'Ch1')
        # path_1 = '_'.join(path_1)
        # save_path_1 = os.path.join(os.path.dirname(save_path),path_1)

        # path_2 = os.path.basename(save_path).split('_')
        # path_2.insert(-1, 'Ch2')
        # path_2 = '_'.join(path_2)
        # save_path_2 = os.path.join(os.path.dirname(save_path),path_2)

        # cv2.imwrite(save_path_0,img_0)
        # cv2.imwrite(save_path_1,img_1)
        # cv2.imwrite(save_path_2,img_2)
       
    return


# In[ ]:





# In[ ]:




