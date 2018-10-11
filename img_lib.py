# Image processing helper functions
import numpy as np
import cv2
import matplotlib.pyplot as plt

def list_images(images, xaxis = "", yaxis ="", cmap=None, ncols = 10, title = ''):
    """
    Display a list of images in a single figure with matplotlib.
        Parameters:
            images: An np.array compatible with plt.imshow.
            yaxis (Default = " "): A string to be used as a label for each image.
            cmap (Default = None): Used to display gray images.
    """
    plt.figure(figsize=(10, 10))    
    n_images = len(images)
    nrows = np.ceil(n_images/ncols) 

    for i in range(n_images):
        plt.subplot(nrows, ncols, i+1)
        #Use gray scale color map if there is only one channel
        cmap = 'gray' if images[i].shape[-1] != 3 else cmap
        plt.imshow(np.squeeze(images[i]), cmap = cmap)
        plt.xlabel(xaxis)
        plt.ylabel(yaxis)
        plt.xticks([])
        plt.yticks([])    
    plt.tight_layout(pad=0, h_pad=0, w_pad=0)    
    plt.title(title, loc = 'center', fontsize=16)        
    plt.show()    

def rotate_image(image, angle_range = 30):
    rows,cols,ch = image.shape
    random_angle = np.random.uniform(angle_range)-angle_range/2
    rot_mat = cv2.getRotationMatrix2D((cols/2,rows/2), random_angle, scale = 1)
    return cv2.warpAffine(image,rot_mat,(cols,rows))

# 20
def translate_image(image, trans_range = 5):
    rows,cols,ch = image.shape
    tr_x = trans_range*np.random.uniform()-trans_range/2
    tr_y = trans_range*np.random.uniform()-trans_range/2
    Trans_M = np.float32([[1,0,tr_x],[0,1,tr_y]])
    return cv2.warpAffine(image,Trans_M,(cols,rows))

def shear_image(image, shear_range = 5):
    rows,cols,ch = image.shape
    pts1 = np.float32([[5,5],[20,5],[5,20]])
    pt1 = 5+shear_range*np.random.uniform()-shear_range/2
    pt2 = 20+shear_range*np.random.uniform()-shear_range/2
    pts2 = np.float32([[pt1,5],[pt2,pt1],[5,pt2]])
    shear_M = cv2.getAffineTransform(pts1,pts2)
    return cv2.warpAffine(image,shear_M,(cols,rows))

def gray_scale_3channel_image(image):  
    gray_scale_image = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
    image[..., 0]    = gray_scale_image
    image[..., 1]    = change_brightness_image(gray_scale_image, isGrayScale = True)
    image[..., 2]    = change_brightness_image(gray_scale_image, isGrayScale = True)
    return image
    
    
# 0.5, 1.5
def change_brightness_image(image, brightness_range = 1.5, isGrayScale = False):  
    random_bright = 1 + brightness_range*np.random.uniform() - brightness_range/2 
    if not isGrayScale:
        image = cv2.cvtColor(image,cv2.COLOR_RGB2YCrCb).astype('float64')
        image[:,:,0] = np.ceil(image[:,:,0]*random_bright)
        image[:,:,0][image[:,:,0] > 255]  = 255
        image = image.astype('uint8')
        image = cv2.cvtColor(image,cv2.COLOR_YCrCb2RGB)
        return image
    else:    
        image = image.astype('float64')
        image = np.ceil(image*random_bright)
        image[image > 255]  = 255
        image = image.astype('uint8')
        return image        

def motion_blur_image(image, size = 3):    
    # generating the kernel
    kernel_motion_blur = np.zeros((size, size))
    kernel_motion_blur[int((size-1)/2), :] = np.ones(size)
    kernel_motion_blur = kernel_motion_blur / size
    return cv2.filter2D(image, -1, kernel_motion_blur)
    

def histogram_equalize_image(image, gridSize = (4, 4)):
    clahe = cv2.createCLAHE(tileGridSize=gridSize)
    img_YCrCb = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)            
    img_YCrCb[...,0] = clahe.apply(img_YCrCb[...,0])            
    image = cv2.cvtColor(img_YCrCb, cv2.COLOR_YCrCb2RGB)
    return image

def normalize_image(image):
        # pixels after normalization are in range [-0.5, +0.5]
        return (image - np.min(image))/(np.max(image)-np.min(image)) - 0.5
