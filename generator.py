# Image processing library functions
import numpy as np
from img_lib import list_images, rotate_image, translate_image, shear_image 
from img_lib import change_brightness_image, motion_blur_image, scale_image

def random_transform_image(image):    
    if np.random.randint(2) == 0:
        return image
    
    transformation_library = ['rotation','translation','shear','brightness','blur', 'scale']    
    transformation_id = transformation_library[np.random.randint(len(transformation_library))]
    
    if transformation_id == 'rotation':
        image = rotate_image(image)
        
    if transformation_id == 'translation':
        image = translate_image(image)
    
    if transformation_id == 'shear':
        image = shear_image(image)

    if transformation_id == 'brightness':
        image = change_brightness_image(image)
        
    if transformation_id == 'blur':
        image = motion_blur_image(image)        

    if transformation_id == 'scale':
        image = scale_image(image)
    
    return image

def data_augmentation(X_data):
    return np.array([random_transform_image(image) for image in X_data]).reshape(X_data.shape[0], 32,32,-1)
