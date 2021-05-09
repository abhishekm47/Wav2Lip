import cv2

def crop_face_from_frame_with_bbox(frame, bbox):

    increase_area = 0.10
    top, bot, left, right  = bbox
    width = right - left
    height = bot - top
    frame_shape = frame.shape

    
    #Computing aspect preserving bbox
    width_increase = max(increase_area, ((1 + 2 * increase_area) * height - width) / (2 * width))
    height_increase = max(increase_area, ((1 + 2 * increase_area) * width - height) / (2 * height))
    
    left = int(left - width_increase * width)
    top = int(top - height_increase * height)
    right = int(right + width_increase * width)
    bot = int(bot + height_increase * height)
    h, w = bot - top, right - left
    
    top2, bot2, left2, right2 = max(0, top), min(bot, frame_shape[0]), max(0, left), min(right, frame_shape[1])
    crop_img = frame[top2:bot2, left2:right2]
    
    top_border = abs(top2 - top)
    bot_border = abs(bot2 - bot)
    left_border = abs(left2 - left)
    right_border = abs(right2 - right)
    
    crop_img = cv2.copyMakeBorder(crop_img, top=top_border, bottom=bot_border, left=left_border, right=right_border, borderType= cv2.BORDER_CONSTANT, value=[0,0,0] )
    crop_img = cv2.resize(crop_img, (512, 512))
    
    #crop_img = cv2.flip( crop_img, 0 ) 
    crop_img = cv2.flip( crop_img, 1 )
    
    return crop_img