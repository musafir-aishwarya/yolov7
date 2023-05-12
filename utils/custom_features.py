def crop_bbox(x, img):
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cropped_img = img[c1[1]:c2[1], c1[0]:c2[0]]
    return cropped_img
