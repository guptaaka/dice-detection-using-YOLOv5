"""
Explode the number of images in dice dataset. The directory at
/content/yolov5/dicedataset/export/ contains two sub-directories: images and
labels. Images contains the jpg files of different dice images (These are
from the dice dataset at Roboflow, as referred to in the other files for
this project). Labels contains the txt files with information about labels
found in the corresponding image with same name as the label file and the 
bounding box of each of those labels in the image.

This script explodes the dataset by rotating the boxes that contain picture of
dice in all the images. In an image, the bounding box of each dice is rotated
one at a time at different angles of regular interval, starting at 5 degree,
until 355 degree at an interval of 5 degree.
"""
def main():
    images_fnames = os.listdir('/content/yolov5/dicedataset/export/images')
    for fname in images_fnames:
        if '.rf.' not in fname:
            # Each image in the dataset is repeated twice, one is named with
            # ".rf." substring in it, while the other does not have this
            # substring in its name, so work on the one that contains the
            # substring, hence skipping its duplicate.
            continue
        path = os.path.join('/content/yolov5/dicedataset/export/images', fname)
        original_img = Image.open(path)
        H, W = (1008, 756) # Dimension of the original image.
        basename = '.'.join(path.split('/')[-1].split('.')[:-1])
        txt_path = os.path.join('/content/yolov5/dicedataset/export/labels', basename+'.txt')
        label_lines = open(txt_path, 'r').readlines()
        for angle in range(5, 360, 5):
            img = original_img.copy()
            txtdata = ''
            new_pts = []
            for line in label_lines:
                # Read the bounding box information of a label.
                dice_class, ox, oy, ow, oh = (float(num) for num in line.split(' '))
                # The line contains each value rescaled according the bounding box dimensions.
                # Find the actual lengths.
                x, y, w, h = (math.floor(ox * W), math.floor(oy * H), math.floor(ow * W), math.floor(oh * H))

                # Take minimum of width and height, so that the cropped dice is cut in a square piece.
                # We do so, to keep the dice within bounds on rotation, this helps in making a clear image
                # and does not hinder with the background image. 
                w, h = (min(w,h), min(w,h))
                # Rescale x and y positions according to the new width and height.
                x1, y1, x2, y2 = (x-(w//2), y-(h//2), x+(w//2), y+(h//2))
                newx, newy, newW, newH = (((x1+x2)/2)/W, ((y1+y2)/2)/H, (x2-x1)/W, (y2-y1)/H)

                new_pts.append((dice_class, newx, newy, newW, newH))

                # Rotate the dice subimage.
                sub_img = img.crop(box=(x1, y1, x2, y2)).rotate(angle)
                img.paste(sub_img, box=(x1, y1))

        img = img.resize((640, 640)) # New image size is smaller as the algorithm works better with this size.

        for dice_class, newx, newy, newW, newH in new_pts:
            txtdata += '%d %f %f %f %f\n' % (dice_class, newx, newy, newW, newH)


        img.save(os.path.join('/content/yolov5/dicedataset/morphed_data/images', '%s_%d.jpg' % (basename, angle)))
        open(os.path.join('/content/yolov5/dicedataset/morphed_data/labels', '%s_%d.txt' % (basename, angle)), 'w').write(txtdata.strip())


if __name__ == '__main__':
    main()
