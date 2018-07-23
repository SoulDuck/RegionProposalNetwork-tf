import os
from scipy.misc import imread
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
def next_img_gtboxes(image_idx):
    IMAGE_FORMAT= '.png'
    data_dir='../faster_rcnn/clutteredMNIST'
    train_name_path = os.path.join(data_dir, 'Names', 'train.txt')
    train_names = [line.rstrip() for line in open(train_name_path, 'r')]
    if image_idx > (len(train_names)-1) :
        image_idx= image_idx % (len(train_names)-1)


    img_path = os.path.join(data_dir, 'Images', train_names[image_idx] + IMAGE_FORMAT)
    annotation_path = os.path.join(data_dir, 'Annotations', train_names[image_idx] + '.txt')
    img = imread(img_path)

    gt_bbox = np.loadtxt(annotation_path, ndmin=2)
    #im_dims = np.array(img.shape[:2]).reshape([1, 2])

    flips = [0, 0]
    flips[0] = np.random.binomial(1, 0.5)
    #img = image_preprocessing.image_preprocessing(img)
    if np.max(img) > 1:
        img = img / 255.
    return img , gt_bbox


def draw_rectangles(img ,bboxes , savepath):
    ax = plt.axes()
    plt.imshow(img)
    h,w=np.shape(img)
    for box in bboxes:

        x1, y1, x2, y2= box  # x1 ,y1 ,x2 ,y2
        if x1 >0 and y1 >0 and x2 > 0 and y2 > 0 and x2 > x1 and y2 > y1 and w > x2 and y2 < h :
            print x1, y1, x2, y2
            print np.shape(img)
            rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
        else:
            continue
    plt.savefig(savepath)
    plt.close()




if '__main__' == __name__:
    img , gt_boxes =next_img_gtboxes(image_idx=1)
    ax=plt.axes()
    for box in gt_boxes:
        x1, y1, x2, y2, label = box  # x1 ,y1 ,x2 ,y2
        rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
    plt.imshow(img)
    plt.show()


