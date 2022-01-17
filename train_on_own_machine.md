Steps:

Step I. Pre-requisites

1. Download yolov5 repositories.
git clone https://github.com/ultralytics/yolov5
cd yolov5

2. Install required modules
pip install -qr requirements.txt

3. Download the dataset

You can either

* download the dice dataset from Roboflow. It has less number of data points, so the images dataset can be exploded by using the _multiply_dataset.py_ script. After this is done, you can divide the entire dataset into three buckets Train, Valid and Test. I used 70:20:10 split. Make sure to split the dataset smartly, in that each of the three directories should have a good number of images belonging to each class, and a good split also in terms of number of dice in an image and classes of dice in that image.

'''
mkdir -p dicedataset
cd dicedataset
curl -L "https://public.roboflow.com/ds/yJwlTSkNrQ?key=OiVqh5lSem" > roboflow.zip; unzip roboflow.zip; rm roboflow.zip
'''

* or, download the dice dataset from Google drive link by using the _download_dice_dataset.py_ script. This dataset has already been exploded into huge number of images and divided into 3 chunks for training, validation and testing in the 70:20:10 ratio, the data has been exploded by using the same logic as present in the _multiply_dataset.py_ script. Along with this dataset, you'll still need the metadata files fom the Roboflow data directly. So, make sure to downalod those files from the Roboflow link posted above.

python download_dice_dataset.py

Step II. Custom setup (run on a Jupyter notebook)
1. Read number of dataset classes
dataset_location = './dicedataset'
import yaml
with open(dataset_location + "/data.yaml", 'r') as stream:
    num_classes = str(yaml.safe_load(stream)['nc']) # 6

2. Customize iPython writefile so we can write variables
from IPython.core.magic import register_line_cell_magic
@register_line_cell_magic
def writetemplate(line, cell):
    with open(line, 'w') as f:
        f.write(cell.format(**globals()))

3. Define custom template in yolov5 directory
%%writetemplate ./models/custom_yolov5s.yaml

# parameters
nc: {num_classes}  # number of classes
depth_multiple: 0.33  # model depth multiple
width_multiple: 0.50  # layer channel multiple

# anchors
anchors:
  - [10,13, 16,30, 33,23]  # P3/8
  - [30,61, 62,45, 59,119]  # P4/16
  - [116,90, 156,198, 373,326]  # P5/32

# YOLOv5 backbone
backbone:
  # [from, number, module, args]
  [[-1, 1, Focus, [64, 3]],  # 0-P1/2
   [-1, 1, Conv, [128, 3, 2]],  # 1-P2/4
   [-1, 3, BottleneckCSP, [128]],
   [-1, 1, Conv, [256, 3, 2]],  # 3-P3/8
   [-1, 9, BottleneckCSP, [256]],
   [-1, 1, Conv, [512, 3, 2]],  # 5-P4/16
   [-1, 9, BottleneckCSP, [512]],
   [-1, 1, Conv, [1024, 3, 2]],  # 7-P5/32
   [-1, 1, SPP, [1024, [5, 9, 13]]],
   [-1, 3, BottleneckCSP, [1024, False]],  # 9
  ]

# YOLOv5 head
head:
  [[-1, 1, Conv, [512, 1, 1]],
   [-1, 1, nn.Upsample, [None, 2, 'nearest']],
   [[-1, 6], 1, Concat, [1]],  # cat backbone P4
   [-1, 3, BottleneckCSP, [512, False]],  # 13

   [-1, 1, Conv, [256, 1, 1]],
   [-1, 1, nn.Upsample, [None, 2, 'nearest']],
   [[-1, 4], 1, Concat, [1]],  # cat backbone P3
   [-1, 3, BottleneckCSP, [256, False]],  # 17 (P3/8-small)

   [-1, 1, Conv, [256, 3, 2]],
   [[-1, 14], 1, Concat, [1]],  # cat head P4
   [-1, 3, BottleneckCSP, [512, False]],  # 20 (P4/16-medium)

   [-1, 1, Conv, [512, 3, 2]],
   [[-1, 10], 1, Concat, [1]],  # cat head P5
   [-1, 3, BottleneckCSP, [1024, False]],  # 23 (P5/32-large)

   [[17, 20, 23], 1, Detect, [nc, anchors]],  # Detect(P3, P4, P5)
  ]

Step III.
python train.py --epochs 30 --data {dataset_location}/data.yaml --cfg ./models/custom_yolov5s.yaml --weights '' --name yolov5s_results

Step IV.
python detect.py --weights runs/train/yolov5s_results_initial/weights/best.pt --conf-thres 0.6 --source ../test/images --save-txt --save-conf
