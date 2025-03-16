# OCR

OCR -> Optical Character Recognition

## Text types:
- digital/computer text
- hand written/print text

## Steps:
- Detect car number plates (as an example)
- Extract text (number) from number plate

## Types of OCR (trained on data - numbers, digit, characters, symbols, etc.)
- easy ocr (open source)
- paddle ocr (open source)
- amazon ocr (paid)
- google ocr (patd)
- azure ocr (paid)


## [Paddle OCR](https://github.com/PaddlePaddle/PaddleOCR)
- PaddleOCR supports Chinese, English, French, German, Korean, and Japanese language. We can set the parameter 'lang' as 'ch'/'en'/'fr'/'german'/'korean'/'japan' to switch the language model in order.
- open google colab
- install all requirements and configure setup
    - !python3 -m pip install paddlepaddle-gpu
    - !pip install "paddleocr>=2.0.1" (if it is asking restart runtime then do that and then follow next cell)
    - Find ubuntu openssl from [here](http://nz2.archive.ubuntu.com/ubuntu/pool/main/o/openssl/?C=M;O=D)
    - Right-click and do 'Copy link address' of the selected openssl file (see below example of this link)
    - http://nz2.archive.ubuntu.com/ubuntu/pool/main/o/openssl/libssl1.1_1.1.1f-1ubuntu2.22_amd64.deb
    - Run this two lines: 
        - !wget http://nz2.archive.ubuntu.com/ubuntu/pool/main/o/openssl/libssl1.1_1.1.1f-1ubuntu2.22_amd64.deb
        - !sudo dpkg -i libssl1.1_1.1.1f-1ubuntu2.22_amd64.deb
- code to execute
<pre>
from paddleocr import PaddleOCR, draw_ocr
# set the language model and ocr object
ocr = PaddleOCR(use_angle_cls=True, lang='en') # this need to run only once to download 

import cv2

from google.colab.patches import cv2_imshow

img_path = '/content/sample_restaurant_bill.jpg'

img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
cv2_imshow(img)

print(img)

# run the ocr method on the ocr model

result = ocr.ocr(img_path)

print(result)

print(len(result[0]))

for res in result:
  print(res[0][1]) 

# see the result one after one 
# it will display extracted test and its confidence score

</pre>


## EasyOCR
- !pip install easyocr
- code to execute
<pre>
import matplotlib.pyplot as plt
import cv2
import easyocr

from IPython.display import Image

Image("/content/sample_number_plate.jpg")

# create ocr object and set language
reader = easyocr.Reader(['en'])

output = reader.readtext("/content/sample_number_plate.jpg")

print(output)

print(output[0][-2])

</pre>


## Utilizing Object Dectection Model and OCR to Extract Text from Objects
- Download haarcascades model from [here](https://github.com/spmallick/mallick_cascades/tree/master/haarcascades)
- We can use any other object detection model (say YOLO, SSD, etc.) instead of haarcascades for object detection
- Create a folder called 'model' and keep the downloaded model inside the folder
- Create another folder called 'plates'
- Open system camera/CCTV camera
- Run this script: python number_plate.py 
- The above script can be found from [here](https://github.com/entbappy/Car-Number-Plates-Detection)
- With this script we will first detect object (say number plate from video) and then from the detected object it will extract text (say car number)
- Once code is running and it is able to access camera then press 'S' in keyboad to save images detected by the object detection model from video camera
- After that OCR model will extract text from saved object (image)