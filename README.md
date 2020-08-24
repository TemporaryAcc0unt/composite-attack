This is repository for paper *Composite Backdoor Attack for Deep Neural Network by Mixing Existing Benign Features*



Dependences:
```
Python3
Pytorch
numpy
PIL
matplotlib
```



Currently, this version only works on the attacking CIFAR10, YouTubeFace and COCO with two trigger labels. Support for more attacks is coming soon.



Attack CIFAR10:
```
python3 attack_cifar.py
```



Attack YouTubeFace:

1. download weight file for VGGFace https://github.com/prlz77/vgg-face.pytorch
2. prepare dataset following `data/prepare_youtubeface.ipynb`
3. `python3 attack_youtubeface.py`



Attack COCO:

```
bash yolov3/data/get_coco2014.sh
python3 attack_coco.py train
python3 attack_coco.py test
cd yolov3
python3 train.py --data data/coco2014_train_attack.data --epochs 20
```
The yolov3 framework is [ultralytics/yolov3](https://github.com/ultralytics/yolov3) 

