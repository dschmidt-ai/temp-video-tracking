# MCMOT-ByteTrack: One-shot multi-class multi-object tracking </br>

## Operating Instructions

1. Clone repo
2. pip install requirements.txt file
3. Download Pretrained models from Google Drive (https://drive.google.com/file/d/1mOTHstEMQmHMF7BS4mGvTWPVZEpvMmxc/view?usp=share_link), and extract the folder under the main project directory
4. The main tracking script is ``` ../tools/tracking_main.py ```
5. Set up the following run config and run through the IDE or simply call the tracking script throught the terminal (feel free to play around with the parameters, they should all have appropriate default settings):

``` 
--demo video --tracker byte --exp_file ../exps/example/mot/yolox_tiny_det.py --n_classes=80
--class_names="person,bicycle,car,motorcycle,airplane,bus,train,truck,boat,traffic light,fire hydrant,stop sign,parking meter,bench,bird,cat,dog,horse,sheep,cow,elephant,bear,zebra,giraffe,backpack,umbrella,handbag,tie,suitcase,frisbee,skis,snowboard,sports ball,kite,baseball bat,baseball glove,skateboard,surfboard,tennis racket,bottle,wine glass,cup,fork,knife,spoon,bowl,banana,apple,sandwich,orange,broccoli,carrot,hot dog,pizza,donut,cake,chair,couch,potted plant,bed,dining table,toilet,tv,laptop,mouse,remote,keyboard,cell phone,microwave,oven,toaster,sink,refrigerator,book,clock,vase,scissors,teddy bear,hair drier,toothbrush" --class_names_filter="person,bicycle,car"
--ckpt ../pretrained/yolox_tiny_32.8.pth --path ../input/347109422.mp4
--device cpu --iou_thresh 0.3 --track_thresh 0.5 --match_thresh 0.8 --min-box-area 10 --nms 0.4
```


## The rest is the Repo's original documentation


</br>
This is an extention work of ByteTrack, which extends the one-class multi-object tracking to multi-class multi-object tracking
</br>
You can refer to origin fork [ByteTrack](https://github.com/ifzhang/ByteTrack)
and the original fork of [OC_SORT](https://github.com/noahcao/OC_SORT)
## Tracking demo of C5(car, bicycle, person, cyclist, tricycle)
![image](https://github.com/CaptainEven/MCMOT-ByteTrack/blob/master/test_13.gif)

## update news! 2022/05/18 To choose the wanted backend(byte | oc). 
Add OC_SORT as tracker's backend.
```
    parser.add_argument("--tracker",
                        type=str,
                        default="byte",
                        help="byte | oc")
```

## update news! 2021/12/01 TensorRT deployment updated! (Compile, release, debug with Visual Studio On Windows)
[TensorRT Deployment](https://github.com/CaptainEven/ByteTrack-MCMOT-TensorRT)

## How to Run the demo
Run the demo_mcmot.py python3 script for demo testing.

## Weights link
[checkpoint](https://pan.baidu.com/s/1PJc09vWK6UJEXp80y27b5g?pwd=ckpt)
### Weights extract code
ckpt

## Test video link
[Test Video for demo](https://pan.baidu.com/s/1RhT7UVtYK_3qiCg36GTb8Q?pwd=test)
### video extract code
test

## FairMOT's implemention of MCMOT: based on CenterNet
[MCMOT](https://github.com/CaptainEven/MCMOT)
