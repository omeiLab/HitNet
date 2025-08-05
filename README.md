# HitNet
A sub-module of my CS project, the AI badminton end-to-end system.
The HitNet are designed to detect **if a frame has a hit event** based on ball trajectory and player keypoints for badminton matches. 

## Data Preparation

Before get started, please prepare a match folder in your repo, we assume that your folder has the name `matches` and has the at least following structure:
```
/matches
│── /match1               
│   │── /TrackNet
│   │   │── clip_1.csv                       # trajectory
│   │   │── clip_2.csv
│   │   │── ...
│   │── /poses
│   │   │── clip_1_bottom.csv                # pose of bottom player 
│   │   │── clip_1_top.csv                   # pose of top player
│   │   │── clip_2_bottom.csv               
│   │   │── clip_2_top.csv
│   │   │── ...
│   │── /hits
│   │   │── clip_1_hit.csv                   # ground truth
│   │   │── clip_2.csv
│   │   │── ...
│── /match3 ...                   
```
Each match is referred to a **set** and clips are the rallies. We use the following to generate these data:
- Trajectory: see [TrackNetV3 with attention](https://github.com/alenzenx/TrackNetV3)

- Pose: see [MMPose](https://github.com/open-mmlab/mmpose/tree/main)

- hits: see [ShuttleSet](https://github.com/wywyWang/CoachAI-Projects/tree/main/ShuttleSet)

However, it can be very time-consuming. So we provide some `.npy` files in the packages which are the processed data for quick start. The data including 30 matches for training and 4 matches for validating.

## Train Model

Simply run `main.py`.

### Metrics

There is unavoidable **highly imbalance** in the data, approximately 35:1 for label 0 (no hit) vs 1 (hit). Hence, the metrics we apply are **recall** and **F1-score**, focusing on recall.

### Result

We achieve the following performance
```
Best threshold = 0.56, Recall = 0.8817
          Val Loss: 0.2796 |   Val Recall: 0.8817 |   Val F1: 0.7530

              precision    recall  f1-score   support

           0       0.99      0.90      0.94     26066
           1       0.42      0.86      0.56      2194

    accuracy                           0.90     28260
   macro avg       0.70      0.88      0.75     28260
weighted avg       0.94      0.90      0.91     28260

[[0.8992941  0.1007059 ]
 [0.13582498 0.86417502]]
```

## Licence Notice

This project includes components adapted from the [monotrack](https://github.com/jhwang7628/monotrack) repository by Adobe, under the Adobe Research License.  
The usage is limited to non-commercial academic research, including undergraduate coursework.
> Copyright © <YEAR>, Adobe Inc. and its licensors. All rights reserved.
> ADOBE RESEARCH LICENSE 
> Adobe grants any person or entity ("you" or "your") obtaining a copy of these certain research materials that are owned by Adobe ("Licensed Materials") a nonexclusive, worldwide, royalty-free, revocable, fully paid license to (A) reproduce, use, modify, and publicly display the Licensed Materials; and (B) redistribute the Licensed Materials, and modifications or derivative works thereof, provided the following conditions are met:
> - The rights granted herein may be exercised for noncommercial research purposes (i.e., academic research and teaching) only. Noncommercial research purposes do not include commercial licensing or distribution, development of commercial products, or any other > activity that results in commercial gain.
> - You may add your own copyright statement to your modifications and/or provide additional or different license terms for use, reproduction, modification, public display, and redistribution of your modifications and derivative works, provided that such license terms limit the use, reproduction, modification, public display, and redistribution of such modifications and derivative works to noncommercial research purposes only.
> - You acknowledge that Adobe and its licensors own all right, title, and interest in the Licensed Materials.
> - All copies of the Licensed Materials must include the above copyright notice, this list of conditions, and the disclaimer below.
>   
> Failure to meet any of the above conditions will automatically terminate the rights granted herein.
> 
> THE LICENSED MATERIALS ARE PROVIDED "AS IS" WITHOUT WARRANTY OF ANY KIND. THE ENTIRE RISK AS TO THE USE, RESULTS, AND PERFORMANCE OF THE LICENSED MATERIALS IS ASSUMED BY YOU. ADOBE DISCLAIMS ALL WARRANTIES, EXPRESS, IMPLIED OR STATUTORY, WITH REGARD TO YOUR USE OF THE LICENSED MATERIALS, INCLUDING, BUT NOT LIMITED TO, NONINFRINGEMENT OF THIRD-PARTY RIGHTS. IN NO EVENT WILL ADOBE BE LIABLE FOR ANY ACTUAL, INCIDENTAL, SPECIAL OR CONSEQUENTIAL DAMAGES, INCLUDING WITHOUT LIMITATION, LOSS OF PROFITS OR OTHER COMMERCIAL LOSS, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THE LICENSED MATERIALS, EVEN IF ADOBE HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGES.
