# Sequential 3D Human Pose Estimation Using Adaptive Point Cloud Sampling  Strategy

This repository provides the python code of the following paper:
>Zhang Z, Hu L, Deng X, et al. Sequential 3D Human Pose Estimation Using Adaptive Point Cloud Sampling Strategy[C]//IJCAI. 2021: 1330-1337.

>Abstract :3D human pose estimation is a fundamental problem in artificial intelligence, and it has wide applications in AR/VR, HCI and robotics.  However, human pose estimation from point clouds still  
suffers from noisy points and estimated jittery artifacts because of handcrafted-based point cloud sampling and single-frame-based estimation strategies. In this paper, we present a new perspective on the 3D human pose estimation method from point cloud sequences. To sample effective point clouds from input, we design a differentiable point cloud sampling method built on density-guided attention mechanism.  To avoid the jitter caused by previous 3D human pose estimation problems, we adopt temporal information to obtain more stable results. Experiments on the ITOP dataset and the NTU-RGBD dataset demonstrate that all of our contributed components are effective, and our method can achieve state-of-the-art performance

## Prerequisites
**Dependencies**
-   Linux
-   python 3.7+
-   tensorflow 1.x
-   g++ 4.8 (other version might also work)

 **Installation**
    
   Please run the follow commands to  compile customized TF operators
    
    ```
    git clone https://github.com/...
    cd tf_ops
    bash compile_all.sh
    ```
The TF operators are included under  `tf_ops`, you need to compile them first. Update  `nvcc`  and  `python`  path if necessary. The code is tested under TF1.15. If you are using earlier version it's possible that you need to remove the  `-D_GLIBCXX_USE_CXX11_ABI=0`  flag in g++ command(check `tf_xxx_compile.sh` under each ops subfolder) in order to compile correctly.
    
-   **Downloads**
    
    You will need to download the following files to run our code:
    
    -   Download  [ITOP Dataset | Zenodo](https://zenodo.org/record/3932973#.Yp8SIxpBxPA) and unzip it to the  **./Dataset**  floder.
    - Download  [pre-training model](https://drive.google.com/file/d/1QyeGcpAej8paJP-cfow0NVRcTGIVKXGE/view?usp=sharing) and unzip it to the **./density_weights_model**  
  

## Usage

Please run the follow command to test the model. Modify  the parameters  in **run.sh** can change the model as test, choose the dataset path and the model save path. All parameters has been set in  **./py_train.py**.
```
bash run.sh
```

## Results

The joint mAPs for our model run are as follows

|body part       |mAPs          （%）                |joint error  （m）                  |
|----------------|-------------------------------|-----------------------------|
|Head|98.33|0.0229574|
|Neck|98.64|0.02504025|
|R Shoulder|92.35|0.03762034|
|L Shoulder|98.42|0.02874033|
|R Elbow|91.29|0.04294496|
|L Elbow|90.79|0.04221966|
|R Hand|82.58|0.05742592|
|L Hand|81.77|0.06297581|
|Torso|99.70| 0.02598399 |
|R Hip|97.27|0.03719915|
|L Hip|95.38|0.0410355  |
|R Knee|94.86|0.03453163|
|L Knee|93.97|0.03532967 |
|R Foot|93.81|0.04826031 |
|L Foot|92.04|0.04858931|



## Citation

If you use this code for your research, please cite our paper:
```
@inproceedings{zhang2021sequential,
  title={Sequential 3D Human Pose Estimation Using Adaptive Point Cloud Sampling Strategy.},
  author={Zhang, Zihao and Hu, Lei and Deng, Xiaoming and Xia, Shihong},
  booktitle={IJCAI},
  pages={1330--1337},
  year={2021}
}
```