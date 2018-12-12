# Stochastic-Guidance
 Learning with Stochastic Guidance for Navigation

By [Linhai Xie](https://www.cs.ox.ac.uk/people/linhai.xie/), Yishu Miao, [Sen Wang](http://senwang.weebly.com/), Phil Blunsom, Zhihua Wang, Changhao Chen,  Niki trigoni, Andrew Markham.

The tensorflow implmentation for the paper: [Learning with Stochastic Guidance for Navigation](https://arxiv.org/abs/1811.10756)

## Contents
0. [Introduction](#Introduction)
0. [Prerequisite](#Prerequisite)
0. [Instruction](#instruction)
0. [Citation](#citation)

## Introduction

In this project we proposed a stochastic switching machanism which is an improved version of our ICRA [paper](https://www.cs.ox.ac.uk/files/9953/Learning%20with%20Training%20Wheels.pdf).

The stochastic switching network is implemented with a fully connected network and trained with REINFORCE algorithm.

For details please see the [paper](https://arxiv.org/abs/1811.10756)

The implementation of DDPG is based on [Emami's work](https://pemami4911.github.io/blog/2016/08/21/ddpg-rl.html).


## Prerequisites

Tensorflow > 1.1

ROS Kinetic

ros stage

matplotlib

cv2

## Instruction

roscore

rosrun stage_ros stageros PATH TO THE FOLDER/AsDDPG/worlds/Obstacles.world

python DDPG_stochastic.py

You can download the prioritized replay buffer [here](https://drive.google.com/file/d/12rD2wjinSkqYSzZS7gJKXYjd7_7bMvmN/view?usp=sharing) or run the above training command once. It takes random actions or selections of controllers to initialize the replay buffer and create a pickle file.


## Citation

If you use this method in your research, please cite:

	@article{xie2018learning,
  		title={Learning with Stochastic Guidance for Navigation},
  		author={Xie, Linhai and Miao, Yishu and Wang, Sen and Blunsom, Phil and Wang, Zhihua and Chen, Changhao and Markham, Andrew and Trigoni, Niki},
  		journal={arXiv preprint arXiv:1811.10756},
  		year={2018}}



