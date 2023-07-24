# <center> 💡🧠🤔 Smart-Tree 🌳🌲🌴 </center>

## 📝 Description:

This GitHub repository contains code from the the paper "Smart-Tree: Neural Medial Axis Approximation of Point Clouds for 3D Tree Skeletonization". <br>
The code provided, is a deep-learning based skeletonization method for point clouds.

<table>
<tr>
  <td style="text-align: center"><img src="images/botanic-pcd.png", height=100%></td>
  <td style="text-align: center"><img src="images/botanic-branch-mesh.png", height=100%></td>
  <td style="text-align: center"><img src="images/botanic-skeleton.png", height=100%></td>
</tr>
<tr>
  <td align="center">Input point cloud.</td>
  <td align="center">Mesh output.</td>
  <td align="center">Skeleton output.</td>
</tr>
</table>


## 💾 Data:

Please download data from this <a href="https://github.com/uc-vision/synthetic-trees">link</a>. <br>

## 🔧 Installation:

First make sure you have conda installed.
To install smart-tree please use: <br>`bash create-env.sh`<br>
Then activate the enviroment using: <br>`conda activate smart-tree`


## 📈 Training:

To train the model open smart_tree/conf/tree-dataset.yaml.

You will need to update (alternatively these can be overwritten with hydra): 

- training.dataset.json_path to the location of where your smart_tree/conf/tree-split.json is stored. 
- training.dataset.directory to the location of where you downloaded the data (you can choose wether to train on the data with foliage or without based on the directory you supply).

You can experiement / adjust hyper-parameter settings too.

The model will then train using the following:

`train-smart-tree`

The best model weights and model will be stored in the generated outputs directory.

## ▶️ Inference / ☠️ Skeletonization:

We supply two different models with weights:
* `noble-elevator-58` contains branch / foliage segmentation. <br>
* `peach-forest-65` is only trained on points from branching structure. <br>

If you wish to run smart-tree using your own weights you will need to update the model paths in the `tree-dataset.yaml`. <br>

To run smart-tree use: <br>
`run-smart-tree +path=cloud_path` <br>
where `cloud_path` is the path of the point cloud you want to skeletonize. <br>
Skeletonization specific parameters can be adjusted within the `smart_tree/conf/tree-dataset.yaml` config.

## 📜 Citation:
Please use the following BibTeX entry to cite our work: <br>

```
@InProceedings{10.1007/978-3-031-36616-1_28,
author="Dobbs, Harry
and Batchelor, Oliver
and Green, Richard
and Atlas, James",
editor="Pertusa, Antonio
and Gallego, Antonio Javier
and S{\'a}nchez, Joan Andreu
and Domingues, In{\^e}s",
title="Smart-Tree: Neural Medial Axis Approximation of Point Clouds for 3D Tree Skeletonization",
booktitle="Pattern Recognition and Image Analysis",
year="2023",
publisher="Springer Nature Switzerland",
address="Cham",
pages="351--362",
abstract="This paper introduces Smart-Tree, a supervised method for approximating the medial axes of branch skeletons from a tree point cloud. Smart-Tree uses a sparse voxel convolutional neural network to extract the radius and direction towards the medial axis of each input point. A greedy algorithm performs robust skeletonization using the estimated medial axis. Our proposed method provides robustness to complex tree structures and improves fidelity when dealing with self-occlusions, complex geometry, touching branches, and varying point densities. We evaluate Smart-Tree using a multi-species synthetic tree dataset and perform qualitative analysis on a real-world tree point cloud. Our experimentation with synthetic and real-world datasets demonstrates the robustness of our approach over the current state-of-the-art method. The dataset (https://github.com/uc-vision/synthetic-trees) and source code (https://github.com/uc-vision/smart-tree) are publicly available.",
isbn="978-3-031-36616-1"
}

```

## 📥 Contact 

Should you have any questions, comments or suggestions please use the following contact details:
harry.dobbs@pg.canterbury.ac.nz
