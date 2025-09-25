# Asynchronous Multi-Object Tracking with an Event Camera

![alt text](./supplementary/thumbnail.jpg)

[Watch the supplementary video](./supplementary/AEMOT_supplementary_video_2025.mp4)

## For academic use only
Event cameras are ideal sensors for enabling robots to detect and track objects in highly dynamic environments due to their low latency output, high temporal resolution, and high dynamic range.
In this paper, we present the Asynchronous Event Multi-Object Tracking (AEMOT) algorithm for detecting and tracking multiple objects by processing individual raw events asynchronously.
AEMOT detects salient event blob features by identifying regions of consistent optical flow using a novel Field of Active Flow Directions built from the Surface of Active Events.
Detected features are tracked as candidate objects using the recently proposed Asynchronous Event Blob (AEB) tracker in order to construct small intensity patches of each candidate object.
A novel learnt validation stage promotes or discards candidate objects based on classification of their intensity patches, with promoted objects having their position, velocity, size, and orientation estimated at their event rate.
We evaluate AEMOT on a new Bee Swarm Dataset, where it tracks dozens of small bees with precision and recall performance exceeding that of alternative event-based detection and tracking algorithms by over 37%.


This paper was accepted by the 2025 IEEE International Conference on Robotics and Automation (ICRA).

Angus Apps, Ziwei Wang, Vladimir Perejogin, Timothy L. Molloy, and Robert Mahony

[[PDF](https://arxiv.org/abs/2505.08126)] [[IEEE Xplore](https://ieeexplore-ieee-org.virtual.anu.edu.au/abstract/document/11127984)]


## Citation
If you use or discuss our method, please cite our paper as follows:


<pre>
@inproceedings{apps2025asynchronous,
  title={Asynchronous Multi-Object Tracking with an Event Camera},
  author={Apps, Angus and Wang, Ziwei and Perejogin, Vladimir and Molloy, Timothy L and Mahony, Robert},
  booktitle={2025 IEEE International Conference on Robotics and Automation (ICRA)},
  pages={794--800},
  year={2025},
  organization={IEEE}
}
</pre>






## Code and Data

### Installation
Dependencies:


- [OpenCV](https://opencv.org/)
- [yaml-cpp](https://github.com/jbeder/yaml-cpp)
- [Eigen3](https://eigen.tuxfamily.org/index.php?title=Main_Page)
- [LibTorch](https://docs.pytorch.org/cppdocs/installing.html)

Note: You may have to specify the path to your OpenCV/yaml-cpp/Eigen/LibTorch library in `CMakeLists.txt`.


### Build
AEMOT is designed to be built as a cmake project. Assuming all prerequisites are installed and you are in the root folder of the repository, then you can follow these steps to build. 

```
    mkdir build
    cd build
    cmake -DCMAKE_BUILD_TYPE=Release ..
    cmake --build . -j8
```

<!-- ### Data
[Click here to download the labelled dataset.]()

The `Bee Swarm Dataset` is located `data/`, and is ready to use with the `bees.yaml` configuration file. -->

### Run  

```
    cd build
    ./aeb_tracker -i {config_file_name}
```
Choose the {config_file_name} from `config/` directory. E.g., `./aeb-tracker -i bees`