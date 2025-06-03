# U-SLAM

This repository contains a work-in-progress underwater SLAM (Simultaneous Localization and Mapping) system for an Autonomous Underwater Vehicle (AUV), developed entirely using **Python** and **ROS2**.

---

## Project Motivation

Underwater navigation is notoriously difficult due to:
- Lack of GPS
- Sensor noise and poor visibility
- Challenging environments (murky water, reflections)

This project aims to solve these challenges by building a robust **SLAM pipeline** that combines multiple sensor modalities to **localize an AUV** and **generate a 3D map** of its underwater environment.

---

## Project Goal

- **ORB feature extraction from image/camera data**
- **Fuse visual, DVL, and SONAR data** to map unknown environments  
- **Localize the AUV in 3D space** using a **particle filter** approach  
- **Visualize the result in RViz2** using an AUV URDF; may move to Pangolin  
- **Fully ROS2-compliant Python implementation** (no C++ or cloning ORB-SLAM3)
- Final goal: Real-time underwater SLAM system for robotics/**defense applications**

---

## Current Status

This project is still **under active development**.  
The following modules are being built:

- Image-based feature tracking and mapping  
- Particle filter localization using DVL and visual input  
- SONAR loop closure using .xtf files  
- ROS2 package organization for modular design  
- RViz2-based visualization with URDF-based AUV model  

---

## Dataset Assumptions

The system expects:
- **Image data** (monocular or stereo)
- **DVL odometry** in CSV format
- **SONAR data** in `.xtf` format
- A **URDF model** of the AUV

---

## Disclaimer

This repository is an evolving research and engineering project.  
Expect:
- Partial/incomplete features
- Refactoring over time
- Documentation to grow alongside development

---

## Contact

For updates, collaborations, or questions, feel free to reach out.

**Author:** Sreejib Pal  
sreejib1945@gmail.com  

---

