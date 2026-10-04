<div align="center">

# U-SLAM

### Underwater SLAM

<p>
  <strong>ROS 2 · Gazebo · Underwater Robotics · Sensor Fusion · SLAM</strong>
</p>

<p>
  <a href="https://github.com/palsreejib/U-SLAM">
    <img src="https://img.shields.io/badge/ROS%202-Humble-22314E?style=for-the-badge&logo=ros" alt="ROS 2">
  </a>
  <a href="https://gazebosim.org/">
    <img src="https://img.shields.io/badge/Gazebo-Simulation-FF6F00?style=for-the-badge" alt="Gazebo">
  </a>
  <a href="https://github.com/palsreejib/U-SLAM">
    <img src="https://img.shields.io/github/repo-size/palsreejib/U-SLAM?style=for-the-badge" alt="Repository Size">
  </a>
</p>

<p>
  <em>
    A simulation and research framework for underwater perception,
    localization, mapping, and SLAM.
  </em>
</p>

</div>

---

U-SLAM is a research project for developing and evaluating **Simultaneous Localization and Mapping (SLAM)** approaches for underwater autonomous systems.

The project provides a ROS 2 and Gazebo-based simulation environment for integrating underwater vehicle dynamics, hydrographic sensors, acoustic perception, and ground-truth information. The framework is intended to support controlled experiments in underwater localization, sensor fusion, mapping, and SLAM.

> **Status:** Active Development

---

## Motivation

Reliable localization is a fundamental challenge for underwater autonomous systems. Unlike terrestrial and aerial environments, GPS is generally unavailable underwater, making autonomous navigation dependent on onboard sensing.

U-SLAM is motivated by the need for a reproducible environment in which different sensing modalities and localization approaches can be studied under controlled conditions.

The project focuses on combining simulated underwater sensing with ground-truth data to enable systematic development and evaluation of SLAM algorithms.

---

## Current System

The current implementation establishes the simulation and sensing foundation for the project.

It includes:

- Underwater vehicle simulation in Gazebo
- Hydrographic and acoustic ROS 2 message interfaces
- DVL simulation
- Pressure/depth sensing
- Ground-truth state information
- Sensor synchronization
- Multibeam sonar simulation
- Underwater environments and models

These components form the infrastructure on which the SLAM and evaluation pipeline will be developed.

---

## System Architecture

```text
                    U-SLAM
                       │
        ┌──────────────┴──────────────┐
        │                             │
   Simulation                    Sensor Layer
        │                             │
        │                ┌────────────┼────────────┐
        │                │            │            │
        │               IMU          DVL       Multibeam
        │                            │            Sonar
        │                │            │            │
        └────────────────┴────────────┴────────────┘
                             │
                             ▼
                    Data Synchronization
                             │
                             ▼
                    Dataset Generation
                             │
                             ▼
                       SLAM Pipeline
                             │
                             ▼
                         Evaluation
