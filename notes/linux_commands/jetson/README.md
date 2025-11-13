## **技术笔记：在 Jetson AGX Orin 上部署多模态模型的探索与分析**

### **目标**

在 Jetson AGX Orin 平台，部署并运行 Qwen3-VL 多模态大模型。本笔记记录环境准备、部署尝试、问题分析、成功参照案例及最终结论，以供未来查阅。

### **第一阶段：环境诊断与系统升级**

部署工作始于对现有平台环境的评估。

**1. 初始环境诊断**

*   **JetPack 版本 (`cat /etc/nv_tegra_release`)**:
    ```
    # R35 (release), REVISION: 4.1, ...
    ```
    *   **分析**: L4T R35.4.1 对应 **JetPack 5.1.2**，版本过旧，无法满足新模型和新推理框架的依赖。

**2. 系统升级至 JetPack 6**

为满足部署要求，将设备系统升级。

*   **升级后 L4T 版本 (`cat /etc/nv_tegra_release`)**:
    ```
    # R36 (release), REVISION: 4.4, ...
    ```
*   **升级后 JetPack 版本 (`sudo apt-cache show nvidia-jetpack`)**:
    ```
    Package: nvidia-jetpack
    Source: nvidia-jetpack (6.2.1)
    Version: 6.2.1+b38
    Architecture: arm64
    Maintainer: NVIDIA Corporation
    Installed-Size: 194
    ...
    ```
    *   **分析**: 系统成功升级至 **JetPack 6.2.1**，为后续部署提供了必要的基础环境。

### **第二阶段：Qwen3-VL 部署尝试 (vLLM)**

根据 Qwen3-VL 官方推荐，首选 vLLM (`>=0.11.0`) 进行部署。

**1. 环境准备**

*   **安装 PyTorch**:
    *   **来源**: [https://pypi.jetson-ai-lab.io/jp6/cu126](https://pypi.jetson-ai-lab.io/jp6/cu126)
    *   **指令**: `pip install torch‑2.8.0‑cp310‑cp310‑linux_aarch64.whl` (及对应的 torchvision/torchaudio)

**2. vLLM 安装与失败分析**

*   **尝试预编译包**: 从社区源安装 `vllm‑0.10.2+cu126‑cp310‑cp310-linux_aarch64.whl`。
    *   **问题 1**: 该版本为 `0.10.2`，不满足 Qwen3-VL 要求的 `>=0.11.0`。
    *   **问题 2**: 尝试从源码编译更高版本的 vLLM，因其脚本和依赖主要面向 x86_64 平台，在 aarch64 上失败。
*   **结论**: vLLM 路径因**核心依赖不满足**而中断。

### **第三阶段：Qwen3-VL 部署尝试 (TensorRT-LLM)**

转向 NVIDIA 官方的 TensorRT-LLM 方案。

*   **相关资源**:
    *   **GitHub**: [https://github.com/NVIDIA/TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM)
    *   **Jetson 指南**: [README4Jetson.md](https://github.com/NVIDIA/TensorRT-LLM/blob/v0.12.0-jetson/README4Jetson.md)
*   **部署尝试与失败分析**:
    *   在编译 Qwen3-VL 模型时，转换过程失败。
    *   **结论**: 当前版本的 TensorRT-LLM **缺少对 Qwen3-VL 模型中新模块的支持**，无法完成模型编译。

### **第四阶段：成功案例 — Docker 部署 Qwen2.5-VL**

在尝试部署 Qwen3-VL 遇到障碍的同时，并行进行了一项基于 Docker 的 Qwen2.5-VL 部署测试，并取得成功。这个案例证明了容器化方案在 Jetson 平台的可行性和便捷性。

*   **核心工具**: `jetson-containers`
    *   **项目简介**: 一个为 Jetson 平台提供预配置、GPU 加速的 Docker 镜像的社区项目，极大地简化了复杂 AI 环境的部署。
    *   **参考链接**:
        *   **项目 GitHub**: [https://github.com/dusty-nv/jetson-containers](https://github.com/dusty-nv/jetson-containers)
        *   **vLLM 容器**: [jetson-containers/packages/llm/vllm](https://github.com/dusty-nv/jetson-containers/tree/master/packages/llm/vllm)
        *   **部署过程参考**: [CSDN 博文](https://blog.csdn.net/qq_37397652/article/details/147249278)

*   **部署指令**:
    1.  **启动 vLLM 容器并挂载模型缓存**:
        ```bash
        jetson-containers run \
            -v /home/ncs/.cache/modelscope/hub/models/Qwen/Qwen2.5-VL-3B-Instruct:/root/.cache/modelscope/hub/models/Qwen/Qwen2.5-VL-3B-Instruct \
            $(autotag vllm)
        ```
    2.  **在容器内安装模型所需依赖**:
        ```bash
        pip install --index-url https://pypi.org/simple modelscope
        pip install --index-url https://pypi.org/simple qwen-vl-utils[decord]
        ```
*   **结论**: **部署成功**。这表明，**当社区提供了适配 Jetson 平台的容器化环境后，部署流程可以被显著简化并取得成功**。失败的原因不在于 Jetson 平台本身的能力，而在于针对特定新模型的生态支持是否到位。

### **最终总结与后续策略**

1.  **当前障碍总结 (针对 Qwen3-VL)**:
    *   **vLLM**: 缺少适用于 Jetson (aarch64) 且版本兼容 (`>=0.11.0`) 的预编译包，源码编译困难。
    *   **TensorRT-LLM**: 框架本身暂不支持 Qwen3-VL 的新模型架构。

2.  **proven 可行的路径 (基于 Qwen2.5-VL 的成功经验)**:
    *   使用 `jetson-containers` 等社区维护的 Docker 镜像是目前在 Jetson 上部署复杂大模型**最可靠、最高效**的方法。

3.  **最终推荐策略**:
    *   对于 Qwen3-VL，当前最稳妥的策略是**等待社区生态的更新**，而非自行解决底层的编译和兼容性问题。
    *   应持续关注以下资源：
        *   **vLLM for Jetson**: 等待 `pypi.jetson-ai-lab.io` 更新更高版本的 vLLM wheel 文件。
        *   **TensorRT-LLM**: 关注 NVIDIA 官方的更新，看其何时会加入对 Qwen3-VL 的支持。
        *   **jetson-containers**: **(最推荐)** 关注该项目的更新，等待其推出支持 Qwen3-VL 的 Docker 镜像。这很可能是解决部署问题的最快途径。