好的，没问题。

我将在您现有的笔记中，对 `netstat` 命令进行更详细的展开说明，并将其完美地融入到**第五部分：网络操作**中，同时确保笔记的其他所有内容都保持原样，不会有任何遗漏。

以下是更新后的完整笔记：

---

## **Ubuntu / Linux 常用命令速查笔记**

### **一、系统与软件包管理**

#### 1. 软件包更新与安装 (APT)

*   **更新软件包列表** (安装/更新前必做)
    ```bash
    sudo apt update
    ```
*   **升级所有已安装的软件包**
    ```bash
    sudo apt upgrade
    ```*   **安装软件**
    ```bash
    sudo apt install <package_name>
    ```*   **卸载软件** (保留配置文件)
    ```bash
    sudo apt remove <package_name>
    ```
*   **彻底卸载软件** (删除所有相关文件)
    ```bash
    sudo apt purge <package_name>
    ```*   **自动清理不再需要的依赖包**
    ```bash
    sudo apt autoremove
    ```
*   **搜索软件包**
    ```bash
    apt search <keyword>
    ```

#### 2. Debian 包管理 (dpkg) - 用于 `.deb` 文件

*   **安装本地 .deb 包**
    ```bash
    sudo dpkg -i <package_file.deb>
    ```
*   **查看已安装的软件包**
    ```bash
    dpkg -l
    dpkg -l | grep <keyword> # 筛选查看
    ```
*   **检查包的安装状态**
    ```bash
    dpkg-query -W -f='${Status}' <package_name>
    ```

#### 3. 系统控制

*   **进入睡眠 (挂起)**
    ```bash
    systemctl suspend
    ```
*   **重启系统**
    ```bash
    sudo reboot
    ```
*   **关闭系统**
    ```bash
    sudo shutdown now
    ```

---

### **二、文件与目录操作**

#### 1. 导航与查看

*   **显示当前工作目录**
    ```bash
    pwd
    ```
*   **列出文件和目录**
    ```bash
    ls             # 普通列表
    ls -l          # 详细列表
    ls -a          # 显示隐藏文件
    ls -lh         # 详细列表并以易读格式显示大小 (e.g., KB, MB)
    ```*   **切换目录**
    ```bash
    cd /path/to/directory  # 切换到指定目录
    cd ..                  # 返回上一级目录
    cd ~                   # 返回家目录
    cd -                   # 返回上一次所在的目录
    ```
*   **以图形化方式打开当前目录**
    ```bash
    nautilus .
    ```

#### 2. 创建与删除

*   **创建新目录**
    ```bash
    mkdir new_folder
    mkdir -p parent_folder/child_folder  # 递归创建多级目录
    ```
*   **创建空文件** (更多见文本编辑章节)
    ```bash
    touch new_file.txt
    ```
*   **删除文件**
    ```bash
    rm file_name
    ```
*   **删除空目录**
    ```bash
    rmdir directory_name
    ```
*   **递归删除目录及其内容** (危险操作，请谨慎使用)
    ```bash
    rm -r directory_name   # 会逐一提示
    rm -rf directory_name  # 强制递归删除，无任何提示！
    ```

#### 3. 复制与移动

*   **复制文件或目录**
    ```bash
    cp source_file destination_file
    cp -r source_directory/ destination_directory/  # -r 递归复制目录
    # 示例: 复制并显示过程 (-v)
    cp -rv ./Euen/ /media/quan/Lexar/Euen
    ```
*   **移动或重命名文件/目录**
    ```bash
    mv old_name new_name
    mv source_file destination_directory/
    ```

---

### **三、文本编辑**

| 工具 | 核心用途 | 学习曲线 | 适用场景 |
| :--- | :--- | :--- | :--- |
| **vim** | 功能强大的文本编辑器 | 陡峭 | 编程、写文档、复杂文本操作 |
| **nano** | 简单易用的文本编辑器 | 极低 | 快速修改配置文件、简单编辑 |
| **touch** | 创建空文件/更新时间戳 | 极低 | 创建项目文件结构、脚本自动化 |

#### 1. Vim (强大的模式编辑器)

*   **启动:** `vim <filename>`
*   **核心模式:**
    *   **普通模式:** 移动、复制、粘贴、删除 (`dd`, `yy`, `p`)。
    *   **插入模式:** 输入文本 (按 `i` 进入, `Esc` 退出)。
    *   **命令模式:** 保存、退出 (按 `:` 进入, 如 `:wq`, `:q!`)。

#### 2. Nano (新手友好的编辑器)

*   **启动:** `nano <filename>`
*   **核心操作 (界面底部有提示):**
    *   `Ctrl + O`: 保存 (Write Out)。
    *   `Ctrl + X`: 退出。
    *   `Ctrl + W`: 搜索。

#### 3. Touch (创建空文件)

*   **主要功能:** 如果文件不存在，则创建一个空文件。
    ```bash
    # 创建单个空文件
    touch new_script.py
    # 同时创建多个
    touch file1.txt file2.txt
    ```
*   **次要功能:** 如果文件已存在，则更新其修改时间戳。

---

### **四、系统信息与监控**

#### 1. 硬件与驱动

*   **检查 NVIDIA GPU 和驱动信息** (NVIDIA 用户)
    ```bash
    nvidia-smi
    ```
*   **检查 CUDA 版本** (NVIDIA 用户)
    ```bash
    nvcc --version
    ```
*   **查看 CPU 信息**
    ```bash
    lscpu
    ```
*   **查看内存使用**
    ```bash
    free -h
    ```

#### 2. 磁盘与文件系统

*   **查看磁盘空间使用情况**
    ```bash
    df -h
    ```
*   **查看指定目录的大小**
    ```bash
    du -sh /path/to/directory
    ```
*   **以树状结构显示目录**
    ```bash
    # 可能需要先安装: sudo apt install tree
    tree /path/to/directory
    ```
*   **统计目录下的文件/子目录数量**
    ```bash
    # 仅统计文件数量
    find /path/to/directory -type f | wc -l
    # 仅统计子目录数量
    find /path/to/directory -type d | wc -l
    ```

---

### **五、网络操作**

#### 1. 网络连接与诊断

*   **查看网络接口信息** (IP 地址等)
    ```bash
    ip addr show   # 推荐
    ifconfig       # 旧版命令，可能需安装
    ```
*   **测试与主机的连通性**
    ```bash
    ping <目标主机或IP>
    ```

#### 2. 端口与进程

*   **查询占用指定端口的进程**
    有多个工具可以实现，功能类似：
    ```bash
    # lsof (list open files) 是最通用的
    sudo lsof -i:<端口号>

    # ss 是 netstat 的现代替代品，速度更快
    sudo ss -tunlp | grep <端口号>

    # netstat 是经典的网络统计工具
    sudo netstat -tunlp | grep <端口号>
    ```

*   **`netstat` 命令详解**
    `netstat` 是一个强大的网络统计工具。最常用的组合是 `netstat -tunlp`：
    | 选项 | 含义 |
    | :--- | :--- |
    | **-t** | 显示 **T**CP 连接 |
    | **-u** | 显示 **U**DP 连接 |
    | **-n** | 以**N**umeric（数字）格式显示地址和端口，不进行域名解析 |
    | **-l** | 仅显示正在 **L**istening (监听) 的服务 |
    | **-p** | 显示占用端口的**P**rogram（程序名）和进程ID (PID) |
    **示例:**
    ```bash
    # 查看所有正在监听的 TCP 和 UDP 端口及其程序
    sudo netstat -tunlp
    # 筛选出与端口 3000 相关的信息
    sudo netstat -tunlp | grep 3000
    ```

---

### **六、进程管理**

*   **查看当前所有进程**
    ```bash
    ps -ef
    ps -ef | grep <keyword>  # 筛选查找特定进程
    ```*   **实时动态监控进程**
    ```bash
    top      # 经典工具
    htop     # 增强版，需安装 (sudo apt install htop)
    ```
*   **结束进程**
    ```bash
    kill <PID>            # 默认发送 SIGTERM (15) 信号，请求正常退出
    kill -9 <PID>         # 发送 SIGKILL (9) 信号，强制立即杀死
    pkill <process_name>  # 按名称结束进程
    pkill -9 <process_name> # 按名称强制结束进程
    ```

---

### **七、远程连接与文件传输**

*   **通过 SSH 登录远程主机**
    ```bash
    ssh <user>@<host_ip>
    # 示例:
    ssh ncs@192.168.9.128
    ```
*   **将远程文件系统挂载到本地** (需安装 `sshfs`)
    ```bash
    sshfs <user>@<host_ip>:<远程路径> <本地挂载点>
    # 示例:
    sshfs ncs@192.168.9.128:/home/ncs/ /home/quan/remote128/
    ```
*   **安全复制文件/目录 (scp)**
    ```bash
    # 从远程复制到本地
    scp <user>@<host_ip>:<远程文件路径> <本地路径>
    # 从本地复制到远程
    scp <本地文件> <user>@<host_ip>:<远程路径>
    # 递归复制目录 (-r)
    scp -r <本地目录> <user>@<host_ip>:<远程目录>
    ```
*   **增量同步文件/目录 (rsync)** (更高效，支持断点续传)
    ```bash
    # -avz: 归档、详细、压缩 --progress: 显示进度
    rsync -avz --progress <源目录/> <user>@<host_ip>:<目标目录>
    ```

---

### **八、Python 与环境管理**

#### 1. 安装多版本 Python

```bash
# 添加 PPA 源
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update
# 安装指定版本
sudo apt install python3.12 python3.12-venv python3.12-dev```

#### 2. 虚拟环境管理 (venv / uv)

*   **使用标准 venv 创建虚拟环境**
    ```bash
    python3.12 -m venv .venv
    ```*   **使用 uv 创建虚拟环境** (新一代高速工具)
    ```bash
    # --seed: 预装 pip, setuptools 和 wheel, 使环境立即可用 (推荐)
    # --python: 指定要使用的 Python 解释器
    uv venv --seed --python 3.9
    ```
*   **激活虚拟环境**
    ```bash
    # Linux / macOS
    source .venv/bin/activate
    # Windows
    .venv\Scripts\activate
    ```
*   **退出虚拟环境**
    ```bash
    deactivate
    ```

#### 3. 包管理

*   **使用 requirements.txt 安装依赖**
    ```bash
    pip install -r requirements.txt
    uv pip install -r requirements.txt # 使用 uv
    ```
*   **导出当前环境的依赖**
    ```bash
    pip freeze > requirements.txt
    ```

---

### **九、其他实用命令**

*   **加载/重载环境变量**
    ```bash
    source ~/.bashrc
    ```
*   **查看所有环境变量**
    ```bash
    printenv
    ```
*   **解压 zip 文件**
    ```bash
    unzip <file.zip> -d <目标目录>
    ```