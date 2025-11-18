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
    ```
*   **安装软件**
    ```bash
    sudo apt install <package_name>
    ```
*   **卸载软件** (保留配置文件)
    ```bash
    sudo apt remove <package_name>
    ```
*   **彻底卸载软件** (删除所有相关文件)
    ```bash
    sudo apt purge <package_name>
    ```
*   **自动清理不再需要的依赖包**
    ```bash
    sudo apt autoremove
    ```
*   **搜索软件包**
    ```bash
    apt search <keyword>
    ```
*   **查看软件包信息**
    ```bash
    apt show <package_name>
    ```
*   **列出可升级的软件包**
    ```bash
    apt list --upgradable
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
*   **查看包安装的所有文件**
    ```bash
    dpkg -L <package_name>
    ```
*   **查询某个文件属于哪个包**
    ```bash
    dpkg -S /path/to/file
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
    sudo shutdown -h now     # 同上
    sudo shutdown -h +10     # 10分钟后关机
    sudo shutdown -r now     # 立即重启
    ```
*   **查看系统信息**
    ```bash
    uname -a                 # 内核版本等
    lsb_release -a           # Ubuntu版本信息
    hostnamectl              # 主机名和系统信息
    ```
*   **查看系统启动时间和运行时长**
    ```bash
    uptime
    ```
*   **查看系统日志**
    ```bash
    journalctl               # 查看所有日志
    journalctl -f            # 实时跟踪日志
    journalctl -u <service>  # 查看特定服务日志
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
    ls -lt         # 按修改时间排序
    ls -lS         # 按文件大小排序
    ls -R          # 递归显示所有子目录
    ```
*   **切换目录**
    ```bash
    cd /path/to/directory  # 切换到指定目录
    cd ..                  # 返回上一级目录
    cd ~                   # 返回家目录
    cd -                   # 返回上一次所在的目录
    ```
*   **以图形化方式打开当前目录**
    ```bash
    nautilus .             # GNOME
    dolphin .              # KDE
    xdg-open .             # 通用方式
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
    rm -i file_name        # 删除前确认
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
    cp -i source destination                        # 覆盖前确认
    # 示例: 复制并显示过程 (-v)
    cp -rv ./Euen/ /media/quan/Lexar/Euen
    ```
*   **移动或重命名文件/目录**
    ```bash
    mv old_name new_name
    mv source_file destination_directory/
    mv -i source destination  # 覆盖前确认
    ```
*   **移动目录下所有内容到当前目录**
    ```bash
    # 移动所有文件和子目录（不包括隐藏文件）
    mv Automodel/* .
    
    # 移动所有文件包括隐藏文件
    mv Automodel/* Automodel/.* . 2>/dev/null
    
    # 完整操作：移动所有内容并删除空文件夹
    mv Automodel/* Automodel/.* . 2>/dev/null
    rmdir Automodel
    ```

#### 4. 文件查找

*   **按名称查找文件**
    ```bash
    find /path -name "filename"
    find . -name "*.py"              # 当前目录查找所有Python文件
    find . -iname "*.txt"            # 不区分大小写
    ```
*   **按类型查找**
    ```bash
    find . -type f                   # 查找文件
    find . -type d                   # 查找目录
    find . -type l                   # 查找符号链接
    ```
*   **按大小查找**
    ```bash
    find . -size +100M               # 大于100MB的文件
    find . -size -1M                 # 小于1MB的文件
    ```
*   **按时间查找**
    ```bash
    find . -mtime -7                 # 7天内修改的文件
    find . -mtime +30                # 30天前修改的文件
    ```
*   **查找并执行操作**
    ```bash
    find . -name "*.log" -delete     # 删除所有.log文件
    find . -name "*.txt" -exec cat {} \;  # 显示所有txt文件内容
    ```
*   **快速查找（需安装 locate）**
    ```bash
    sudo apt install mlocate
    sudo updatedb                    # 更新文件数据库
    locate filename                  # 快速查找
    ```

#### 5. 文件权限

*   **查看文件权限**
    ```bash
    ls -l filename
    ```
*   **修改文件权限**
    ```bash
    chmod 755 filename               # rwxr-xr-x
    chmod u+x filename               # 给所有者添加执行权限
    chmod -R 755 directory           # 递归修改目录权限
    ```
    **权限数字说明：**
    - 4 = 读(r), 2 = 写(w), 1 = 执行(x)
    - 755 = rwxr-xr-x (所有者全部，组和其他读+执行)
    - 644 = rw-r--r-- (所有者读写，其他只读)

*   **修改文件所有者**
    ```bash
    sudo chown user:group filename
    sudo chown -R user:group directory
    ```

#### 6. 符号链接

*   **创建软链接（符号链接）**
    ```bash
    ln -s /path/to/original /path/to/link
    # 示例：创建Python链接
    sudo ln -s /usr/bin/python3.10 /usr/bin/python
    ```
*   **创建硬链接**
    ```bash
    ln /path/to/original /path/to/link
    ```

---

### **三、文本编辑与处理**

| 工具 | 核心用途 | 学习曲线 | 适用场景 |
| :--- | :--- | :--- | :--- |
| **vim** | 功能强大的文本编辑器 | 陡峭 | 编程、写文档、复杂文本操作 |
| **nano** | 简单易用的文本编辑器 | 极低 | 快速修改配置文件、简单编辑 |
| **touch** | 创建空文件/更新时间戳 | 极低 | 创建项目文件结构、脚本自动化 |
| **cat/less/more** | 查看文件内容 | 极低 | 快速查看文件 |
| **grep** | 文本搜索 | 中等 | 日志分析、代码搜索 |
| **sed** | 流编辑器 | 中等 | 批量文本替换 |
| **awk** | 文本处理 | 较高 | 数据提取、报表生成 |

#### 1. Vim (强大的模式编辑器)

*   **启动:** `vim <filename>`
*   **核心模式:**
    *   **普通模式:** 移动、复制、粘贴、删除 (`dd`, `yy`, `p`)。
    *   **插入模式:** 输入文本 (按 `i` 进入, `Esc` 退出)。
    *   **命令模式:** 保存、退出 (按 `:` 进入, 如 `:wq`, `:q!`)。
*   **常用命令:**
    ```bash
    :w          # 保存
    :q          # 退出
    :wq         # 保存并退出
    :q!         # 强制退出不保存
    /pattern    # 搜索
    :%s/old/new/g  # 全局替换
    ```

#### 2. Nano (新手友好的编辑器)

*   **启动:** `nano <filename>`
*   **核心操作 (界面底部有提示):**
    *   `Ctrl + O`: 保存 (Write Out)。
    *   `Ctrl + X`: 退出。
    *   `Ctrl + W`: 搜索。
    *   `Ctrl + K`: 剪切行。
    *   `Ctrl + U`: 粘贴。

#### 3. Touch (创建空文件)

*   **主要功能:** 如果文件不存在，则创建一个空文件。
    ```bash
    # 创建单个空文件
    touch new_script.py
    # 同时创建多个
    touch file1.txt file2.txt
    ```
*   **次要功能:** 如果文件已存在，则更新其修改时间戳。

#### 4. 查看文件内容

*   **显示整个文件**
    ```bash
    cat filename                    # 显示全部内容
    cat file1 file2 > combined      # 合并文件
    ```
*   **分页查看**
    ```bash
    less filename                   # 推荐，支持前后翻页
    more filename                   # 只能向后翻页
    ```
*   **查看文件开头/结尾**
    ```bash
    head filename                   # 前10行
    head -n 20 filename             # 前20行
    tail filename                   # 后10行
    tail -n 20 filename             # 后20行
    tail -f filename                # 实时跟踪文件更新（查看日志必备）
    ```

#### 5. 文本搜索与处理

*   **grep - 搜索文本**
    ```bash
    grep "pattern" filename         # 搜索包含pattern的行
    grep -r "pattern" directory     # 递归搜索目录
    grep -i "pattern" filename      # 不区分大小写
    grep -n "pattern" filename      # 显示行号
    grep -v "pattern" filename      # 显示不匹配的行
    grep -c "pattern" filename      # 统计匹配行数
    grep -E "regex" filename        # 使用正则表达式
    ```
*   **sed - 流编辑器（替换文本）**
    ```bash
    sed 's/old/new/' filename       # 替换每行第一个匹配
    sed 's/old/new/g' filename      # 替换所有匹配
    sed -i 's/old/new/g' filename   # 直接修改文件
    sed -n '5,10p' filename         # 打印第5到10行
    ```
*   **awk - 文本处理**
    ```bash
    awk '{print $1}' filename       # 打印第一列
    awk -F':' '{print $1}' /etc/passwd  # 指定分隔符
    awk '$3 > 100' filename         # 打印第三列大于100的行
    ```
*   **wc - 统计**
    ```bash
    wc filename                     # 行数 单词数 字节数
    wc -l filename                  # 只统计行数
    wc -w filename                  # 只统计单词数
    ```
*   **sort - 排序**
    ```bash
    sort filename                   # 按字母排序
    sort -n filename                # 按数字排序
    sort -r filename                # 反向排序
    sort -u filename                # 去重排序
    ```
*   **uniq - 去重**
    ```bash
    sort filename | uniq            # 去除重复行
    sort filename | uniq -c         # 统计重复次数
    ```

#### 6. 文件对比

*   **diff - 比较文件差异**
    ```bash
    diff file1 file2
    diff -u file1 file2             # 统一格式输出
    ```

---

### **四、系统信息与监控**

#### 1. 硬件与驱动

*   **检查 NVIDIA GPU 和驱动信息** (NVIDIA 用户)
    ```bash
    nvidia-smi
    nvidia-smi -l 1                 # 每秒刷新一次
    watch -n 1 nvidia-smi           # 使用watch监控
    ```
*   **检查 CUDA 版本** (NVIDIA 用户)
    ```bash
    nvcc --version
    cat /usr/local/cuda/version.txt
    ```
*   **查看 CPU 信息**
    ```bash
    lscpu
    cat /proc/cpuinfo
    nproc                           # CPU核心数
    ```
*   **查看内存使用**
    ```bash
    free -h
    free -h -s 2                    # 每2秒刷新
    ```
*   **查看所有硬件信息**
    ```bash
    sudo lshw                       # 详细硬件信息
    sudo lshw -short                # 简短格式
    ```
*   **查看PCI设备（显卡等）**
    ```bash
    lspci | grep -i vga
    lspci | grep -i nvidia
    ```
*   **查看USB设备**
    ```bash
    lsusb
    ```

#### 2. 磁盘与文件系统

*   **查看磁盘空间使用情况**
    ```bash
    df -h
    df -h /                         # 查看根分区
    ```
*   **查看指定目录的大小**
    ```bash
    du -sh /path/to/directory
    du -h --max-depth=1 /path       # 显示一级子目录大小
    du -sh * | sort -h              # 当前目录各项按大小排序
    ```
*   **以树状结构显示目录**
    ```bash
    # 可能需要先安装: sudo apt install tree
    tree /path/to/directory
    tree -L 2                       # 只显示2层
    tree -d                         # 只显示目录
    ```
*   **统计目录下的文件/子目录数量**
    ```bash
    # 仅统计文件数量
    find /path/to/directory -type f | wc -l
    # 仅统计子目录数量
    find /path/to/directory -type d | wc -l
    ```
*   **查看磁盘IO**
    ```bash
    iostat                          # 需安装 sysstat
    sudo iotop                      # 实时IO监控
    ```
*   **查看挂载点**
    ```bash
    mount
    findmnt
    ```

#### 3. 系统性能监控

*   **实时系统监控**
    ```bash
    top                             # 经典工具
    htop                            # 增强版（需安装）
    btop                            # 现代化界面（需安装）
    ```
*   **查看系统负载**
    ```bash
    uptime
    w                               # 谁在线及负载
    ```

---

### **五、网络操作**

#### 1. 网络连接与诊断

*   **查看网络接口信息** (IP 地址等)
    ```bash
    ip addr show                    # 推荐
    ip a                            # 简写
    ifconfig                        # 旧版命令，可能需安装
    ```
*   **测试与主机的连通性**
    ```bash
    ping <目标主机或IP>
    ping -c 4 google.com            # 只ping 4次
    ```
*   **追踪路由**
    ```bash
    traceroute google.com
    tracepath google.com
    ```
*   **DNS查询**
    ```bash
    nslookup google.com
    dig google.com
    host google.com
    ```
*   **查看路由表**
    ```bash
    ip route show
    route -n
    ```
*   **查看防火墙状态**
    ```bash
    sudo ufw status
    sudo iptables -L
    ```

#### 2. 端口与进程

*   **查询占用指定端口的进程**
    有多个工具可以实现，功能类似：
    ```bash
    # lsof (list open files) 是最通用的
    sudo lsof -i:<端口号>
    sudo lsof -i:8080

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

#### 3. 网络工具

*   **下载文件**
    ```bash
    wget <URL>
    wget -O custom_name.zip <URL>   # 指定文件名
    wget -c <URL>                   # 断点续传
    curl -O <URL>                   # 使用curl下载
    curl -L <URL> -o filename       # 跟随重定向
    ```
*   **测试网速**
    ```bash
    sudo apt install speedtest-cli
    speedtest-cli
    ```
*   **查看网络连接**
    ```bash
    netstat -an                     # 所有连接
    ss -s                           # 连接统计
    ```

---

### **六、进程管理**

*   **查看当前所有进程**
    ```bash
    ps -ef                          # 标准格式
    ps aux                          # BSD格式，更详细
    ps -ef | grep <keyword>         # 筛选查找特定进程
    ```
*   **实时动态监控进程**
    ```bash
    top                             # 经典工具
    htop                            # 增强版，需安装 (sudo apt install htop)
    btop                            # 现代化版本
    ```
*   **查看进程树**
    ```bash
    pstree
    pstree -p                       # 显示PID
    ```
*   **结束进程**
    ```bash
    kill <PID>                      # 默认发送 SIGTERM (15) 信号
    kill -9 <PID>                   # 发送 SIGKILL (9) 信号，强制杀死
    kill -15 <PID>                  # 友好退出
    pkill <process_name>            # 按名称结束进程
    pkill -9 <process_name>         # 按名称强制结束
    killall <process_name>          # 结束所有同名进程
    ```
*   **后台运行进程**
    ```bash
    command &                       # 后台运行
    nohup command &                 # 后台运行且不受终端关闭影响
    nohup command > output.log 2>&1 &  # 重定向输出到日志
    ```
*   **查看后台任务**
    ```bash
    jobs                            # 查看当前终端的后台任务
    bg                              # 将任务放到后台
    fg                              # 将任务调到前台
    ```
*   **screen / tmux（终端复用）**
    ```bash
    # screen
    screen                          # 创建新会话
    screen -S session_name          # 创建命名会话
    screen -ls                      # 列出所有会话
    screen -r session_name          # 恢复会话
    Ctrl+A, D                       # 分离会话
    
    # tmux
    tmux                            # 创建新会话
    tmux new -s session_name        # 创建命名会话
    tmux ls                         # 列出会话
    tmux attach -t session_name     # 附加到会话
    Ctrl+B, D                       # 分离会话
    ```

---

### **七、远程连接与文件传输**

*   **通过 SSH 登录远程主机**
    ```bash
    ssh <user>@<host_ip>
    ssh -p <port> <user>@<host_ip>  # 指定端口
    # 示例:
    ssh ncs@192.168.9.128
    ```
*   **SSH密钥管理**
    ```bash
    # 生成SSH密钥对
    ssh-keygen -t rsa -b 4096
    # 复制公钥到远程主机
    ssh-copy-id user@host
    # 或手动复制
    cat ~/.ssh/id_rsa.pub | ssh user@host "mkdir -p ~/.ssh && cat >> ~/.ssh/authorized_keys"
    ```
*   **将远程文件系统挂载到本地** (需安装 `sshfs`)
    ```bash
    sshfs <user>@<host_ip>:<远程路径> <本地挂载点>
    # 示例:
    sshfs ncs@192.168.9.128:/home/ncs/ /home/quan/remote128/
    # 卸载
    fusermount -u /home/quan/remote128/
    ```
*   **安全复制文件/目录 (scp)**
    ```bash
    # 从远程复制到本地
    scp <user>@<host_ip>:<远程文件路径> <本地路径>
    # 从本地复制到远程
    scp <本地文件> <user>@<host_ip>:<远程路径>
    # 递归复制目录 (-r)
    scp -r <本地目录> <user>@<host_ip>:<远程目录>
    # 指定端口
    scp -P 2222 file user@host:/path
    ```
*   **增量同步文件/目录 (rsync)** (更高效，支持断点续传)
    ```bash
    # -avz: 归档、详细、压缩 --progress: 显示进度
    rsync -avz --progress <源目录/> <user>@<host_ip>:<目标目录>
    # 删除目标目录中源目录没有的文件
    rsync -avz --delete <源/> <目标/>
    # 排除某些文件
    rsync -avz --exclude='*.log' source/ dest/
    ```
*   **SSH隧道（端口转发）**
    ```bash
    # 本地端口转发
    ssh -L local_port:remote_host:remote_port user@ssh_server
    # 远程端口转发
    ssh -R remote_port:local_host:local_port user@ssh_server
    # 动态端口转发（SOCKS代理）
    ssh -D 1080 user@ssh_server
    ```

---

### **八、Python 与环境管理**

#### 1. 安装多版本 Python

```bash
# 添加 PPA 源
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update
# 安装指定版本
sudo apt install python3.12 python3.12-venv python3.12-dev
```

#### 2. 虚拟环境管理 (venv / uv)

*   **使用标准 venv 创建虚拟环境**
    ```bash
    python3.12 -m venv .venv
    ```
*   **使用 uv 创建虚拟环境** (新一代高速工具)
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
    pip list --format=freeze > requirements.txt
    ```
*   **升级pip**
    ```bash
    pip install --upgrade pip
    python -m pip install --upgrade pip
    ```
*   **查看已安装的包**
    ```bash
    pip list
    pip show <package_name>         # 查看包详情
    ```

#### 4. Conda环境管理

```bash
# 创建环境
conda create -n myenv python=3.10
# 激活环境
conda activate myenv
# 退出环境
conda deactivate
# 列出所有环境
conda env list
# 删除环境
conda remove -n myenv --all
# 导出环境
conda env export > environment.yml
# 从文件创建环境
conda env create -f environment.yml
```

---

### **九、压缩与解压**

#### 1. tar（打包）

```bash
# 打包并压缩（gzip）
tar -czvf archive.tar.gz /path/to/directory
# 解压
tar -xzvf archive.tar.gz
tar -xzvf archive.tar.gz -C /path/to/extract

# 打包不压缩
tar -cvf archive.tar /path/to/directory
# 解包
tar -xvf archive.tar

# 打包并压缩（bzip2，压缩率更高但更慢）
tar -cjvf archive.tar.bz2 /path/to/directory
# 解压
tar -xjvf archive.tar.bz2

# 查看压缩包内容
tar -tvf archive.tar.gz

# tar参数说明
# -c: create (创建)
# -x: extract (解压)
# -z: gzip压缩
# -j: bzip2压缩
# -v: verbose (显示详细过程)
# -f: file (指定文件名)
```

#### 2. zip / unzip

```bash
# 压缩文件
zip archive.zip file1 file2
# 压缩目录
zip -r archive.zip /path/to/directory

# 解压
unzip archive.zip
unzip archive.zip -d /path/to/extract  # 解压到指定目录

# 查看压缩包内容
unzip -l archive.zip

# 解压时排除某些文件
unzip archive.zip -x "*.log"
```

#### 3. gzip / gunzip

```bash
# 压缩文件（会替换原文件）
gzip filename
# 保留原文件
gzip -k filename

# 解压
gunzip filename.gz
gzip -d filename.gz

# 查看压缩文件内容
zcat filename.gz
zless filename.gz
```

#### 4. 其他压缩格式

```bash
# 7z（需安装 p7zip-full）
7z a archive.7z /path/to/directory    # 压缩
7z x archive.7z                        # 解压

# rar（需安装 unrar）
unrar x archive.rar                    # 解压
```

---

### **十、其他实用命令**

#### 1. 环境变量

*   **加载/重载环境变量**
    ```bash
    source ~/.bashrc
    source ~/.bash_profile
    . ~/.bashrc                        # 同source
    ```
*   **查看所有环境变量**
    ```bash
    printenv
    env
    ```
*   **查看特定环境变量**
    ```bash
    echo $PATH
    echo $HOME
    printenv PATH
    ```
*   **临时设置环境变量**
    ```bash
    export VAR_NAME=value
    export PATH=$PATH:/new/path
    ```
*   **永久设置环境变量**
    ```bash
    # 编辑 ~/.bashrc 或 ~/.profile
    echo 'export VAR_NAME=value' >> ~/.bashrc
    source ~/.bashrc
    ```

#### 2. 别名（Alias）

```bash
# 查看所有别名
alias

# 创建临时别名
alias ll='ls -alh'
alias gs='git status'

# 永久别名（添加到 ~/.bashrc）
echo "alias ll='ls -alh'" >> ~/.bashrc
source ~/.bashrc

# 删除别名
unalias ll
```

#### 3. 历史命令

```bash
# 查看命令历史
history
history | grep "keyword"

# 执行历史命令
!n                   # 执行第n条命令
!!                   # 执行上一条命令
!string              # 执行最近以string开头的命令

# 清除历史
history -c

# 搜索历史（Ctrl+R）
# 按 Ctrl+R 后输入关键词，再次按 Ctrl+R 继续搜索
```

#### 4. 日期与时间

```bash
# 显示当前日期时间
date
date "+%Y-%m-%d %H:%M:%S"

# 设置系统时间（需root）
sudo date -s "2024-01-01 12:00:00"

# 查看日历
cal
cal 2024             # 显示2024年日历
cal 12 2024          # 显示2024年12月

# 时区设置
timedatectl
sudo timedatectl set-timezone Asia/Shanghai
```

#### 5. 用户与组管理

```bash
# 查看当前用户
whoami
id
who                  # 查看登录用户

# 切换用户
su - username
sudo -i              # 切换到root

# 添加用户
sudo adduser username
sudo useradd -m username

# 删除用户
sudo deluser username
sudo userdel -r username  # 同时删除home目录

# 修改密码
passwd               # 修改自己的密码
sudo passwd username # 修改其他用户密码

# 添加用户到组
sudo usermod -aG groupname username
sudo usermod -aG sudo username  # 添加sudo权限

# 查看用户组
groups
groups username
```

#### 6. 系统服务管理（systemctl）

```bash
# 查看服务状态
sudo systemctl status service_name

# 启动服务
sudo systemctl start service_name

# 停止服务
sudo systemctl stop service_name

# 重启服务
sudo systemctl restart service_name

# 重新加载配置
sudo systemctl reload service_name

# 开机自启
sudo systemctl enable service_name

# 禁用开机自启
sudo systemctl disable service_name

# 查看所有服务
systemctl list-units --type=service

# 查看开机启动的服务
systemctl list-unit-files --type=service --state=enabled
```

#### 7. 定时任务（Cron）

```bash
# 编辑当前用户的定时任务
crontab -e

# 查看当前用户的定时任务
crontab -l

# 删除当前用户的所有定时任务
crontab -r

# Cron表达式格式
# * * * * * command
# 分 时 日 月 周
# 示例：
# 0 2 * * * /path/to/script.sh        # 每天凌晨2点执行
# */5 * * * * /path/to/script.sh      # 每5分钟执行
# 0 */2 * * * /path/to/script.sh      # 每2小时执行
# 0 0 * * 0 /path/to/script.sh        # 每周日午夜执行
```

#### 8. 系统清理

```bash
# 清理包缓存
sudo apt clean
sudo apt autoclean

# 清理不需要的依赖
sudo apt autoremove

# 清理日志
sudo journalctl --vacuum-time=7d    # 只保留7天的日志
sudo journalctl --vacuum-size=100M  # 只保留100M日志

# 清理临时文件
sudo rm -rf /tmp/*
sudo rm -rf /var/tmp/*

# 查找大文件
find / -type f -size +100M 2>/dev/null
du -ah / | sort -rh | head -n 20    # 查找最大的20个文件/目录
```

#### 9. 管道与重定向

```bash
# 管道（将前一个命令的输出作为后一个命令的输入）
command1 | command2
ps aux | grep python
cat file.txt | grep "pattern" | wc -l

# 输出重定向
command > file.txt               # 覆盖写入
command >> file.txt              # 追加写入
command 2> error.log             # 错误输出重定向
command > output.log 2>&1        # 标准输出和错误都重定向
command &> all.log               # 同上（简写）

# 输入重定向
command < input.txt

# Here Document
cat << EOF > file.txt
多行内容
可以直接写在这里
EOF

# tee（同时输出到屏幕和文件）
command | tee output.txt
command | tee -a output.txt      # 追加模式
```

#### 10. 脚本执行

```bash
# 使脚本可执行
chmod +x script.sh

# 执行脚本的多种方式
./script.sh                      # 需要执行权限
bash script.sh                   # 不需要执行权限
sh script.sh
source script.sh                 # 在当前shell执行
. script.sh                      # 同source

# 检查脚本语法
bash -n script.sh
shellcheck script.sh             # 需安装shellcheck
```

#### 11. 字符串处理

```bash
# 计算字符串长度
echo "hello" | wc -c
echo ${#string}

# 字符串拼接
str1="Hello"
str2="World"
echo "$str1 $str2"

# 字符串替换
echo "hello world" | sed 's/world/linux/'

# 字符串分割
echo "a:b:c" | cut -d':' -f2     # 输出b
IFS=':' read -ra ADDR <<< "a:b:c"

# 大小写转换
echo "Hello" | tr '[:upper:]' '[:lower:]'  # 转小写
echo "hello" | tr '[:lower:]' '[:upper:]'  # 转大写
```

#### 12. 性能测试

```bash
# 测试命令执行时间
time command

# 压力测试（CPU）
stress --cpu 4 --timeout 60s     # 需安装stress

# 内存压力测试
stress --vm 2 --vm-bytes 1G --timeout 60s

# 磁盘写入速度测试
dd if=/dev/zero of=testfile bs=1G count=1 oflag=direct

# 磁盘读取速度测试
dd if=testfile of=/dev/null bs=1M
```

#### 13. 快捷键

```bash
# 命令行快捷键
Ctrl + A          # 移动到行首
Ctrl + E          # 移动到行尾
Ctrl + U          # 删除光标前的所有内容
Ctrl + K          # 删除光标后的所有内容
Ctrl + W          # 删除光标前的一个单词
Ctrl + L          # 清屏（同clear命令）
Ctrl + R          # 搜索历史命令
Ctrl + C          # 终止当前命令
Ctrl + Z          # 暂停当前命令（可用fg恢复）
Ctrl + D          # 退出当前shell
Tab               # 命令/文件名自动补全
Tab Tab           # 显示所有可能的补全

# 终端操作
Ctrl + Shift + C  # 复制
Ctrl + Shift + V  # 粘贴
Ctrl + Shift + T  # 新建标签页
Ctrl + Shift + W  # 关闭标签页
Alt + 数字        # 切换标签页
```

#### 14. 其他技巧

```bash
# 查看命令的位置
which python3
whereis python3

# 查看命令的手册
man command
man ls
info command

# 查看命令的简短说明
whatis command

# 查看命令的用法示例
tldr command                     # 需安装tldr

# 后台任务管理
command &                        # 后台运行
jobs                             # 查看后台任务
fg %1                            # 将后台任务1调到前台
bg %1                            # 将暂停的任务1继续在后台运行

# 创建目录并进入
mkdir -p project/src && cd project/src

# 返回上一个目录
cd -

# 快速备份文件
cp file.txt{,.bak}               # 等同于 cp file.txt file.txt.bak

# 批量重命名
rename 's/\.txt$/.md/' *.txt     # 将所有.txt改为.md

# 查看文件的MD5/SHA校验和
md5sum file.txt
sha256sum file.txt

# 生成随机密码
openssl rand -base64 12
tr -dc A-Za-z0-9 < /dev/urandom | head -c 16

# 监控文件变化
watch -n 1 'ls -lh'              # 每秒执行一次命令
watch -d 'df -h'                 # 高亮显示变化

# 创建快速临时文件
mktemp
mktemp -d                        # 创建临时目录

# 二维码生成（需安装qrencode）
qrencode "Hello World" -o qr.png
```

---

### **十一、Docker 常用命令**

#### 1. 镜像管理

```bash
# 搜索镜像
docker search ubuntu

# 拉取镜像
docker pull ubuntu:22.04

# 查看本地镜像
docker images
docker image ls

# 删除镜像
docker rmi image_name
docker rmi image_id

# 构建镜像
docker build -t myimage:tag .

# 导出/导入镜像
docker save -o myimage.tar myimage:tag
docker load -i myimage.tar

# 查看镜像历史
docker history image_name
```

#### 2. 容器管理

```bash
# 运行容器
docker run -it ubuntu bash              # 交互式运行
docker run -d nginx                     # 后台运行
docker run -p 8080:80 nginx            # 端口映射
docker run -v /host/path:/container/path ubuntu  # 挂载目录
docker run --name mycontainer ubuntu   # 指定容器名

# 查看运行中的容器
docker ps
docker ps -a                           # 查看所有容器（包括停止的）

# 启动/停止/重启容器
docker start container_id
docker stop container_id
docker restart container_id

# 进入运行中的容器
docker exec -it container_id bash
docker attach container_id

# 查看容器日志
docker logs container_id
docker logs -f container_id            # 实时跟踪日志

# 删除容器
docker rm container_id
docker rm -f container_id              # 强制删除运行中的容器

# 查看容器详细信息
docker inspect container_id

# 容器与主机间复制文件
docker cp container_id:/path/to/file /host/path
docker cp /host/file container_id:/path/

# 查看容器资源使用
docker stats
docker stats container_id
```

#### 3. Docker Compose

```bash
# 启动服务
docker-compose up
docker-compose up -d               # 后台运行

# 停止服务
docker-compose down
docker-compose down -v             # 同时删除volumes

# 查看服务状态
docker-compose ps

# 查看日志
docker-compose logs
docker-compose logs -f service_name

# 重启服务
docker-compose restart

# 构建服务
docker-compose build
docker-compose up --build
```

---

### **十二、Git 常用命令**

#### 1. 基础操作

```bash
# 配置
git config --global user.name "Your Name"
git config --global user.email "email@example.com"
git config --list                  # 查看配置

# 初始化仓库
git init

# 克隆仓库
git clone <url>
git clone <url> <directory_name>

# 查看状态
git status
git status -s                      # 简短格式

# 添加文件到暂存区
git add file.txt
git add .                          # 添加所有文件
git add *.py                       # 添加所有Python文件

# 提交
git commit -m "commit message"
git commit -am "message"           # 添加并提交已跟踪的文件
```

#### 2. 分支管理

```bash
# 查看分支
git branch
git branch -a                      # 查看所有分支（包括远程）
git branch -v                      # 查看分支及最后一次提交

# 创建分支
git branch branch_name

# 切换分支
git checkout branch_name
git switch branch_name             # 新命令

# 创建并切换分支
git checkout -b branch_name
git switch -c branch_name

# 合并分支
git merge branch_name

# 删除分支
git branch -d branch_name
git branch -D branch_name          # 强制删除

# 重命名分支
git branch -m old_name new_name
```

#### 3. 远程操作

```bash
# 查看远程仓库
git remote
git remote -v                      # 查看详细信息

# 添加远程仓库
git remote add origin <url>

# 推送
git push origin main
git push -u origin main            # 首次推送并设置上游
git push --all                     # 推送所有分支

# 拉取
git pull origin main
git fetch origin                   # 只获取不合并

# 删除远程分支
git push origin --delete branch_name
```

#### 4. 历史与回退

```bash
# 查看提交历史
git log
git log --oneline                  # 简洁格式
git log --graph --oneline          # 图形化显示
git log -n 5                       # 查看最近5次提交

# 查看文件修改历史
git log -p file.txt

# 查看某次提交的详情
git show commit_id

# 回退到某个版本
git reset --hard commit_id
git reset --soft commit_id         # 保留工作区修改
git reset --mixed commit_id        # 默认选项

# 撤销工作区的修改
git checkout -- file.txt
git restore file.txt               # 新命令

# 撤销暂存区的修改
git reset HEAD file.txt
git restore --staged file.txt      # 新命令
```

---

### **十三、常见问题排查**

#### 1. 端口被占用

```bash
# 查找占用端口的进程
sudo lsof -i :8080
sudo netstat -tunlp | grep 8080

# 杀死进程
sudo kill -9 <PID>
```

#### 2. 磁盘空间不足

```bash
# 查找大文件
sudo du -ah / | sort -rh | head -20

# 清理系统
sudo apt autoremove
sudo apt clean
sudo journalctl --vacuum-size=100M

# 清理Docker
docker system prune -a
```

#### 3. 权限问题

```bash
# 修改文件所有者
sudo chown -R $USER:$USER /path/to/directory

# 修改权限
sudo chmod -R 755 /path/to/directory
```

#### 4. 网络问题

```bash
# 测试DNS
nslookup google.com
dig google.com

# 查看路由
traceroute google.com

# 刷新DNS缓存
sudo systemd-resolve --flush-caches

# 重启网络服务
sudo systemctl restart NetworkManager
```

---

### **十四、学习资源与技巧**

#### 1. 获取帮助

```bash
# 查看命令帮助
man command
command --help
info command

# 在线资源
tldr command                       # 简化版手册（需安装）
```

#### 2. 学习建议

- **从基础开始**：先掌握文件操作、进程管理等基本命令
- **多实践**：在虚拟机或测试环境中练习
- **学会查文档**：善用 `man` 和 `--help`
- **理解而非死记**：理解命令的逻辑，不要死记硬背
- **使用别名**：为常用命令创建简短别名
- **写脚本**：将重复操作写成脚本自动化

#### 3. 安全建议

- 🔴 **危险命令警告**：
  ```bash
  rm -rf /                         # 删除整个系统！
  chmod -R 777 /                   # 让所有文件完全开放！
  dd if=/dev/zero of=/dev/sda      # 清空硬盘！
  ```
- ✅ **安全实践**：
  - 使用 `rm -i` 删除前确认
  - 重要操作前先备份
  - 不要以root身份运行不明脚本
  - 定期更新系统和软件包

---

### **十五、速查表总结**

| 类别 | 常用命令 |
|------|----------|
| **文件操作** | ls, cd, cp, mv, rm, mkdir, touch, find |
| **文本处理** | cat, grep, sed, awk, less, head, tail |
| **系统信息** | top, htop, df, du, free, ps, uname |
| **网络** | ping, ssh, scp, rsync, netstat, wget, curl |
| **权限** | chmod, chown, sudo |
| **压缩** | tar, zip, unzip, gzip |
| **包管理** | apt, dpkg |
| **进程** | ps, kill, jobs, bg, fg |
| **文本编辑** | vim, nano |

---

**提示**：这份笔记涵盖了Linux日常使用的大部分命令，建议收藏并在实践中不断查阅和补充！