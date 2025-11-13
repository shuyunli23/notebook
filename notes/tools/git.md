## **Git 核心概念与实用指南**

### **一、Git 是什么？**

Git 是一个**分布式版本控制系统**，主要用于：
*   **追踪文件变更**：记录每一次代码的修改历史。
*   **版本回退**：可以轻松回到过去的任意一个版本。
*   **团队协作**：支持多人同时在同一个项目上工作，并能高效地合并代码。

---

### **二、首次安装与配置 (一次性操作)**

#### 1. 安装 Git

*   **Ubuntu / Debian**: `sudo apt install git`
*   **CentOS / RHEL**: `sudo yum install git`
*   **macOS**: `brew install git` (推荐) 或从官网下载
*   **Windows**: 从 [Git for Windows](https://git-scm.com/download/win) 官网下载安装。

#### 2. 配置个人信息

安装后，首先设置你的身份信息，这会记录在每一次提交中。

```bash
# 设置你的用户名
git config --global user.name "你的名字"

# 设置你的邮箱
git config --global user.email "你的邮箱@example.com"

# (可选) 检查配置是否成功
git config --global --list
```

---

### **三、核心工作流程 (本地操作)**

#### 1. 创建仓库

*   **方式一：初始化一个新项目**
    ```bash
    mkdir my-project
    cd my-project
    git init  # 在当前目录生成 .git 文件夹，代表仓库创建成功
    ```
*   **方式二：克隆一个远程项目**
    ```bash
    git clone https://github.com/username/repository.git
    ```

#### 2. 日常提交流程

Git 的核心是“三区”概念：**工作区 -> 暂存区 -> 本地仓库**。

![Git Three Areas](https://git-scm.com/images/about/areas.png)

1.  **`git status`**: 查看当前仓库状态。这是最常用的命令，可以告诉你哪些文件被修改了、哪些文件还未被跟踪。

2.  **`git add <文件名>`**: 将工作区的修改添加到**暂存区 (Staging Area)**。
    ```bash
    # 添加单个文件
    git add main.py

    # 添加所有已修改和新文件
    git add .
    ```

3.  **`git commit -m "提交说明"`**: 将暂存区的所有内容提交到**本地仓库**，形成一个历史版本。
    ```bash
    git commit -m "feat: 添加用户登录功能"
    ```
    > **提示**: 提交说明 (commit message) 应该清晰、简洁地描述本次修改的内容。

---

### **四、远程仓库协作 (GitHub / Gitee)**

#### 1. 关联远程仓库

```bash
# 查看已有关联
git remote -v

# 添加一个新的远程仓库，通常命名为 origin
git remote add origin https://github.com/用户名/仓库名.git
```

#### 2. 推送与拉取

*   **`git push`**: 将本地仓库的提交推送到远程仓库。
    ```bash
    # 第一次推送时，需要指定分支并建立关联
    git push -u origin main

    # 之后可以直接使用
    git push
    ```
*   **`git pull`**: 从远程仓库拉取最新的代码，并与本地分支合并。
    ```bash
    git pull
    ```

#### 3. ⚠️ **重要：关于 `git push` 的身份验证**

当你 `git push` 时，如果提示输入用户名和密码：
```
Username for 'https://github.com': your-username
Password for 'https://your-username@github.com':
```
**注意：这里的 Password 不能再输入你的 GitHub 登录密码！**

自 2021 年起，GitHub 不再支持密码验证。你必须使用 **Personal Access Token (PAT)**。

**如何生成和使用 PAT：**
1.  登录 GitHub → 右上角头像 → **Settings**。
2.  左侧菜单 → **Developer settings** → **Personal access tokens** → **Tokens (classic)**。
3.  点击 **Generate new token (classic)**。
4.  设置一个名称，选择有效期，并勾选 `repo` 权限。
5.  生成 Token，**立即复制并妥善保存**（它只显示一次）。
6.  在 `git push` 提示输入 Password 时，**粘贴这个 Token**。

> **推荐**: 为了避免每次都输入，建议配置 **SSH 密钥**进行免密登录，这是更安全、更便捷的长久之计。

---

### **五、分支 (Branch) 管理**

分支是 Git 的核心功能，它允许你独立开发新功能，而不影响主线代码。

*   **`git branch`**: 查看所有本地分支。
*   **`git checkout -b <新分支名>`**: 创建一个新分支，并立即切换过去。
*   **`git checkout <分支名>`**: 切换到已存在的分支。
*   **`git merge <要合并的分支名>`**: 将指定分支的修改合并到**当前**所在的分支。

**标准开发流程：**
```bash
# 1. 从主分支创建一个新功能分支
git checkout -b feature/user-profile

# 2. 在新分支上进行开发和提交...
git add .
git commit -m "feat: 完成用户个人资料页"

# 3. 开发完成后，切回主分支
git checkout main

# 4. 将功能分支合并到主分支
git merge feature/user-profile

# 5. (可选) 删除已经合并的功能分支
git branch -d feature/user-profile
```

---

### **六、查看历史与撤销操作**

#### 1. 查看提交历史 (`git log`)

*   **`git log`**: 显示详细的提交历史。
*   **`git log --oneline`**: 以单行简洁模式显示。
*   **`git log --graph --oneline --all`**: 图形化地展示所有分支的历史。
*   **`git show <commit_id>`**: 查看某次提交的具体修改内容。
*   **`git show <commit_id>:<文件路径>`**: 查看某次提交时，某个文件的完整内容。

#### 2. 撤销操作

*   **撤销工作区的修改**:
    ```bash
    git checkout -- <文件名>
    ```
*   **将文件从暂存区撤回**:
    ```bash
    git reset HEAD <文件名>
    ```
*   **回退到某个历史版本** (危险操作，谨慎使用):
    ```bash
    # --soft: 仅移动HEAD指针，保留工作区和暂存区的修改
    git reset --soft <commit_id>

    # --hard: 彻底回退，丢弃之后的所有修改
    git reset --hard <commit_id>
    ```

---

### **七、实战演练：从零开始管理一个项目**

1.  **本地初始化**
    ```bash
    mkdir my-flask-app && cd my-flask-app
    git init
    echo "print('Hello World')" > app.py
    ```

2.  **创建 `.gitignore` 文件** (忽略不必要的文件)
    ```bash
    echo "__pycache__/\nenv/\n.vscode/" > .gitignore
    ```

3.  **首次提交**
    ```bash
    git add .
    git commit -m "Initial commit: Add app.py and .gitignore"
    ```

4.  **连接到 GitHub** (假设已在 GitHub 创建空仓库)
    ```bash
    git remote add origin https://github.com/your-username/my-flask-app.git
    git branch -M main  # 确保本地分支名为 main
    ```

5.  **首次推送**
    ```bash
    git push -u origin main
    # 按提示输入用户名和 Personal Access Token
    ```

6.  **日常开发**
    ```bash
    # ...修改代码...
    git add .
    git commit -m "feat: Implement new feature"
    git push
    ```

---

### **八、Git 命令速查表**

| 功能 | 命令 |
| :--- | :--- |
| **基础配置** | `git config --global user.name "..."`<br>`git config --global user.email "..."` |
| **创建仓库** | `git init`<br>`git clone <url>` |
| **日常提交** | `git status`<br>`git add .`<br>`git commit -m "..."` |
| **远程同步** | `git push`<br>`git pull`<br>`git remote -v` |
| **分支操作** | `git branch`<br>`git checkout -b <name>`<br>`git checkout <name>`<br>`git merge <name>` |
| **历史查看** | `git log --oneline`<br>`git show <commit_id>` |