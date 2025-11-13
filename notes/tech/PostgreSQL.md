## **指南：使用 Docker & pgvector 管理 PostgreSQL 数据库**

本文档记录了如何使用 `docker-compose` 快速部署两个独立的 PostgreSQL (pgvector) 服务，并演示了如何导入 SQL 数据及进行验证。

### **一、项目文件结构**

在开始之前，请确保你的项目目录结构如下。`.sql` 文件是需要导入的数据库表结构和数据。

```
.
├── docker-compose.yml          # Docker 服务编排文件
└── otci_data_20250903/         # SQL 数据目录
    ├── kb_chunks.sql
    ├── kb_documents.sql
    └── prompt_versions.sql
```

### **二、Docker Compose 配置 (`docker-compose.yml`)**

此配置定义了两个独立的 PostgreSQL 服务，分别用于生产和测试，它们使用不同的端口和数据卷，互不干扰。

```yaml
version: '3.8'

services:
  # --- 服务 1: 主要数据库 ---
  postgres1:
    image: pgvector/pgvector:pg16          # 使用支持 pgvector 扩展的 PG16 镜像
    container_name: pgvector-container-1   # 自定义容器名称
    environment:
      - POSTGRES_USER=postgres             # 数据库用户名
      - POSTGRES_PASSWORD=postgres@pass    # 数据库密码
      - POSTGRES_DB=postgres               # 默认数据库名
      - PGDATA=/var/lib/postgresql/data/pgdata # 数据存储路径
    volumes:
      # 将宿主机的 /mnt/pgvector 目录映射到容器的数据目录，实现数据持久化
      - /mnt/pgvector:/var/lib/postgresql/data
      # (推荐) 将 SQL 文件目录挂载到初始化目录，用于首次启动时自动导入
      - ./otci_data_20250903:/docker-entrypoint-initdb.d
    ports:
      # 将宿主机的 5433 端口映射到容器的 5432 端口
      - "5433:5432"
    restart: always                        # 容器退出后总是自动重启
    shm_size: "1g"                         # 增加共享内存大小，有助于性能

  # --- 服务 2: 测试数据库 ---
  postgres2:
    image: pgvector/pgvector:pg16
    container_name: pgvector-container-2
    environment:
      - POSTGRES_USER=postgres
      - POSTGRES_PASSWORD=postgres@pass
      - POSTGRES_DB=postgres2              # 为测试库指定一个不同的数据库名
    volumes:
      - /mnt/pgvector2:/var/lib/postgresql/data
    ports:
      # 使用 5434 端口，避免与服务1冲突
      - "5434:5432"
    restart: always
    shm_size: "1g"
```

### **三、启动与管理数据库服务**

1.  **启动所有服务 (后台模式)**
    ```bash
    docker-compose up -d
    ```

2.  **检查容器运行状态**
    ```bash
    docker ps
    ```
    你应该能看到 `pgvector-container-1` 和 `pgvector-container-2` 两个容器正在运行。

3.  **停止并移除服务**
    ```bash
    docker-compose down
    ```

### **四、连接数据库**

你可以使用任何数据库客户端或 `psql` 命令行工具进行连接。

*   **连接 `postgres1` (端口 5433):**
    ```bash
    psql -h localhost -p 5433 -U postgres -d postgres
    ```
*   **连接 `postgres2` (端口 5434):**
    ```bash
    psql -h localhost -p 5434 -U postgres -d postgres2
    ```
    > 提示：执行命令后，会要求输入密码，即 `postgres@pass`。

### **五、导入 SQL 数据**

#### **方法一：自动初始化 (推荐)**

如 `docker-compose.yml` 所示，通过将包含 `.sql` 文件的目录挂载到容器的 `/docker-entrypoint-initdb.d` 目录，Docker 会在**容器第一次创建和启动时**，自动按字母顺序执行目录下的所有 `.sh`, `.sql`, `.sql.gz` 文件。

这是最简单、最适合初始化的方法。如果你已经启动过容器，需要先执行 `docker-compose down -v` 删除数据卷，再重新 `docker-compose up -d` 才能触发。

#### **方法二：手动导入到正在运行的容器**

如果你的容器已经运行，或者你需要导入新的 SQL 文件，可以使用此方法。

1.  **将 SQL 文件复制到容器内 (如果 `docker-compose.yml` 没有挂载)**
    ```bash
    # 语法: docker cp <本地路径> <容器名>:<容器内路径>
    docker cp ./otci_data_20250903/kb_documents.sql pgvector-container-1:/tmp/
    docker cp ./otci_data_20250903/kb_chunks.sql pgvector-container-1:/tmp/
    ```

2.  **进入容器的 Bash 环境**
    ```bash
    docker exec -it pgvector-container-1 bash
    ```

3.  **在容器内执行导入命令**
    ```bash
    # 语法: psql -U <用户名> -d <数据库名> < <SQL文件路径>
    psql -U postgres -d postgres < /tmp/kb_documents.sql
    psql -U postgres -d postgres < /tmp/kb_chunks.sql
    # 执行完毕后，输入 exit 退出容器
    ```

### **六、验证数据导入**

连接到 `postgres1` 数据库后，你可以使用 `psql` 的元命令和标准 SQL 查询来验证数据。

```bash
# 连接到数据库
psql -h localhost -p 5433 -U postgres -d postgres
```

在 `psql` 交互环境中执行以下命令：

```sql
-- 元命令 (以 \ 开头)
\l              -- List: 查看所有数据库
\dt             -- Describe Tables: 查看当前数据库的所有表
\d kb_documents -- Describe: 查看 'kb_documents' 表的结构

-- SQL 查询
SELECT COUNT(*) FROM kb_documents;
SELECT COUNT(*) FROM kb_chunks;
SELECT * FROM kb_documents LIMIT 5; -- 查看表的前5行数据
```

> **💡 psql 分页器提示:** 当查询结果过长时，`psql` 会自动进入分页模式。
> *   **`空格键`**: 向下翻一页。
> *   **`Enter键`**: 向下滚动一行。
> *   **`q`**: 退出分页器，返回 `psql` 提示符。

---

### **附录：常用命令速查表 (Cheat Sheet)**

#### **Docker & Docker Compose**

| 命令 | 作用 |
| :--- | :--- |
| `docker-compose up -d` | 在后台创建并启动服务 |
| `docker-compose down` | 停止并移除服务 |
| `docker-compose down -v` | 停止、移除服务并删除数据卷 |
| `docker ps` | 查看正在运行的容器 |
| `docker logs <container_name>` | 查看容器的日志 |
| `docker exec -it <container_name> bash` | 进入容器的交互式 Shell |
| `docker cp <src> <container>:<dest>` | 在宿主机和容器之间复制文件 |

#### **psql (PostgreSQL 命令行)**

| 命令 | 作用 |
| :--- | :--- |
| `psql -h <host> -p <port> -U <user> -d <db>` | 连接到数据库 |
| `\l` | 列出所有数据库 |
| `\c <database_name>` | 连接到同一个实例下的另一个数据库 |
| `\dt` | 列出当前数据库的表 |
| `\d <table>` | 显示表的结构 (列、类型等) |
| `\d+ <table>` | 显示更详细的表信息 |
| `\dn` | 列出所有 schema |
| `\q` | 退出 `psql` |