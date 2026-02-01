# 在Windows上安装Milvus指南

## 方法一：使用Docker Desktop（推荐）

### 步骤1：安装Docker Desktop

1. **下载Docker Desktop for Windows**
   访问：https://www.docker.com/products/docker-desktop/
   下载Windows版本

2. **安装Docker Desktop**
   - 双击下载的 `Docker Desktop Installer.exe`
   - 勾选 "Use WSL 2 instead of Hyper-V"（推荐）
   - 点击 "Ok" 完成安装

3. **重启计算机**
   安装完成后需要重启

4. **启动Docker Desktop**
   - 从开始菜单启动 Docker Desktop
   - 等待Docker引擎启动（系统托盘图标显示为运行状态）

5. **验证安装**
   打开PowerShell或命令提示符，运行：
   ```bash
   docker --version
   docker ps
   ```

### 步骤2：使用Docker Compose启动Milvus

1. **创建docker-compose.yml文件**

   在项目根目录创建 `docker-compose.yml`：

   ```yaml
   version: '3.5'

   services:
     etcd:
       container_name: milvus-etcd
       image: quay.io/coreos/etcd:v3.5.5
       environment:
         - ETCD_AUTO_COMPACTION_MODE=revision
         - ETCD_AUTO_COMPACTION_RETENTION=1000
         - ETCD_QUOTA_BACKEND_BYTES=4294967296
         - ETCD_SNAPSHOT_COUNT=50000
       volumes:
         - ${DOCKER_VOLUME_DIRECTORY:-./volumes}/etcd:/etcd
       command: etcd -advertise-client-urls=http://127.0.0.1:2379 -listen-client-urls http://0.0.0.0:2379 --data-dir /etcd
       healthcheck:
         test: ["CMD", "etcdctl", "endpoint", "health"]
         interval: 10s
         timeout: 20s
         retries: 3

     minio:
       container_name: milvus-minio
       image: minio/minio:RELEASE.2023-03-20T20-16-18Z
       environment:
         MINIO_ACCESS_KEY: minioadmin
         MINIO_SECRET_KEY: minioadmin
       volumes:
         - ${DOCKER_VOLUME_DIRECTORY:-./volumes}/minio:/minio_data
       command: minio server /minio_data
       healthcheck:
         test: ["CMD", "curl", "-f", "http://localhost:9000/minio/health/live"]
         interval: 30s
         timeout: 20s
         retries: 3

     standalone:
       container_name: milvus-standalone
       image: milvusdb/milvus:latest
       command: ["milvus", "run", "standalone"]
       environment:
         ETCD_ENDPOINTS: etcd:2379
         MINIO_ADDRESS: minio:9000
       volumes:
         - ${DOCKER_VOLUME_DIRECTORY:-./volumes}/milvus:/var/lib/milvus
       ports:
         - "19530:19530"
         - "9091:9091"
       depends_on:
         - "etcd"
         - "minio"

   networks:
     default:
       name: milvus
   ```

2. **启动Milvus**
   在项目根目录打开PowerShell，运行：
   ```bash
   docker-compose up -d
   ```

3. **验证Milvus运行状态**
   ```bash
   docker ps
   ```
   应该看到3个容器在运行：
   - milvus-etcd
   - milvus-minio
   - milvus-standalone

---

## 方法二：使用单条Docker命令（简单快速）

如果您已经安装了Docker，可以直接运行：

```bash
docker run -d --name milvus-standalone \
  -p 19530:19530 \
  -p 9091:9091 \
  -v $(pwd)/milvus_data:/var/lib/milvus \
  milvusdb/milvus:latest
```

**注意：** 此方法仅用于开发测试，生产环境建议使用Docker Compose。

---

## 方法三：使用Milvus Standalone安装程序（无需Docker）

如果不想安装Docker，可以使用Milvus的Windows安装程序：

1. **下载Milvus Standalone**
   访问：https://github.com/milvus-io/milvus/releases

2. **安装**
   下载 `milvus-standalone-windows.zip`
   解压到本地目录

3. **运行**
   ```bash
   cd milvus-standalone
   .\milvus.exe run standalone
   ```

---

## 验证Milvus安装

安装完成后，运行以下命令测试连接：

```bash
# 检查端口是否监听
netstat -an | findstr 19530

# 或使用Python测试
python -c "from pymilvus import connections; connections.connect('default', host='localhost', port='19530'); print('Milvus连接成功!')"
```

---

## 常用Docker命令

```bash
# 启动Milvus
docker-compose up -d

# 停止Milvus
docker-compose down

# 查看日志
docker logs milvus-standalone

# 重启Milvus
docker-compose restart

# 删除所有数据（危险！）
docker-compose down -v
```

---

## 故障排查

### 问题1：Docker Desktop无法启动
- 确保启用了WSL 2
- 检查系统虚拟化是否启用
- 更新Windows到最新版本

### 问题2：端口19530被占用
```bash
# 查看占用端口的进程
netstat -ano | findstr :19530

# 终止进程
taskkill /PID <进程ID> /F
```

### 问题3：内存不足
- Docker Desktop设置中增加分配的内存
- 建议至少分配4GB内存

---

## 下一步

安装完成后，运行完整测试：

```bash
# 运行所有测试
pytest rag_project/tests/ -v

# 只运行Milvus相关测试
pytest rag_project/tests/test_milvus_manager.py -v
pytest rag_project/tests/test_pipeline.py -v
```

---

## 相关链接

- Milvus官方文档：https://milvus.io/docs
- Docker Desktop下载：https://www.docker.com/products/docker-desktop
- PyMilvus文档：https://milvus.io/api-reference/pymilvus/v2.3.x/About.md
