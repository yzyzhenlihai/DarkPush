## 项目整体架构
#### 项目文件目录
```
├── 📄 .env.example                    # 环境变量示例文件
├── 📄 .gitignore                      # Git忽略文件配置
├── 📄 Dockerfile.backend               # 后端Docker镜像构建文件
├── 📄 LICENSE                          # 项目许可证
├── 📄 README.md                        # 项目说明文档
├── 📄 config.py                        # 系统配置文件
├── 📄 main.py                          # 主程序入口文件
├── 📄 requirements.txt                 # Python依赖包列表
├── 📄 start.bat / start.sh            # 项目启动脚本（Windows/Linux）
├── 📄 stop.bat / stop.sh              # 项目停止脚本（Windows/Linux）
├── 📄 view.png                         # 项目界面截图
│
├── 📁 data\                            # 数据文件夹
│   ├── 📁 cypher\                      # Neo4j图数据库文件
│   │   ├── 📄 neo4j_import.cypher     # 数据导入脚本
│   │   ├── 📄 nodes.csv               # 节点数据文件
│   │   └── 📄 relationships.csv       # 关系数据文件
│   ├── 📁 dishes\                      # 菜谱分类数据
│   │   ├── 📁 aquatic\                 # 水产类
│   │   ├── 📁 breakfast\               # 早餐类
│   │   ├── 📁 condiment\               # 调料类
│   │   ├── 📁 dessert\                 # 甜品类
│   │   ├── 📁 drink\                   # 饮品类
│   │   ├── 📁 meat_dish\               # 荤菜类
│   │   ├── 📁 semi-finished\            # 半成品类
│   │   ├── 📁 soup\                    # 汤类
│   │   ├── 📁 staple\                  # 主食类
│   │   ├── 📁 template\                # 模板类
│   │   └── 📁 vegetable_dish\          # 素菜类
│   ├── 📄 docker-compose.yml           # 数据服务Docker配置
│   ├── 📄 recipes_with_images.json     # 带图片的菜谱数据
│   └── 📁 tips\                        # 烹饪技巧文档
│       ├── 📁 advanced\                 # 高级技巧
│       ├── 📁 learn\                    # 学习资料
│       ├── 📄 厨房准备.md              # 厨房准备指南
│       ├── 📄 如何选择现在吃什么.md    # 选择指南
│       └── 📄 食材相克与禁忌.md        # 食材搭配禁忌
│
├── 📁 frontend\                        # 前端应用
│   ├── 📄 .eslintrc.json              # ESLint代码检查配置
│   ├── 📄 Dockerfile                  # 前端Docker镜像构建文件
│   ├── 📄 next-env.d.ts               # Next.js类型声明
│   ├── 📄 next.config.js              # Next.js配置文件
│   ├── 📄 package-lock.json           # NPM依赖锁定文件
│   ├── 📄 package.json                # NPM包配置
│   ├── 📄 postcss.config.js           # PostCSS配置
│   ├── 📄 tailwind.config.js          # Tailwind CSS配置
│   ├── 📄 tsconfig.json               # TypeScript配置
│   ├── 📁 public\                      # 静态资源
│   └── 📁 src\                         # 源代码
│       ├── 📁 app\                     # Next.js App Router
│       ├── 📁 components\              # React组件
│       ├── 📁 hooks\                   # 自定义React Hooks
│       ├── 📁 lib\                     # 工具库
│       ├── 📁 store\                   # 状态管理
│       └── 📁 types\                   # TypeScript类型定义
│
├── 📁 nginx\                           # Nginx配置
│   └── 📄 nginx.conf                   # Nginx反向代理配置
│
├── 📁 public\                          # 公共静态资源
│   └── 📄 og.png                       # 社交媒体预览图
│
├── 📁 rag_modules\                     # RAG核心模块（Python）
│   ├── 📄 __init__.py                 # 模块初始化
│   ├── 📄 generation_integration.py   # 生成集成模块
│   ├── 📄 graph_data_preparation.py   # 图数据预处理
│   ├── 📄 graph_indexing.py           # 图索引构建
│   ├── 📄 graph_rag_retrieval.py      # 图RAG检索
│   ├── 📄 hybrid_retrieval.py         # 混合检索
│   ├── 📄 intelligent_query_router.py # 智能查询路由
│   ├── 📄 milvus_index_construction.py # Milvus向量索引
│   ├── 📄 recipe_recommendation.py     # 菜谱推荐
│   ├── 📄 session_cache_manager.py     # 会话缓存管理
│   └── 📄 web_service_handler.py       # Web服务处理
│
└── 📄 docker-compose.yml              # 主Docker Compose配置
```
#### 技术栈概览
前端技术栈：
- Next.js 14 - React全栈框架
- TypeScript - 类型安全
- Tailwind CSS - 样式框架
- Zustand - 状态管理
- Framer Motion - 动画效果
- React Markdown - Markdown渲染
后端技术栈：
- Python 3.11 + Flask - Web服务
- Neo4j - 图数据库（存储菜谱关系）
- Milvus - 向量数据库（语义搜索）
- Docker - 容器化部署
- Nginx - 反向代理
AI/ML技术：
- 图RAG (Graph RAG) - 图检索增强生成
- 混合检索 - 向量+关键词+图检索
- BGE嵌入模型 - 中文文本向量化
- Moonshot AI - 大语言模型
## 学习路径规划
### 第一阶段：基础环境搭建
1. Docker基础
```bash
# 学习目标：理解容器化概念
docker --version
docker-compose --version
docker run hello-world
```
2. 项目启动和探索
```bash
# 克隆项目
git clone https://github.com/FutureUnreal/What-to-eat-today.git
cd What-to-eat-today

# Windows用户
start.bat

# Linux/macOS用户
chmod +x start.sh stop.sh
./start.sh
```
3. 访问各个服务熟悉功能
- 主应用： http://localhost
- 前端： http://localhost:3000
- 后端API： http://localhost:8000
- Neo4j控制台： http://localhost:7474
### 第二阶段：前端技术
1. **Next.js基础**
- 学习Next.js路由系统
- 理解React Server Components
- 掌握TypeScript在React中的应用
2. **前端关键文件学习**
```TypeScript
// 重点文件：frontend/src/app/page.tsx
// 学习：页面组件结构、状态管理

// 重点文件：frontend/src/store/useAppStore.ts
// 学习：Zustand状态管理、数据持久化

// 重点文件：frontend/src/lib/api.ts
// 学习：API调用封装、错误处理
```
3. **UI组件库**
- Tailwind CSS实用类
- Radix UI组件使用
- 响应式设计实现

### 第三阶段：后端和AI技术
1. Python后端基础
```python
# 重点文件：main.py
# 学习：Flask应用结构、模块化设计

# 重点文件：rag_modules/web_service_handler.py
# 学习：RESTful API设计、流式响应
```
2. 图数据库Neo4j
```python
# 学习Cypher查询语言
# 查看文件：data/cypher/neo4j_import.cypher

# 理解图数据模型
# Recipe -> Ingredient -> CookingStep 的关系
```
3. 向量数据库Milvus
```python
# 学习文件：rag_modules/milvus_index_construction.py
# 理解：文本向量化、相似度搜索、索引构建
```

### 第四阶段：RAG核心技术
1. 混合检索系统
```python
# 学习文件：rag_modules/hybrid_retrieval.py
# 核心概念：
# - 双层检索范式（实体级 + 主题级）
# - 关键词提取和匹配
# - 图结构+向量检索结合
# - Round-robin轮询合并策略
```
2. 图RAG检索
```python
# 学习文件：rag_modules/graph_rag_retrieval.py
# 核心概念：
# - 图查询结构(GraphQuery)
# - 图路径分析(GraphPath)
# - 多跳遍历、子图提取
# - 关系推理
```
3. 智能查询路由
```python
# 学习文件：rag_modules/intelligent_query_router.py
# 核心概念：
# - 查询复杂度分析
# - 关系密集度计算
# - 自动路由策略选择
# - 置信度评估
```
### 第五阶段：系统集成和优化
1. 会话管理和缓存
```python
# 学习文件：rag_modules/session_cache_manager.py
# 理解：语义缓存、会话上下文管理
```
2. 菜谱推荐算法
```python
# 学习文件：rag_modules/recipe_recommendation.py
# 理解：个性化推荐、协同过滤
```
3. 生成集成
```python
# 学习文件：rag_modules/generation_integration.py
# 理解：提示工程、答案生成优化
```



## 分工情况

#### 成员 A：核心对话与 RAG 检索 (The "Answer" Owner)【李子民】

**核心使命**：负责用户提问后的核心回答逻辑，解决“怎么做”、“是什么”的问题。 **关注点**：准确性、RAG 检索链路、聊天体验。

- **前端 (Next.js)**:
  - 负责 **Chat 界面组件** (`components/Chat/` - 需新建)。
  - 负责 **Markdown 渲染器**，优化 AI 返回的菜谱格式（处理加粗、列表）。
  - 实现 **流式响应 (Streaming)** 的前端接收逻辑。
- **后端 (Python)**:
  - 负责 `rag_modules/intelligent_query_router.py`：判断用户意图。
  - 负责 `rag_modules/hybrid_retrieval.py` 和 `graph_rag_retrieval.py`：核心检索逻辑。
  - 负责 `rag_modules/generation_integration.py`：与大模型交互的 Prompt 编写。
- **数据 (Milvus & Neo4j)**:
  - 负责 **Milvus 向量库**的构建与查询 (`milvus_index_construction.py`)。
  - 负责回答 factual 问题（如“番茄炒蛋怎么做？”）时的图数据提取。

####  成员 B：推荐系统与图谱探索 (The "Discovery" Owner)【余正阳】

**核心使命**：负责“不知道吃什么”时的推荐逻辑，以及菜谱详情的深度展示。 **关注点**：个性化、图谱关系可视化、菜谱数据管理。

- **前端 (Next.js)**:
  - 负责 **首页推荐卡片** 和 **菜谱详情页**。
  - 负责 **知识图谱可视化**（展示“食材相克”、“营养成分”等节点关系）。
  - 负责菜谱分类浏览页面 (`dishes/` 数据的展示)。
- **后端 (Python)**:
  - 负责 `rag_modules/recipe_recommendation.py`：编写基于图的推荐算法。
  - 开发 `/recommend` 相关的 API 接口。
  - 处理 `dishes/` 目录下所有 JSON/Markdown 数据的解析逻辑。
- **数据 (Neo4j)**:
  - 是 **Neo4j** 的主要负责人。
  - 负责 `data/cypher/neo4j_import.cypher` 的维护。
  - 负责设计图谱 schema（节点：菜品、食材、口味；关系：包含、克制、适合）。



#### 成员 C：系统架构与会话管理 (The "Platform" Owner)【苑震坤】

**核心使命**：负责系统的地基、用户状态管理以及运维部署。 **关注点**：稳定性、历史记录、环境搭建、多媒体资源。

- **前端 (Next.js)**:
  - 负责 **App Shell**（整体 Layout、侧边栏、导航）。
  - 负责 **状态管理 Store** (`store/`) 的搭建，处理 User Session。
  - 负责 **历史记录列表** UI。
  - 处理图片资源的加载与展示（`public/` 和 `view.png` 等）。
- **后端 (Python)**:
  - 负责 **项目入口** (`main.py`, `config.py`) 的稳定性。
  - 负责 `rag_modules/session_cache_manager.py`：实现多轮对话的上下文记忆。
  - 负责 `rag_modules/web_service_handler.py`：统一 API 的错误处理和日志。
- **运维 (DevOps)**:
  - 负责 **Docker 环境** (`Dockerfile`, `docker-compose.yml`)。
  - 负责 **Nginx 配置** 和 启动脚本 (`start.sh`).
  - 负责 Git 仓库管理（`.gitignore`）和依赖管理 (`requirements.txt`).
