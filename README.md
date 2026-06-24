# TraceMark - AI 智能图像水印系统

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104-green.svg)](https://fastapi.tiangolo.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

基于 Qwen-VL 多模态大模型的智能数字水印系统，支持自适应强度嵌入、AI 防伪检测等功能。

---

## 🌟 功能特性

- **智能水印嵌入**：基于 LLM 的图像内容分析，自动选择最优嵌入策略
- **自适应强度**：根据图像复杂度动态调整水印强度
- **双算法支持**：LSB 空域水印（PNG）+ DCT 频域水印（JPG）
- **批量处理**：支持多张图片批量嵌入水印
- **AI 防伪检测**：检测图像篡改并生成综合防伪报告
- **操作日志**：完整的操作记录与统计功能

---

## 📋 环境要求

| 依赖 | 版本 | 说明 |
|------|------|------|
| Python | ≥ 3.10 | 编程语言 |
| FastAPI | ≥ 0.100 | Web 框架 |
| OpenCV | ≥ 4.8 | 图像处理 |
| Pillow | ≥ 10.0 | 图像读写 |
| openai | ≥ 1.0 | 大模型 API 调用 |
| scipy | ≥ 1.10 | DCT 变换 |

---

## 🚀 快速开始

### 1. 克隆项目

```bash
git clone https://github.com/yourusername/TraceMark.git
cd TraceMark
```

### 2. 创建虚拟环境

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python -m venv venv
source venv/bin/activate
```

### 3. 安装依赖

```bash
pip install -r requirements.txt
```

### 4. 配置 API Key

#### 方式一：环境变量（推荐）

**Windows PowerShell：**
```powershell
$env:BAILIAN_API_KEY="sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxx"
```

**Linux/Mac：**
```bash
export BAILIAN_API_KEY="sk-xxxxxxxxxxxxxxxxxxxxxxxxx"
```

#### 方式二：配置文件

创建 `backend/.env` 文件：
```env
BAILIAN_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

### 5. 启动服务

```bash
cd backend
python main.py
```

访问地址：
- **前端页面**: http://localhost:8000
- **API 文档**: http://localhost:8000/docs

---

## 📖 使用说明

### 1. 单张图片水印嵌入

1. 在首页选择图片（支持 PNG/JPG 格式）
2. 输入自定义 UID（如 U2025001）
3. 选择是否启用自适应强度
4. 点击"嵌入水印"按钮

### 2. 批量水印嵌入

1. 在"批量嵌入水印"区域选择多张图片
2. 输入 UID 前缀（如 U2025）
3. 点击"批量嵌入水印"按钮
4. 系统会自动生成 UID-001, UID-002, ...

### 3. 水印验证

1. 在"水印验证"区域上传含水印的图片
2. 点击"验证水印"按钮
3. 系统会自动提取并显示 UID

### 4. AI 防伪检测

1. 在"AI 防伪检测"区域上传图片
2. 点击"开始检测"按钮
3. 系统会生成综合防伪报告

---

## 🔌 API 接口

| 接口 | 方法 | 功能 |
|------|------|------|
| `/watermark` | POST | 智能嵌入水印 |
| `/adaptive-watermark` | POST | 自适应强度嵌入 |
| `/verify` | POST | 提取水印 |
| `/detect` | POST | AI 防伪检测 |
| `/batch-watermark` | POST | 批量嵌入水印 |
| `/watermark-history` | GET | 查询操作历史 |
| `/watermark-stats` | GET | 获取统计数据 |
| `/clear-history` | DELETE | 清除操作历史 |

---

## 🏗️ 项目结构

```
TraceMark/
├── backend/
│   ├── main.py              # FastAPI 主入口
│   ├── database.py          # SQLite 数据库操作 
│   ├── static/
│   │   └── index.html       # 前端页面
│   ├── watermarking/
│   │   ├── lsb.py           # LSB 水印算法
│   │   ├── dct.py           # DCT 水印算法
│   │   ├── llm_strategist.py # LLM 策略生成
│   │   ├── adaptive_watermark.py # 自适应水印
│   │   ├── mask_generator.py # 显著性掩码生成
│   │   ├── prompt_template.py # 提示词模板
│   │   └── u2net.pth        # U²-Net 模型权重
└── README.md                # 项目说明
```

---

## 🎯 水印算法说明

### LSB 空域水印（PNG）
- 基于最低有效位嵌入
- 支持三级强度：low/medium/high
- 无损图像隐蔽性好

### DCT 频域水印（JPG）
- 基于 8×8 分块 DCT 变换
- 嵌入中频系数，兼顾鲁棒性与隐蔽性
- 抗 JPEG 压缩能力强

---

## ⚙️ 配置说明

### 环境变量

| 变量名 | 说明 | 默认值 |
|--------|------|--------|
| BAILIAN_API_KEY | 阿里云百炼 API Key | - |
| BAILIAN_PLATFORM | 平台标识（可选） | bailian |

---

## ❓ 常见问题

### Q1: 启动时报错 "未设置 BAILIAN_API_KEY"

**解决方案**：
```powershell
# Windows
$env:BAILIAN_API_KEY="sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxx"
```

### Q2: 批量水印只能选择一张图片

**解决方案**：按住 Ctrl/Cmd 键选择多个文件，或直接拖拽文件。

---
