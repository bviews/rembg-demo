# PDS 背景去除与产品分割 API

使用 RMBG-2.0 最新 AI 背景去除算法进行产品包装的智能分割处理。

## 🚀 功能特性

- 基于 RMBG-2.0 最新背景去除模型
- 智能产品分割和裁剪
- 高精度 mask 生成
- CPU/GPU 自动适配

## 📋 环境要求

```bash
pip install -r requirements.txt
```

## 🔑 RMBG-2.0 权限配置

### 1. 申请访问权限
访问 [briaai/RMBG-2.0](https://huggingface.co/briaai/RMBG-2.0) 申请模型访问权限

### 2. 获取 Access Token
1. 前往 [Hugging Face Settings](https://huggingface.co/settings/tokens)
2. 创建新的 Access Token (选择 "read" 权限)
3. 复制生成的 token (格式: `hf_xxxxxxxx`)

### 3. 配置环境变量

**方法一：临时设置**
```bash
export HUGGINGFACE_HUB_TOKEN="hf_xxxxxxxx"
python segment.py
```

**方法二：永久设置**
```bash
# 添加到 ~/.bashrc 或 ~/.zshrc
echo 'export HUGGINGFACE_HUB_TOKEN="hf_xxxxxxxx"' >> ~/.bashrc
source ~/.bashrc
```

**方法三：创建 .env 文件**
```bash
# 在项目根目录创建 .env 文件
echo "HUGGINGFACE_HUB_TOKEN=hf_xxxxxxxx" > .env
```

## 📖 使用方法

### 基本使用
```bash
# 配置 token 后运行
export HUGGINGFACE_HUB_TOKEN="hf_xxxxxxxx"
python segment.py
```

### 代码示例
```python
from segment import process_image_with_rmbg

# 处理图片
result_path, mask_path, product_count = process_image_with_rmbg(
    image_path="demo.jpg",
    output_dir="output_crops",
    min_area=500
)
```

## 📁 输出文件

运行后会在 `output_crops/` 目录生成：

- `no_background.png` - 去除背景的完整图片
- `mask.png` - 生成的 mask 图片
- `product_0.png`, `product_1.png`, ... - 分割的产品图片

## ⚙️ 参数配置

在 `segment.py` 中可以修改：

```python
# 最小面积阈值（过滤小噪点）
min_area_threshold = 500

# 输入输出路径
input_image = "demo.jpg"
output_directory = "output_crops"
```

## 🔧 故障排除

### 权限错误（401/403）
```
🔑 权限错误: 请设置正确的 Hugging Face token:
   export HUGGINGFACE_HUB_TOKEN='hf_xxxxxxxx'
   或切换到无需权限的模型 (如 briaai/RMBG-1.4)
```

**解决方案：**
1. 确保已申请 RMBG-2.0 访问权限
2. 检查 token 是否正确设置
3. 访问 [RMBG-2.0 模型页面](https://huggingface.co/briaai/RMBG-2.0) 申请权限

### 网络连接问题
首次运行需要下载模型文件，请确保网络连接稳定。

### 内存不足
RMBG-2.0 需要较多内存，建议：
- 使用 GPU 加速
- 确保至少 4GB 可用内存

## 📝 更新日志

- ✅ 专注于 RMBG-2.0 模型，简化代码结构
- ✅ 支持从环境变量读取 Hugging Face token
- ✅ 自动检测权限状态和用户指导
- ✅ 增强错误处理和详细提示
- ✅ 支持 CPU/GPU 自动选择
