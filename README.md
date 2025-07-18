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

### API 服务启动

```bash
# 配置 token 后启动 API 服务
export HUGGINGFACE_HUB_TOKEN="hf_xxxxxxxx"
python main.py
```

API 服务将在 `http://localhost:8000` 启动

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

## 🔌 API 接口文档

### 基础信息
- **Base URL**: `http://localhost:8000`
- **API Version**: v1
- **Content-Type**: `multipart/form-data`

### 端点

#### 1. 健康检查
```http
GET /
```

**响应示例：**
```json
{
  "message": "PDS RMBG API is running"
}
```

#### 2. 背景去除与产品分割
```http
POST /v1/rmbg
```

**请求参数：**
- `image` (file): 图片文件，支持 JPG/PNG 格式

**响应格式：**
```json
{
  "status": 0,
  "msg": "",
  "data": {
    "package_images": [
      {
        "index": 1,
        "mask_b64": "base64编码的mask图片",
        "image_b64": "base64编码的产品图片"
      }
    ]
  }
}
```

**状态码说明：**
- `status: 0` - 成功
- `status: 1` - 失败，错误信息在 `msg` 字段中

**cURL 示例：**
```bash
curl -X POST "http://localhost:8000/v1/rmbg" \
  -H "Content-Type: multipart/form-data" \
  -F "image=@demo.jpg"
```

**Python 示例：**
```python
import requests
import base64
from PIL import Image
import io

# 上传图片
with open("demo.jpg", "rb") as f:
    files = {"image": f}
    response = requests.post("http://localhost:8000/v1/rmbg", files=files)

# 处理响应
result = response.json()
if result["status"] == 0:
    for pkg in result["data"]["package_images"]:
        # 解码图片
        image_data = base64.b64decode(pkg["image_b64"])
        mask_data = base64.b64decode(pkg["mask_b64"])
        
        # 保存图片
        with open(f"product_{pkg['index']}.png", "wb") as f:
            f.write(image_data)
        with open(f"mask_{pkg['index']}.png", "wb") as f:
            f.write(mask_data)
```

**JavaScript 示例：**
```javascript
const formData = new FormData();
formData.append('image', fileInput.files[0]);

fetch('http://localhost:8000/v1/rmbg', {
  method: 'POST',
  body: formData
})
.then(response => response.json())
.then(data => {
  if (data.status === 0) {
    data.data.package_images.forEach(pkg => {
      // 创建图片元素
      const img = document.createElement('img');
      img.src = `data:image/png;base64,${pkg.image_b64}`;
      document.body.appendChild(img);
    });
  }
});
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
