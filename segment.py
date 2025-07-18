import logging
import os

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from transformers import AutoModelForImageSegmentation

from settings import settings

logger = logging.getLogger(__name__)

# 全局模型变量
_model = None
_device = None
_transform = None


def initialize_model():
    """
    初始化RMBG-2.0模型，在程序启动时调用
    """
    global _model, _device, _transform
    
    if _model is not None:
        logger.info("模型已经初始化，跳过重复加载")
        return
    
    try:
        logger.info("正在初始化 RMBG-2.0 模型...")
        
        # 从设置中读取 Hugging Face token
        access_token = settings.huggingface_hub_token
        
        # 加载模型
        if access_token:
            logger.info("已检测到 Hugging Face token，正在进行身份验证...")
            _model = AutoModelForImageSegmentation.from_pretrained(
                settings.MODEL_NAME, trust_remote_code=True, token=access_token
            )
        else:
            logger.info("未检测到 HUGGINGFACE_HUB_TOKEN，尝试无认证访问...")
            _model = AutoModelForImageSegmentation.from_pretrained(
                settings.MODEL_NAME, trust_remote_code=True
            )
        
        # 设置计算精度和设备
        _device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"使用设备: {_device}")
        
        if _device == "cuda":
            torch.set_float32_matmul_precision("high")
        
        _model.to(_device)
        _model.eval()
        
        # 预定义图像预处理变换
        image_size = (settings.image_size, settings.image_size)
        _transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        
        logger.info("RMBG-2.0 模型初始化完成！")
        
    except Exception as e:
        logger.error(f"模型初始化失败: {str(e)}")
        raise


def remove_background(image_path):
    """
    使用 RMBG-2.0 算法去除图片背景

    Args:
        image_path (str): 输入图片路径

    Returns:
        tuple: (去除背景后的图片, mask图片)
    """
    global _model, _device, _transform
    
    # 确保模型已初始化
    if _model is None:
        raise RuntimeError("模型未初始化，请先调用 initialize_model()")
    
    try:
        logger.info("正在使用 RMBG-2.0 模型去除背景...")

        # 读取原始图片
        image = Image.open(image_path).convert("RGB")
        original_size = image.size
        logger.info(f"原图尺寸: {original_size}")

        # 图像预处理
        input_images = _transform(image).unsqueeze(0).to(_device)

        logger.info("正在进行背景去除推理...")
        # 执行推理
        with torch.no_grad():
            preds = _model(input_images)[-1].sigmoid().cpu()

        # 处理预测结果
        pred = preds[0].squeeze()
        pred_pil = transforms.ToPILImage()(pred)
        mask = pred_pil.resize(original_size)

        # 创建去背景图片
        no_bg_image = image.copy()
        no_bg_image.putalpha(mask)

        logger.info(f"背景去除完成！图片尺寸: {no_bg_image.size}")
        logger.info("Mask图片生成完成！")

        return no_bg_image, mask

    except Exception as e:
        logger.error(f"背景去除失败: {str(e)}")
        raise


def split_foregrounds(no_bg_image, output_dir, min_area=500):
    """
    从去除背景的图片中分割出多个产品包装
    
    Args:
        no_bg_image (PIL.Image): 已去除背景的 RGBA 图片
        output_dir (str): 输出目录
        min_area (int): 最小面积阈值，用于过滤小噪点
    
    Returns:
        int: 分割出的产品数量
    """
    logger.info("开始分割产品包装...")

    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 转换为 numpy 数组
    rgba = np.array(no_bg_image)

    # 提取 alpha 通道作为 mask
    alpha = rgba[:, :, 3]

    # 转为二值图
    _, binary = cv2.threshold(alpha, 0, 255, cv2.THRESH_BINARY)

    # 形态学操作，去除噪点并连接破碎的区域
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    # 查找轮廓
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    logger.info(f"检测到 {len(contours)} 个轮廓")

    count = 0
    saved_objects = []

    for i, cnt in enumerate(contours):
        # 计算轮廓面积和边界框
        area = cv2.contourArea(cnt)
        x, y, w, h = cv2.boundingRect(cnt)

        # 过滤太小的区域
        if area < min_area:
            logger.debug(f"跳过轮廓 {i}: 面积太小 ({area} < {min_area})")
            continue

        # 创建更精确的掩码
        mask = np.zeros(alpha.shape, dtype=np.uint8)
        cv2.fillPoly(mask, [cnt], 255)

        # 在边界框基础上稍微扩展，确保完整包含对象
        padding = 10
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(rgba.shape[1], x + w + padding)
        y2 = min(rgba.shape[0], y + h + padding)

        # 裁剪 RGBA 图像和掩码
        crop_rgba = rgba[y1:y2, x1:x2]
        crop_mask = mask[y1:y2, x1:x2]

        # 应用掩码，只保留轮廓内的像素
        crop_rgba[:, :, 3] = np.minimum(crop_rgba[:, :, 3], crop_mask)

        # 转换为 PIL 图像并保存
        crop_img = Image.fromarray(crop_rgba)
        filename = f"product_{count}.png"
        filepath = os.path.join(output_dir, filename)
        crop_img.save(filepath)

        saved_objects.append({
            'filename': filename,
            'area': area,
            'bbox': (x, y, w, h),
            'size': crop_img.size
        })

        logger.info(f"保存产品 {count}: {filename}, 面积: {area:.0f}, 尺寸: {crop_img.size}")
        count += 1

    logger.info(f"分割完成！共保存 {count} 个产品包装到目录: {output_dir}")

    # 打印详细统计信息
    if saved_objects:
        logger.info("产品统计信息:")
        for i, obj in enumerate(saved_objects):
            logger.info(f"  产品 {i}: {obj['filename']} - 面积: {obj['area']:.0f}, 尺寸: {obj['size']}")

    return count


def process_image_with_rmbg(image_path, output_dir, min_area=500):
    """
    完整的图像处理流程：RMBG-2.0 背景去除 + 产品分割

    Args:
        image_path (str): 输入图片路径
        output_dir (str): 输出目录
        min_area (int): 最小面积阈值

    Returns:
        tuple: (去背景图片保存路径, mask图片保存路径, 分割产品数量)
    """
    print(f"开始处理图片: {image_path}")
    print(f"使用模型: RMBG-2.0")
    print(f"输出目录: {output_dir}")
    print("=" * 50)

    try:
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)

        # 步骤1: 去除背景
        no_bg_image, mask_image = remove_background(image_path)

        # 保存去背景的完整图片
        no_bg_path = os.path.join(output_dir, "no_background.png")
        no_bg_image.save(no_bg_path)
        print(f"去背景图片已保存: {no_bg_path}")

        # 保存mask图片
        mask_path = os.path.join(output_dir, "mask.png")
        mask_image.save(mask_path)
        print(f"Mask图片已保存: {mask_path}")

        # 步骤2: 分割产品包装
        product_count = split_foregrounds(no_bg_image, output_dir, min_area)

        print("=" * 50)
        print("处理完成！")
        print(f"去背景图片: {no_bg_path}")
        print(f"Mask图片: {mask_path}")
        print(f"分割产品数量: {product_count}")

        return no_bg_path, mask_path, product_count

    except Exception as e:
        print(f"处理失败: {str(e)}")
        raise


if __name__ == "__main__":
    # 配置参数
    input_image = "demo.jpg"
    output_directory = "output_crops"

    # 检查环境变量中的 Hugging Face token
    token = os.getenv("HUGGINGFACE_HUB_TOKEN")
    if not token:
        print("⚠️  注意: RMBG-2.0 模型需要 Hugging Face access token")
        print("请设置环境变量: export HUGGINGFACE_HUB_TOKEN='hf_xxxxxxxx'")
        print("继续尝试无认证访问...")
    else:
        print(f"✅ 已检测到 Hugging Face token (前缀: {token[:8]}...)")

    # 最小面积阈值（像素），用于过滤小噪点
    min_area_threshold = 500

    print("🚀 产品包装分割工具 - RMBG-2.0")
    print("使用最新的 AI 背景去除算法 + 智能产品分割")
    print("=" * 60)

    # 检查输入文件是否存在
    if not os.path.exists(input_image):
        print(f"❌ 错误: 输入文件不存在: {input_image}")
        print("请确保 demo.jpg 文件在当前目录中")
        exit(1)

    try:
        # 执行处理
        result_path, mask_path, product_count = process_image_with_rmbg(
            input_image, output_directory, min_area_threshold
        )

        print(f"\n✅ 处理成功完成!")
        print(f"📁 输出目录: {output_directory}")
        print(f"🖼️  去背景图片: {result_path}")
        print(f"🎭 Mask图片: {mask_path}")
        print(f"📦 分割产品数量: {product_count}")

        if product_count > 0:
            print(f"\n💡 提示: 产品文件保存为 product_0.png, product_1.png, ...")
        else:
            print(f"\n⚠️  未检测到有效产品，您可以尝试:")
            print(f"   - 降低最小面积阈值 (当前: {min_area_threshold})")
            print(f"   - 检查输入图片是否包含清晰的产品")
            print(f"   - 尝试不同的背景去除模型")

    except Exception as e:
        print(f"\n❌ 处理失败: {str(e)}")
        print("\n🔧 可能的解决方案:")
        print("   1. 检查网络连接（首次运行需要下载模型）")
        print("   2. 确保有足够的磁盘空间")
        print("   3. 检查输入图片格式是否支持")
        print("   4. 尝试重新安装依赖: pip install -r requirements.txt")
        if any(
            error_code in str(e)
            for error_code in ["401", "403", "Unauthorized", "gated repo"]
        ):
            print("   5. 🔑 权限错误: 请设置正确的 Hugging Face token:")
            print("      export HUGGINGFACE_HUB_TOKEN='hf_xxxxxxxx'")
            print("      访问 https://huggingface.co/briaai/RMBG-2.0 申请模型权限")
