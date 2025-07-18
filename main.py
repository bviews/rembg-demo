import base64
import io
import logging
import os
import tempfile
from contextlib import asynccontextmanager
from typing import List

from fastapi import FastAPI, File, UploadFile
from PIL import Image
from pydantic import BaseModel

from segment import initialize_model, remove_background, split_foregrounds
from settings import settings

# 配置日志
logging.basicConfig(
    level=getattr(logging, settings.log_level.upper()),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_: FastAPI):
    """应用生命周期管理"""
    # 启动时初始化模型
    logger.info("正在启动应用...")
    try:
        initialize_model()
        logger.info("应用启动完成，模型已加载")
    except Exception as e:
        logger.error(f"应用启动失败: {str(e)}")
        raise
    
    yield
    
    # 关闭时清理资源
    logger.info("正在关闭应用...")


app = FastAPI(title=settings.api_title, version=settings.api_version, lifespan=lifespan)


class PackageImage(BaseModel):
    index: int
    mask_b64: str
    image_b64: str


class BaseApiOut(BaseModel):
    status: int
    msg: str
    data: dict


class RMBGData(BaseModel):
    package_images: List[PackageImage]


def image_to_base64(image: Image.Image) -> str:
    """Convert PIL Image to base64 string"""
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    img_str = base64.b64encode(buffer.getvalue()).decode()
    return img_str


def process_uploaded_image(image_file: UploadFile) -> BaseApiOut:
    """Process uploaded image and return segmentation results"""
    try:
        # Validate file type - only support JPG/PNG
        allowed_types = ["image/jpeg", "image/jpg", "image/png"]
        if not image_file.content_type or image_file.content_type.lower() not in allowed_types:
            return BaseApiOut(
                status=1,
                msg="不支持的图片格式，仅支持JPG/PNG格式",
                data={"package_images": []}
            )
        
        # Reset file pointer to beginning
        image_file.file.seek(0)
        
        # Step 1: Remove background - directly use file object
        no_bg_image, _ = remove_background(image_file.file)
        
        # Step 2: Split foregrounds using temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            product_count = split_foregrounds(no_bg_image, temp_dir, min_area=settings.min_area_threshold)
            
            package_images = []
            
            # Process each detected product
            for i in range(product_count):
                product_path = os.path.join(temp_dir, f"product_{i}.png")
                if os.path.exists(product_path):
                    # Load product image
                    product_image = Image.open(product_path)
                    
                    # Create mask for this specific product
                    # Extract alpha channel as mask
                    if product_image.mode == "RGBA":
                        mask_for_product = product_image.split()[-1]  # Get alpha channel
                    else:
                        # Create a simple mask if no alpha channel
                        mask_for_product = Image.new("L", product_image.size, 255)
                    
                    # Convert to base64
                    product_b64 = image_to_base64(product_image)
                    mask_b64 = image_to_base64(mask_for_product)
                    
                    package_images.append(PackageImage(
                        index=i + 1,
                        mask_b64=mask_b64,
                        image_b64=product_b64
                    ))
            
            return BaseApiOut(
                status=0,
                msg="",
                data={"package_images": [img.model_dump() for img in package_images]}
            )
                
    except Exception as e:
        return BaseApiOut(
            status=1,
            msg=str(e),
            data={"package_images": []}
        )


@app.post("/v1/rmbg", response_model=BaseApiOut)
async def rmbg_endpoint(image: UploadFile = File(...)):
    """
    Remove background and segment products from uploaded image
    
    Args:
        image: Uploaded image file (JPG/PNG)
        
    Returns:
        BaseApiOut with segmented products and their masks
    """
    return process_uploaded_image(image)


@app.get("/")
async def root():
    """Health check endpoint"""
    return {"message": "PDS RMBG API is running"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)