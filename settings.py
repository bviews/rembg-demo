from typing import ClassVar
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """应用配置设置"""
    
    # Hugging Face Hub Token
    huggingface_hub_token: str = Field(
        default="",
        description="Hugging Face Hub access token for RMBG-2.0 model"
    )
    
    # 模型配置 - 固定使用 RMBG-2.0
    MODEL_NAME: ClassVar[str] = "briaai/RMBG-2.0"
    
    # 图像处理配置
    image_size: int = Field(
        default=1024,
        description="Model input image size"
    )
    
    min_area_threshold: int = Field(
        default=500,
        description="Minimum area threshold for product segmentation"
    )
    
    # 日志配置
    log_level: str = Field(
        default="INFO",
        description="Logging level"
    )
    
    # API配置
    api_title: str = Field(
        default="PDS RMBG API",
        description="API title"
    )
    
    api_version: str = Field(
        default="1.0.0",
        description="API version"
    )
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )


# 创建全局配置实例
settings = Settings()