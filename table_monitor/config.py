from dataclasses import dataclass
from typing import Optional, Tuple

@dataclass
class MonitorConfig:
    """Конфигурация мониторинга"""
    video_path: str
    roi: Optional[Tuple[int, int, int, int]] = None
    window_scale: float = 0.7
    lock_scale: bool = False
    confidence_threshold: float = 0.25 # Порог уверенности
    occupied_frames_threshold: int = 3 # Стабильность
    frames_to_skip: int = 2 # Пропуск кадров
    model_path: str = 'yolov8n.pt'
    log_dir: str = "logs"
    log_level: str = "INFO"
    save_statistics: bool = True
    statistics_dir: str = "statistics"
    screenshots_dir: str = "screenshots"