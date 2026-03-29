import logging
from pathlib import Path
from datetime import datetime
from logging.handlers import RotatingFileHandler

def setup_logging(log_dir: str = "logs", log_level: str = "INFO"):
    """Настройка системы логирования"""
    # Создаем папку для логов
    log_path = Path(log_dir)
    log_path.mkdir(exist_ok=True)
    
    # Имя файла лога с датой
    log_file = log_path / f"table_monitor_{datetime.now().strftime('%Y%m%d')}.log"
    
    # Создаем логгер
    logger = logging.getLogger('TableMonitor')
    logger.setLevel(getattr(logging, log_level))
    
    # Очищаем существующие обработчики
    if logger.handlers:
        logger.handlers.clear()
    
    # Форматирование
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    console_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    
    # Файловый обработчик с ротацией
    file_handler = RotatingFileHandler(
        log_file, 
        maxBytes=10_000_000,
        backupCount=5,
        encoding='utf-8'
    )
    file_handler.setLevel(getattr(logging, log_level))
    file_handler.setFormatter(file_formatter)
    
    # Консольный обработчик
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, log_level))
    console_handler.setFormatter(console_formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    # Создаем отдельный логгер для событий
    events_logger = logging.getLogger('TableMonitor.Events')
    events_logger.setLevel(logging.INFO)
    events_logger.handlers.clear()
    
    events_file = log_path / f"events_{datetime.now().strftime('%Y%m%d')}.log"
    events_handler = RotatingFileHandler(
        events_file,
        maxBytes=10_000_000,
        backupCount=5,
        encoding='utf-8'
    )
    events_handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    ))
    events_logger.addHandler(events_handler)
    
    return logger, events_logger