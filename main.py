import sys
import logging
from pathlib import Path
from datetime import datetime

# Добавляем путь к проекту
sys.path.insert(0, str(Path(__file__).parent))

from table_monitor import TableMonitor, MonitorConfig


def parse_arguments():
    """Парсинг аргументов командной строки"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Table Monitor - система мониторинга занятости столов"
    )
    
    parser.add_argument(
        '--video', '-v',
        type=str,
        required=True,
        help='Путь к видеофайлу'
    )
    
    return parser.parse_args()


def create_default_config(video_path: str) -> MonitorConfig:
    """Создание конфигурации по умолчанию"""
    return MonitorConfig(
        video_path=video_path,
        window_scale=0.7,
        lock_scale=False,
        confidence_threshold=0.25,
        occupied_frames_threshold=3,
        frames_to_skip=2,
        log_dir="logs",
        log_level="INFO",
        save_statistics=True,
        statistics_dir="statistics",
        screenshots_dir="screenshots"
    )


def main():
    """Главная функция"""
    # Парсим аргументы
    args = parse_arguments()
    
    try:
        # Создаем конфигурацию
        config = create_default_config(args.video)
        
        # Создаем и запускаем мониторинг
        monitor = TableMonitor(config)
        results = monitor.run()
        
        print("\nАнализ завершен!")
        
        # Сохраняем финальную статистику
        if not results.empty and config.save_statistics:
            stats_file = f"final_statistics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            results.to_csv(stats_file, index=False)
            print(f"Финальная статистика сохранена в: {stats_file}")
        
    except FileNotFoundError:
        logging.error(f"Файл не найден: {args.video}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n\nПрограмма остановлена пользователем (Ctrl+C)")
        sys.exit(0)
    except Exception as e:
        logging.error(f"Критическая ошибка: {e}", exc_info=True)
        print("\nУстановите зависимости: см. README")
        sys.exit(1)


if __name__ == "__main__":
    main()