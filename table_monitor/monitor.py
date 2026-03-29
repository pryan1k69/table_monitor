import cv2
import pandas as pd
from datetime import datetime
import time
import numpy as np
from typing import Tuple, List, Dict, Optional
from pathlib import Path

from .config import MonitorConfig
from .logger import setup_logging
from .detector import PersonDetector
from .visualizer import Visualizer


class TableMonitor:
    def __init__(self, config: MonitorConfig):
        self.config = config
        
        # Настройка логирования
        self.logger, self.events_logger = setup_logging(config.log_dir, config.log_level)
        
        self.logger.info("="*60)
        self.logger.info("ЗАПУСК МОНИТОРИНГА СТОЛИКА")
        self.logger.info("="*60)
        self.logger.info(f"Конфигурация: {config.__dict__}")
        
        # Инициализация видео
        self.cap = cv2.VideoCapture(config.video_path)
        if not self.cap.isOpened():
            error_msg = f"Не удалось открыть видео: {config.video_path}"
            self.logger.error(error_msg)
            raise ValueError(error_msg)
        
        # Получаем оригинальные размеры видео
        self.original_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.original_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames_in_video = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.video_duration = self.total_frames_in_video / self.fps if self.fps > 0 else 0
        
        self.logger.info(f"   Информация о видео:")
        self.logger.info(f"   - Размер: {self.original_width}x{self.original_height}")
        self.logger.info(f"   - FPS: {self.fps:.2f}")
        self.logger.info(f"   - Всего кадров: {self.total_frames_in_video}")
        self.logger.info(f"   - Длительность: {self.video_duration:.1f} секунд")
        
        print(f"Оригинальный размер видео: {self.original_width}x{self.original_height}")
        
        # Параметры отображения
        self.window_scale = config.window_scale
        self.lock_scale = config.lock_scale
        self.display_scale = config.window_scale
        self.current_window_width = int(self.original_width * config.window_scale)
        self.current_window_height = int(self.original_height * config.window_scale)
        
        # Параметры перемотки
        self.seek_step = 5.0  # Шаг перемотки в секундах
        self.current_time = 0.0
        self.seeking = False
        self.manual_seek = False
        self.last_trackbar_pos = -1
        self.auto_update_trackbar = True
        
        # Инициализация визуализатора
        self.visualizer = Visualizer()
        
        # Настройка окна
        self._setup_window()
        
        # Загрузка модели
        self.logger.info(f"Загрузка модели YOLO: {config.model_path}")
        self.detector = PersonDetector(config.model_path, config.confidence_threshold)
        
        # Выбор области столика
        if config.roi is None:
            self.roi = self._select_roi_interactive()
        else:
            self.roi = config.roi
            self.logger.info(f"Используется заданная область столика: {self.roi}")
        
        # Состояния системы
        self.current_state = "empty"
        self.events: List[Dict] = []
        self.last_empty_time: Optional[float] = None
        
        # Параметры детекции
        self.confidence_threshold = config.confidence_threshold
        self.occupied_frames_threshold = config.occupied_frames_threshold
        self.frames_to_skip = config.frames_to_skip
        
        # Счетчики стабильности
        self.empty_counter = 0
        self.occupied_counter = 0
        
        # Статистика
        self.total_frames = 0
        self.processed_frames = 0
        self.processing_start_time = None
        self.detection_times = []
        
        # Создаем директории
        if config.save_statistics:
            Path(config.statistics_dir).mkdir(exist_ok=True)
        Path(config.screenshots_dir).mkdir(exist_ok=True)
        
        self.logger.info("Мониторинг инициализирован успешно")
    
    def _setup_window(self):
        """Настройка окна отображения"""
        window_name = "Table Monitor"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        
        # Устанавливаем размер окна
        cv2.resizeWindow(window_name, self.current_window_width, self.current_window_height)
        
        # Позволяем пользователю изменять размер окна мышкой
        cv2.setWindowProperty(window_name, cv2.WND_PROP_AUTOSIZE, cv2.WINDOW_NORMAL)
        
        # Создаем трекбар для перемотки
        cv2.createTrackbar('Position', window_name, 0, 100, self._on_trackbar_change)
    
    def _on_trackbar_change(self, pos):
        """Обработчик изменения положения трекбара"""
        # Игнорируем, если это автоматическое обновление
        if not self.auto_update_trackbar:
            return
            
        # Игнорируем, если позиция не изменилась
        if pos == self.last_trackbar_pos:
            return
            
        self.last_trackbar_pos = pos
        self.manual_seek = True
        
        # Переводим проценты в секунды
        target_time = (pos / 100.0) * self.video_duration
        self._seek_to_time(target_time)
        
        # Небольшая задержка для завершения перемотки
        cv2.waitKey(10)
    
    def _seek_to_time(self, target_time: float):
        # Ограничиваем время
        target_time = max(0.0, min(target_time, self.video_duration))
        
        # Устанавливаем позицию в кадрах
        target_frame = int(target_time * self.fps)
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
        
        # Обновляем текущее время
        self.current_time = target_time
        
        # Сбрасываем счетчики стабильности при перемотке
        self.empty_counter = 0
        self.occupied_counter = 0
        
        self.logger.debug(f"Перемотка на {target_time:.1f} сек (кадр {target_frame})")
        
        # Обновляем позицию трекбара без вызова обработчика
        self.auto_update_trackbar = False
        new_pos = int((target_time / self.video_duration) * 100) if self.video_duration > 0 else 0
        cv2.setTrackbarPos('Position', 'Table Monitor', new_pos)
        self.last_trackbar_pos = new_pos
        self.auto_update_trackbar = True
    
    def _seek_relative(self, delta_seconds: float):
        new_time = self.current_time + delta_seconds
        self._seek_to_time(new_time)
        
        self.logger.info(f"Перемотка: {delta_seconds:+.1f} сек -> {self.current_time:.1f} сек")
    
    def _select_roi_interactive(self) -> Tuple[int, int, int, int]:
        # Читаем первый кадр в оригинальном размере
        ret, frame = self.cap.read()
        if not ret:
            error_msg = "Не удалось прочитать видео для выбора области"
            self.logger.error(error_msg)
            raise ValueError(error_msg)
        
        # Создаем копию для отображения с текущим масштабом
        display_frame = self.visualizer.scale_for_display(frame, self.display_scale)
        
        print("\n=== ВЫБОР ОБЛАСТИ СТОЛИКА ===")
        print("1. Выделите область стола с помощью мыши")
        print("2. Нажмите Enter/Пробел для подтверждения")
        print("3. Нажмите 'c' для отмены и повторного выбора")
        print("4. Используйте колесико мыши для масштабирования окна (если нужно)")
        print("================================\n")
        
        # Временное окно для выбора ROI
        temp_window = "Выберите область столика"
        cv2.namedWindow(temp_window, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(temp_window, self.current_window_width, self.current_window_height)
        
        # Выбор ROI на масштабированном изображении
        scaled_roi = cv2.selectROI(temp_window, display_frame, False)
        cv2.destroyWindow(temp_window)
        
        # Проверка, что область выбрана
        if scaled_roi == (0, 0, 0, 0):
            self.logger.warning("ROI не выбран, используется весь кадр")
            return (0, 0, self.original_width, self.original_height)
        
        # Конвертируем координаты обратно в оригинальный масштаб
        x, y, w, h = scaled_roi
        original_x = int(x / self.display_scale)
        original_y = int(y / self.display_scale)
        original_w = int(w / self.display_scale)
        original_h = int(h / self.display_scale)
        
        # Проверяем границы
        original_x = max(0, min(original_x, self.original_width - 1))
        original_y = max(0, min(original_y, self.original_height - 1))
        original_w = min(original_w, self.original_width - original_x)
        original_h = min(original_h, self.original_height - original_y)
        
        # Сбрасываем видео на начало
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        return (original_x, original_y, original_w, original_h)
    
    def detect_people_in_roi(self, frame: np.ndarray) -> bool:
        """Детекция людей в области столика (в оригинальных координатах)"""
        return self.detector.detect_people_in_roi(frame, self.roi)
    
    def update_state(self, has_people: bool):
        # Обновляем состояния
        if has_people:
            self.occupied_counter += 1
            self.empty_counter = 0
            if self.occupied_counter >= self.occupied_frames_threshold:
                if self.current_state != "occupied":
                    self.current_state = "occupied"
                    self._log_event("occupied", "Стол занят")
                    if self.last_empty_time is not None:
                        wait_time = self.current_time - self.last_empty_time
                        self.logger.info(f"Время ожидания клиента: {wait_time:.1f} сек")
        else:
            self.empty_counter += 1
            self.occupied_counter = 0
            if self.empty_counter >= self.occupied_frames_threshold:
                if self.current_state != "empty":
                    self.current_state = "empty"
                    self._log_event("empty", "Стол пуст")
                    self.last_empty_time = self.current_time
    
    def _log_event(self, event_type: str, description: str):
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        video_time = self.current_time
        
        event = {
            'timestamp': current_time,
            'event_type': event_type,
            'description': description,
            'video_time': video_time,
            'frame_number': self.total_frames
        }
        self.events.append(event)
        
        # Логируем в специальный файл событий
        self.events_logger.info(f"{event_type.upper()} | {description} | "
                               f"Время видео: {video_time:.1f}с | Кадр: {self.total_frames}")
    
    def draw_visualization(self, frame: np.ndarray) -> np.ndarray:
        # Добавляем текущее время в информацию
        progress_percent = (self.current_time / self.video_duration * 100) if self.video_duration > 0 else 0
        
        info = {
            'state': self.current_state,
            'frame': self.total_frames,
            'events': len(self.events),
            'scale': self.display_scale,
            'current_time': self.current_time,
            'duration': self.video_duration,
            'progress': progress_percent
        }
        
        return self.visualizer.draw_visualization(
            frame, self.roi, self.current_state, self.display_scale,
            self.total_frames, len(self.events), info
        )
    
    def change_scale(self, delta: float):
        if self.lock_scale:
            self.logger.warning("Попытка изменить масштаб, но масштаб заблокирован")
            return
        
        # Изменяем масштаб (ограничиваем от 0.3 до 2.0)
        new_scale = self.display_scale + delta
        self.display_scale = max(0.3, min(2.0, new_scale))
        
        # Обновляем размер окна
        self.current_window_width = int(self.original_width * self.display_scale)
        self.current_window_height = int(self.original_height * self.display_scale)
        
        cv2.resizeWindow("Table Monitor", self.current_window_width, self.current_window_height)
        
        self.logger.info(f"Масштаб изменен: {self.display_scale:.2f} "
                        f"({self.current_window_width}x{self.current_window_height})")
    
    def reset_scale(self):
        if self.lock_scale:
            self.logger.warning("Попытка сбросить масштаб, но масштаб заблокирован")
            return
        
        self.display_scale = self.window_scale
        self.current_window_width = int(self.original_width * self.display_scale)
        self.current_window_height = int(self.original_height * self.display_scale)
        
        cv2.resizeWindow("Table Monitor", self.current_window_width, self.current_window_height)
        
        self.logger.info(f"Масштаб сброшен: {self.display_scale:.2f}")
    
    def save_statistics(self):
        if not self.config.save_statistics:
            return
        
        self.logger.info("Сохранение статистики...")
        
        # Сохраняем события в CSV
        if self.events:
            df_events = pd.DataFrame(self.events)
            events_csv = Path(self.config.statistics_dir) / f"events_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            df_events.to_csv(events_csv, index=False, encoding='utf-8')
            self.logger.info(f"События сохранены в: {events_csv}")
        
        self.logger.info("Статистика сохранена")
    
    def calculate_statistics(self) -> pd.DataFrame:
        if not self.events:
            self.logger.warning("Нет зарегистрированных событий")
            return pd.DataFrame()
        
        df = pd.DataFrame(self.events)
       
        self.logger.info("Расчет статистики")
        
        # Анализ времени ожидания
        empty_times = df[df['event_type'] == 'empty']['video_time'].tolist()
        occupied_times = df[df['event_type'] == 'occupied']['video_time'].tolist()
        
        wait_times = []
        for empty_time in empty_times:
            next_occupied = [t for t in occupied_times if t > empty_time]
            if next_occupied:
                wait_times.append(next_occupied[0] - empty_time)
        
        if wait_times:
            print(f"\nАНАЛИТИКА ВРЕМЕНИ ОЖИДАНИЯ:")
            print(f"  - Проанализировано циклов: {len(wait_times)}")
            print(f"  - Среднее время: {np.mean(wait_times):.1f} секунд")
            print(f"  - Медиана: {np.median(wait_times):.1f} секунд")
            print(f"  - Минимум: {min(wait_times):.1f} секунд")
            print(f"  - Максимум: {max(wait_times):.1f} секунд")
            
            self.logger.info(f"Статистика ожидания: среднее={np.mean(wait_times):.1f}с, "
                           f"медиана={np.median(wait_times):.1f}с")
        
        # Метрики производительности
        if self.detector.detection_times:
            avg_time = np.mean(self.detector.detection_times) * 1000
            print(f"\nПРОИЗВОДИТЕЛЬНОСТЬ:")
            print(f"  - Среднее время детекции: {avg_time:.1f} мс")
            print(f"  - Обработано кадров: {self.processed_frames}/{self.total_frames}")
        
        return df
    
    def run(self) -> pd.DataFrame:
        print("\n" + "="*60)
        print("ЗАПУСК АНАЛИЗА ВИДЕО")
        print("="*60)
        print("Управление:")
        print("  - 'q' - выход")
        print("  - 'p' - пауза")
        print("  - '+' - увеличить масштаб")
        print("  - '-' - уменьшить масштаб")
        print("  - 'r' - сбросить масштаб")
        print("  - 's' - скриншот")
        print("  - '[' - перемотка назад на 30 секунд")
        print("  - ']' - перемотка вперед на 30 секунд")
        print("  - 'Home' - в начало видео")
        print("  - 'End' - в конец видео")
        print("  - Трекбар - перемотка мышкой")
        print("="*60 + "\n")
        
        self.logger.info("Запуск основного цикла обработки")
        self.processing_start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        paused = False
        frame_count = 0
        start_time = time.time()
        
        # Получаем начальную позицию
        self.current_time = 0.0
        
        try:
            while True:
                if not paused:
                    ret, frame = self.cap.read()
                    if not ret:
                        # Если достигли конца, завершаем
                        if not self.manual_seek:
                            self.logger.info("Достигнут конец видео")
                            break
                        else:
                            self.manual_seek = False
                            continue
                    
                    frame_count += 1
                    self.total_frames = frame_count
                    
                    # Обновляем текущее время
                    self.current_time = self.cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                    
                    # Обновляем позицию трекбара только при нормальном воспроизведении
                    if not self.manual_seek and not self.seeking:
                        progress = int((self.current_time / self.video_duration) * 100) if self.video_duration > 0 else 0
                        
                        # Обновляем только если позиция изменилась
                        if progress != self.last_trackbar_pos:
                            self.auto_update_trackbar = False
                            cv2.setTrackbarPos('Position', 'Table Monitor', progress)
                            self.last_trackbar_pos = progress
                            self.auto_update_trackbar = True
                    
                    # Сбрасываем флаг ручной перемотки после обновления кадра
                    if self.manual_seek:
                        self.manual_seek = False
                    
                    if frame_count % (self.frames_to_skip + 1) == 0:
                        self.processed_frames += 1
                        has_people = self.detect_people_in_roi(frame)
                        self.update_state(has_people)
                    
                    # Визуализация
                    display_frame = self.draw_visualization(frame)
                    
                    if paused:
                        display_frame = self.visualizer.draw_paused(display_frame)
                    
                    cv2.imshow('Table Monitor', display_frame)
                
                # Обработка клавиш
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    self.logger.info("Анализ остановлен")
                    break
                elif key == ord('p') or key == 32:  # 32 - пробел
                    paused = not paused
                    self.logger.info(f"Пауза: {'включена' if paused else 'выключена'}")
                elif key == ord('+') or key == ord('='):
                    self.change_scale(0.1)
                elif key == ord('-'):
                    self.change_scale(-0.1)
                elif key == ord('r'):
                    self.reset_scale()
                elif key == ord('s') and not paused:
                    screenshot_file = Path(self.config.screenshots_dir) / f"screenshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                    cv2.imwrite(str(screenshot_file), display_frame)
                    self.logger.info(f"Скриншот сохранен: {screenshot_file}")
                
                # Перемотка на 30 секунд (используем [ и ])
                elif key == 91:  # '['
                    self.manual_seek = True
                    self._seek_relative(-30.0)
                    paused = False
                    continue
                
                elif key == 93:  # ']'
                    self.manual_seek = True
                    self._seek_relative(30.0)
                    paused = False
                    continue
                
                # Home - в начало
                elif key == 80 or key == 2424832:  # Home
                    self.manual_seek = True
                    self._seek_to_time(0)
                    paused = False
                    continue
                
                # End - в конец
                elif key == 82 or key == 2424833:  # End
                    self.manual_seek = True
                    self._seek_to_time(self.video_duration - 0.1)
                    paused = False
                    continue
        
        except Exception as e:
            self.logger.error(f"Критическая ошибка в основном цикле: {e}", exc_info=True)
            raise
        
        finally:
            elapsed_time = time.time() - start_time
            self.logger.info(f"Обработка завершена. Время работы: {elapsed_time:.1f} секунд")
            self.logger.info(f"Итоги: кадров={self.total_frames}, событий={len(self.events)}")
            
            self.cap.release()
            cv2.destroyAllWindows()
            
            # Сохраняем статистику
            self.save_statistics()
        
        return self.calculate_statistics()