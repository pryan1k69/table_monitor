import cv2
import numpy as np
from typing import Tuple

class Visualizer: 
    def __init__(self):
        self.display_scale = 1.0
        
    def scale_for_display(self, frame: np.ndarray, scale: float) -> np.ndarray:
        """Масштабирует кадр для отображения"""
        if scale == 1.0:
            return frame.copy()
        
        new_width = int(frame.shape[1] * scale)
        new_height = int(frame.shape[0] * scale)
        return cv2.resize(frame, (new_width, new_height))
    
    def scale_coordinates(self, x: int, y: int, w: int, h: int, 
                          scale: float, to_display: bool = True) -> Tuple[int, int, int, int]:
        """Масштабирует координаты между оригинальным и отображаемым размером"""
        if to_display:
            return (int(x * scale), 
                    int(y * scale),
                    int(w * scale), 
                    int(h * scale))
        else:
            return (int(x / scale), 
                    int(y / scale),
                    int(w / scale), 
                    int(h / scale))
    
    def draw_visualization(self, frame: np.ndarray, roi: Tuple[int, int, int, int], 
                          current_state: str, display_scale: float, 
                          total_frames: int, events_count: int, info: dict = None) -> np.ndarray:
        # Масштабируем кадр для отображения
        display_frame = self.scale_for_display(frame, display_scale)
        
        # Масштабируем координаты ROI
        display_x, display_y, display_w, display_h = self.scale_coordinates(
            roi[0], roi[1], roi[2], roi[3], display_scale, to_display=True
        )
        
        # Выбираем цвет
        color = (0, 255, 0) if current_state == "empty" else (0, 0, 255)
        state_text = "FREE" if current_state == "empty" else "OCCUPIED"
        
        # Рисуем прямоугольник
        cv2.rectangle(display_frame, (display_x, display_y), 
                     (display_x + display_w, display_y + display_h), color, 3)
        
        # Текст состояния с фоном
        text_size = cv2.getTextSize(state_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
        cv2.rectangle(display_frame, 
                     (display_x, display_y - text_size[1] - 10),
                     (display_x + text_size[0] + 10, display_y - 5),
                     (0, 0, 0), -1)
        cv2.putText(display_frame, state_text, (display_x + 5, display_y - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
        # Информационная панель с временем
        if info:
            current_time = info.get('current_time', 0)
            duration = info.get('duration', 0)
            progress = info.get('progress', 0)
            
            info_text = [
                f"Table: {state_text}",
                f"Time: {current_time:.1f}/{duration:.1f}s ({progress:.0f}%)",
                f"Frame: {total_frames}",
                f"Events: {events_count}",
                f"Scale: {display_scale:.0%}"
            ]
        else:
            info_text = [
                f"Table: {state_text}",
                f"Frame: {total_frames}",
                f"Events: {events_count}",
                f"Scale: {display_scale:.0%}"
            ]
        
        for i, text in enumerate(info_text):
            y_pos = 30 + i * 30
            cv2.putText(display_frame, text, (10, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Инструкции с перемоткой
        instructions = "Q:Quit | P:Play/Pause | +/-:Zoom | R:Reset | S:Screenshot | [/]:+/-30s | Home/End"
        text_size = cv2.getTextSize(instructions, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        
        if text_size[0] < display_frame.shape[1]:
            cv2.putText(display_frame, instructions, 
                       (display_frame.shape[1] - text_size[0] - 10, 
                        display_frame.shape[0] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        return display_frame
    
    def draw_paused(self, frame: np.ndarray) -> np.ndarray:
        """Добавляет надпись PAUSED на кадр"""
        cv2.putText(frame, "PAUSED", (frame.shape[1] - 100, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        return frame