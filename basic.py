import sys
import threading
import time
import numpy as np
from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QCheckBox, QSlider
)
from PyQt6.QtCore import Qt, QTimer
import pyaudio
from pyaec import Aec

# --- Audio Configuration ---
CHUNK = 1024
FORMAT = pyaudio.paInt16  # Format 16-bit
CHANNELS = 1
RATE = 16000  # Sampling = 16kHz, good for voice processing


# --- AEC Application Class ---
class AECLoopbackApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Mô phỏng Khử Tiếng Vọng (AEC)")
        self.setGeometry(100, 100, 500, 300)
        self.p = None
        self.stream = None
        self.is_running = False
        self.aec_enabled = False
        self.mic_volume_factor = 1.0

        self.aec = Aec(1024, 128, RATE, True)

        # Biến đệm loa (Đây sẽ là tín hiệu xa cần thiết cho AEC thực tế)
        # We need a buffer to store the signal sent to the speaker (far-end signal)
        self.far_end_buffer = np.zeros(CHUNK * 2, dtype=np.int16)

        self.init_ui()
        self.init_pyaudio()

    def init_pyaudio(self):
        """Init PyAudio."""
        try:
            self.p = pyaudio.PyAudio()
        except Exception as e:
            self.status_label.setText(f"Lỗi PyAudio: {e}")
            self.start_button.setEnabled(False)

    def init_ui(self):
        """Init user UI interface."""
        main_layout = QVBoxLayout()
        main_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # Tiêu đề và Trạng thái
        title_label = QLabel("Acoustic Echo Cancellation (AEC) Demo")
        title_label.setStyleSheet("font-size: 20px; font-weight: bold; color: #1e40af;")
        main_layout.addWidget(title_label, alignment=Qt.AlignmentFlag.AlignCenter)

        self.status_label = QLabel("Trạng thái: Đã dừng")
        self.status_label.setStyleSheet("font-size: 14px; margin-bottom: 15px;")
        main_layout.addWidget(self.status_label, alignment=Qt.AlignmentFlag.AlignCenter)

        # Thanh trượt Âm lượng Mic (Microphone Volume Slider)
        vol_layout = QHBoxLayout()
        vol_layout.addWidget(QLabel("Tăng Âm Lượng Loa (Gây Echo Mạnh hơn):"))
        self.vol_slider = QSlider(Qt.Orientation.Horizontal)
        self.vol_slider.setMinimum(10)
        self.vol_slider.setMaximum(300)
        self.vol_slider.setValue(100)
        self.vol_slider.setSingleStep(1)
        self.vol_slider.valueChanged.connect(self.update_volume)
        self.vol_display = QLabel("1.0x")
        vol_layout.addWidget(self.vol_slider)
        vol_layout.addWidget(self.vol_display)
        main_layout.addLayout(vol_layout)

        # AEC Toggle và Nút điều khiển
        control_layout = QHBoxLayout()

        self.aec_checkbox = QCheckBox("Bật AEC")
        self.aec_checkbox.setStyleSheet("font-size: 16px;")
        self.aec_checkbox.stateChanged.connect(self.toggle_aec)
        control_layout.addWidget(self.aec_checkbox, alignment=Qt.AlignmentFlag.AlignRight)

        self.start_button = QPushButton("Bắt Đầu Loopback")
        self.start_button.setStyleSheet(
            "padding: 10px 20px; font-size: 16px; background-color: #22c55e; color: white; border-radius: 8px;"
        )
        self.start_button.clicked.connect(self.start_stop_stream)
        control_layout.addWidget(self.start_button, alignment=Qt.AlignmentFlag.AlignLeft)

        main_layout.addLayout(control_layout)

        # Hướng dẫn
        info_label = QLabel(
            "<br><b>Cách Demo:</b><br>"
            "1. Bấm <b>Bắt Đầu Loopback</b>.<br>"
            "2. Nói vào Microphone. Bạn sẽ nghe thấy giọng nói của mình bị trễ/vọng.<br>"
            "3. Tăng thanh trượt âm lượng loa để tiếng vọng mạnh hơn.<br>"
            "4. Bật <b>Bật AEC</b> (nếu có thuật toán thực tế) để thấy sự khác biệt."
        )
        info_label.setStyleSheet("font-size: 12px; color: #6b7280; margin-top: 15px;")
        main_layout.addWidget(info_label)

        self.setLayout(main_layout)

        # Timer để cập nhật UI nếu cần
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.check_status)
        self.timer.start(100)  # Cập nhật mỗi 100ms

    def update_volume(self, value):
        """Cập nhật hệ số âm lượng loa."""
        self.mic_volume_factor = value / 100.0
        self.vol_display.setText(f"{self.mic_volume_factor:.2f}x")

    def toggle_aec(self, state):
        """Bật hoặc tắt tính năng AEC."""
        self.aec_enabled = state == Qt.CheckState.Checked.value
        self.status_label.setText(
            f"Trạng thái: Đang chạy | AEC: {'BẬT' if self.aec_enabled else 'TẮT'}"
        )


    def aec_process(self, near_end_frame: np.ndarray, far_end_frame: np.ndarray) -> np.ndarray:
        """
        *** NƠI TRIỂN KHAI THUẬT TOÁN AEC THỰC TẾ ***

        Hàm này nhận:
        - near_end_frame: Tín hiệu từ Microphone (có chứa tiếng vọng).
        - far_end_frame: Tín hiệu đã phát ra loa (tiếng vọng mong muốn khử).

        Hiện tại, nó chỉ là một bộ lọc cắt cơ bản để mô phỏng "sự khác biệt" khi AEC được BẬT.
        Để có AEC thực tế, bạn cần sử dụng một thuật toán như NLMS (Normalized Least Mean Squares).
        """
        if self.aec_enabled:
            # Ví dụ mô phỏng đơn giản: ngưỡng tín hiệu thấp để giảm tiếng ồn nền (không phải AEC thực)
            # Tín hiệu sẽ được làm sạch một chút.
            threshold = 500  # Ngưỡng (thử nghiệm với giá trị này)
            # processed_frame = np.where(np.abs(near_end_frame) > threshold, near_end_frame, 0)
            processed_frame = self.aec.cancel_echo(near_end_frame, far_end_frame)
            return processed_frame.astype(np.int16)
        else:
            # Nếu AEC TẮT, chỉ cần chuyển tiếp (loopback) tín hiệu gốc.
            return near_end_frame.astype(np.int16)

    def audio_callback(self, in_data, frame_count, time_info, status):
        """
        Hàm Callback PyAudio: Được gọi khi có dữ liệu từ microphone.
        Thực hiện Loopback và xử lý AEC.
        """
        try:
            # 1. Chuyển đổi dữ liệu input (bytes) sang mảng numpy (tín hiệu cận)
            near_end_frame = np.frombuffer(in_data, dtype=np.int16)

            # 2. Lấy tín hiệu xa (far_end_frame) từ buffer (tín hiệu đã phát ra trước đó)
            # Trong một AEC thực tế, bạn cần đồng bộ hóa chính xác tín hiệu loa (far-end)
            # với tín hiệu mic (near-end) theo độ trễ.
            # Ở đây ta lấy một phần của buffer (đơn giản hóa)
            far_end_frame = self.far_end_buffer[:frame_count]

            # 3. Xử lý AEC
            processed_frame = self.aec_process(near_end_frame, far_end_frame)

            # 4. Tăng âm lượng đầu ra để làm tiếng vọng rõ ràng hơn (nếu AEC TẮT)
            output_frame = (processed_frame * self.mic_volume_factor).clip(-32768, 32767).astype(np.int16)

            # 5. Cập nhật buffer tín hiệu xa cho lần gọi tiếp theo
            self.far_end_buffer = np.roll(self.far_end_buffer, -frame_count)
            self.far_end_buffer[-frame_count:] = output_frame

            # 6. Trả lại dữ liệu output (bytes)
            return output_frame.tobytes(), pyaudio.paContinue

        except Exception as e:
            print(f"Lỗi trong Callback: {e}")
            return in_data, pyaudio.paAbort

    def start_stream(self):
        """Bắt đầu luồng âm thanh."""
        if self.is_running:
            return

        try:
            self.stream = self.p.open(
                format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                output=True,
                frames_per_buffer=CHUNK,
                stream_callback=self.audio_callback,
                # Thiết lập độ trễ thấp nhất có thể để echo rõ ràng
                # suggest_input_device_index=...,
                # suggest_output_device_index=...,
            )
            self.stream.start_stream()
            self.is_running = True
            self.start_button.setText("Dừng Loopback")
            self.start_button.setStyleSheet(
                "padding: 10px 20px; font-size: 16px; background-color: #ef4444; color: white; border-radius: 8px;"
            )
            self.status_label.setText(
                f"Trạng thái: Đang chạy | AEC: {'BẬT' if self.aec_enabled else 'TẮT'}"
            )
        except Exception as e:
            self.status_label.setText(f"Lỗi khởi động luồng: {e}")
            self.is_running = False

    def stop_stream(self):
        """Dừng luồng âm thanh."""
        if not self.is_running:
            return

        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
            self.stream = None

        self.is_running = False
        self.start_button.setText("Bắt Đầu Loopback")
        self.start_button.setStyleSheet(
            "padding: 10px 20px; font-size: 16px; background-color: #22c55e; color: white; border-radius: 8px;"
        )
        self.status_label.setText("Trạng thái: Đã dừng")

    def start_stop_stream(self):
        """Xử lý sự kiện bấm nút Start/Stop."""
        if self.is_running:
            self.stop_stream()
        else:
            self.start_stream()

    def check_status(self):
        """Kiểm tra trạng thái luồng định kỳ."""
        if self.is_running and self.stream and not self.stream.is_active():
            # Nếu luồng bị dừng ngoài ý muốn
            print("Luồng âm thanh bị ngắt. Dừng lại.")
            self.stop_stream()

    def closeEvent(self, event):
        """Dọn dẹp tài nguyên khi đóng ứng dụng."""
        self.stop_stream()
        if self.p:
            self.p.terminate()
        event.accept()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = AECLoopbackApp()
    window.show()
    sys.exit(app.exec())