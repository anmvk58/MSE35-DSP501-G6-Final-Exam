import sys
import threading
import time
import numpy as np
from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QCheckBox, QSlider
)
from PyQt6.QtCore import Qt, QTimer
import pyaudio

# --- Cấu hình Audio (Audio Configuration) ---
CHUNK = 1024
FORMAT = pyaudio.paInt16  # Định dạng 16-bit
CHANNELS = 1
RATE = 16000  # Tốc độ lấy mẫu 16kHz, lý tưởng cho xử lý giọng nói


# --- Lớp Ứng dụng AEC (AEC Application Class) ---
class AECLoopbackApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Mô phỏng Khử Tiếng Vọng (AEC)")
        self.setGeometry(100, 100, 500, 350)
        self.p = None
        self.stream = None
        self.is_running = False
        self.aec_enabled = False
        self.mic_volume_factor = 1.0

        # Biến đệm loa (Đây sẽ là tín hiệu xa cần thiết cho AEC thực tế)
        # We need a buffer to store the signal sent to the speaker (far-end signal)
        self.far_end_buffer = np.zeros(CHUNK * 2, dtype=np.int16)

        # --- NLMS Parameters for AEC Implementation ---
        # L: Chiều dài bộ lọc (số taps) - quyết định khả năng mô hình hóa tiếng vọng trễ
        # Tăng lên 1024 để mô hình hóa trễ dài hơn (~64ms)
        self.NLMS_FILTER_LENGTH = 1024
        # MU: Kích thước bước (step size) - Giảm tốc độ học để tăng ổn định
        self.NLMS_MU = 0.005
        # EPSILON: Hằng số chuẩn hóa nhỏ để tránh chia cho 0
        self.NLMS_EPSILON = 1e-6

        # Ngưỡng phát hiện hội thoại kép (DTD) - Nếu Năng lượng Cận > Năng lượng Xa * Ngưỡng, thì DTD
        self.DTD_THRESHOLD = 5.0

        # W: Trọng số (weights) của bộ lọc thích ứng
        self.nlms_weights = np.zeros(self.NLMS_FILTER_LENGTH, dtype=np.float64)

        # Buffer lịch sử cho tín hiệu xa (far-end history for NLMS taps)
        self.far_end_nlms_history = np.zeros(self.NLMS_FILTER_LENGTH, dtype=np.float64)

        self.init_ui()
        self.init_pyaudio()

    def init_pyaudio(self):
        """Khởi tạo PyAudio."""
        try:
            self.p = pyaudio.PyAudio()
        except Exception as e:
            self.status_label.setText(f"Lỗi PyAudio: {e}")
            self.start_button.setEnabled(False)

    def init_ui(self):
        """Khởi tạo giao diện người dùng."""
        main_layout = QVBoxLayout()
        main_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # Tiêu đề và Trạng thái
        title_label = QLabel("Acoustic Echo Cancellation (AEC) Demo - NLMS Implemented with DTD")
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

        self.aec_checkbox = QCheckBox("Bật AEC (NLMS + DTD)")
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
            "4. Bật <b>Bật AEC (NLMS + DTD)</b>. Quá trình hội tụ cần vài giây khi không có tiếng nói."
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
        """Bật hoặc tắt tính năng AEC và reset NLMS weights."""
        self.aec_enabled = state == Qt.CheckState.Checked.value
        if self.aec_enabled:
            # Reset trọng số khi bật AEC để bắt đầu lại quá trình hội tụ
            self.nlms_weights = np.zeros(self.NLMS_FILTER_LENGTH, dtype=np.float64)
            self.far_end_nlms_history = np.zeros(self.NLMS_FILTER_LENGTH, dtype=np.float64)

        self.status_label.setText(
            f"Trạng thái: Đang chạy | AEC: {'BẬT (NLMS + DTD)' if self.aec_enabled else 'TẮT'}"
        )

    def aec_process(self, near_end_frame: np.ndarray, far_end_frame: np.ndarray) -> np.ndarray:
        """
        *** TRIỂN KHAI THUẬT TOÁN AEC BẰNG NLMS (Normalized Least Mean Squares) CÓ DTD ***

        Hàm này thực hiện Khử Tiếng Vọng bằng bộ lọc thích ứng.
        """
        if not self.aec_enabled:
            # Nếu AEC TẮT, chỉ cần chuyển tiếp (loopback) tín hiệu gốc.
            return near_end_frame.astype(np.int16)

        # Chuyển đổi sang float64 cho tính toán NLMS (quan trọng cho độ chính xác)
        near_end_float = near_end_frame.astype(np.float64)
        far_end_float = far_end_frame.astype(np.float64)
        processed_output = near_end_float.copy()

        L = self.NLMS_FILTER_LENGTH
        mu = self.NLMS_MU
        epsilon = self.NLMS_EPSILON

        # --- Phát hiện Hội thoại kép (Double-Talk Detection - DTD) Heuristic ---
        # Tính năng lượng (công suất) của tín hiệu cận và tín hiệu xa
        near_energy = np.sum(near_end_float ** 2)
        far_energy = np.sum(far_end_float ** 2)

        # Nếu năng lượng cận vượt quá năng lượng xa một ngưỡng nhất định, giả định là Double-Talk
        # và đóng băng việc cập nhật trọng số.
        is_double_talk = near_energy > (far_energy * self.DTD_THRESHOLD)

        # Xử lý từng mẫu (sample-by-sample) trong khung (CHUNK)
        for i in range(len(near_end_frame)):
            # 1. Cập nhật buffer lịch sử (shift register) cho tín hiệu xa
            # Dịch chuyển các mẫu cũ đi và thêm mẫu xa mới nhất vào đầu
            self.far_end_nlms_history = np.roll(self.far_end_nlms_history, 1)
            self.far_end_nlms_history[0] = far_end_float[i]

            # Lấy vector tín hiệu lịch sử (x)
            x = self.far_end_nlms_history

            # 2. Ước tính Tiếng Vọng (y_hat)
            # y_hat(n) = W^T * X(n)
            y_hat = np.dot(self.nlms_weights, x)

            # 3. Tính Sai số (e - Tín hiệu đã khử vọng)
            # e(n) = d(n) - y_hat(n)
            d = near_end_float[i]
            error = d - y_hat

            # 4. Cập nhật Trọng số (W) theo NLMS CHỈ KHI KHÔNG CÓ DOUBLE-TALK
            if not is_double_talk:
                # W(n+1) = W(n) + [mu / (epsilon + ||X(n)||^2)] * e(n) * X(n)
                x_norm_sq = np.dot(x, x)
                # Tránh lỗi nếu x_norm_sq quá lớn, chỉ lấy x_norm_sq + epsilon
                normalization_factor = epsilon + x_norm_sq

                update_factor = (mu / normalization_factor) * error
                self.nlms_weights = self.nlms_weights + update_factor * x

            # 5. Lưu kết quả đã xử lý (sai số) - đây chính là tín hiệu sau khi khử vọng
            processed_output[i] = error

        # Đảm bảo trọng số không quá lớn (Clamping weights for stability)
        self.nlms_weights = np.clip(self.nlms_weights, -1e10, 1e10)

        # Chuyển đổi lại sang định dạng int16 và giới hạn phạm vi [-32768, 32767]
        return np.clip(processed_output, -32768, 32767).astype(np.int16)

    def audio_callback(self, in_data, frame_count, time_info, status):
        """
        Hàm Callback PyAudio: Được gọi khi có dữ liệu từ microphone.
        Thực hiện Loopback và xử lý AEC.
        """
        try:
            # 1. Chuyển đổi dữ liệu input (bytes) sang mảng numpy (tín hiệu cận)
            near_end_frame = np.frombuffer(in_data, dtype=np.int16)

            # 2. Lấy tín hiệu xa (far_end_frame) từ buffer (tín hiệu đã phát ra trước đó)
            # Đây là tín hiệu cần đồng bộ hóa với tín hiệu mic để ước tính echo
            far_end_frame = self.far_end_buffer[:frame_count]

            # 3. Xử lý AEC bằng NLMS
            processed_frame = self.aec_process(near_end_frame, far_end_frame)

            # 4. Tăng âm lượng đầu ra để làm tiếng vọng rõ ràng hơn (dù AEC BẬT hay TẮT)
            output_frame = (processed_frame * self.mic_volume_factor).clip(-32768, 32767).astype(np.int16)

            # 5. Cập nhật buffer tín hiệu xa cho lần gọi tiếp theo (quan trọng cho bước 2)
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
                f"Trạng thái: Đang chạy | AEC: {'BẬT (NLMS + DTD)' if self.aec_enabled else 'TẮT'}"
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

        # Reset NLMS weights khi dừng
        self.nlms_weights = np.zeros(self.NLMS_FILTER_LENGTH, dtype=np.float64)
        self.far_end_nlms_history = np.zeros(self.NLMS_FILTER_LENGTH, dtype=np.float64)

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