import sys
import time
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QCheckBox, QSlider
)
from PyQt6.QtCore import Qt, QTimer
import pyaudio

# --- Audio Configuration ---
CHUNK = 1024
FORMAT = pyaudio.paInt16  # Format 16-bit
CHANNELS = 1
RATE = 16000  # Sampling = 16kHz

# --- NLMS Parameters ---
FILTER_LEN = 1024          # length of adaptive filter (adjust for performance)
NLMS_MU = 0.5              # step size (0 < mu <= 1)
EPS = 1e-8                 # small constant for normalization


class AECLoopbackApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Mô phỏng Khử Tiếng Vọng (AEC) - NLMS")
        self.setGeometry(100, 100, 600, 350)

        self.p = None
        self.stream = None
        self.is_running = False
        self.aec_enabled = False
        self.mic_volume_factor = 1.0

        # far_end_buffer lưu các mẫu đã phát ra loa (newest ở cuối)
        # độ dài = FILTER_LEN + CHUNK - 1 để tạo sliding windows cho một block CHUNK
        self.far_end_buffer = np.zeros(FILTER_LEN + CHUNK - 1, dtype=np.int16)

        # NLMS weights (float32)
        self.w = np.zeros(FILTER_LEN, dtype=np.float32)

        self.init_ui()
        self.init_pyaudio()

    def init_pyaudio(self):
        try:
            self.p = pyaudio.PyAudio()
        except Exception as e:
            self.status_label.setText(f"Lỗi PyAudio: {e}")
            self.start_button.setEnabled(False)

    def init_ui(self):
        main_layout = QVBoxLayout()
        main_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        title_label = QLabel("Acoustic Echo Cancellation (AEC) Demo - NLMS")
        title_label.setStyleSheet("font-size: 20px; font-weight: bold; color: #1e40af;")
        main_layout.addWidget(title_label, alignment=Qt.AlignmentFlag.AlignCenter)

        self.status_label = QLabel("Trạng thái: Đã dừng")
        self.status_label.setStyleSheet("font-size: 14px; margin-bottom: 15px;")
        main_layout.addWidget(self.status_label, alignment=Qt.AlignmentFlag.AlignCenter)

        vol_layout = QHBoxLayout()
        vol_layout.addWidget(QLabel("Tăng Âm Lượng Loa (Gây Echo Mạnh hơn):"))
        self.vol_slider = QSlider(Qt.Orientation.Horizontal)
        self.vol_slider.setMinimum(10)
        self.vol_slider.setMaximum(300)
        self.vol_slider.setValue(100)
        self.vol_slider.setSingleStep(1)
        self.vol_slider.valueChanged.connect(self.update_volume)
        self.vol_display = QLabel("1.00x")
        vol_layout.addWidget(self.vol_slider)
        vol_layout.addWidget(self.vol_display)
        main_layout.addLayout(vol_layout)

        control_layout = QHBoxLayout()
        self.aec_checkbox = QCheckBox("Bật AEC (NLMS)")
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

        info_label = QLabel(
            "<br><b>Cách Demo:</b><br>"
            "1. Bấm <b>Bắt Đầu Loopback</b>.<br>"
            "2. Nói vào Microphone. Bạn sẽ nghe thấy giọng nói của mình bị trễ/vọng.<br>"
            "3. Tăng thanh trượt âm lượng loa để tiếng vọng mạnh hơn.<br>"
            "4. Bật <b>Bật AEC (NLMS)</b> để thấy sự khác biệt."
        )
        info_label.setStyleSheet("font-size: 12px; color: #6b7280; margin-top: 15px;")
        main_layout.addWidget(info_label)
        self.setLayout(main_layout)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.check_status)
        self.timer.start(100)

    def update_volume(self, value):
        self.mic_volume_factor = value / 100.0
        self.vol_display.setText(f"{self.mic_volume_factor:.2f}x")

    def toggle_aec(self, state):
        self.aec_enabled = state == Qt.CheckState.Checked.value
        self.status_label.setText(
            f"Trạng thái: {'Đang chạy' if self.is_running else 'Đã dừng'} | AEC: {'BẬT' if self.aec_enabled else 'TẮT'}"
        )

    def aec_process_nlms_block(self, near_end_frame: np.ndarray) -> np.ndarray:
        """
        Block-NLMS implementation.

        - near_end_frame: numpy array float32/int16 of current mic block (length = CHUNK)
        - self.far_end_buffer: contains past played samples (newest at the end),
          length = FILTER_LEN + CHUNK - 1 so we can create sliding windows for CHUNK outputs.
        Returns processed_frame (int16) which is near_end - estimated_echo (i.e., the residual).
        Also updates self.w (weights).
        """
        # convert to float32 for processing
        near_f = near_end_frame.astype(np.float32)

        # Create sliding windows of the far-end reference:
        # sliding_window_view returns shape (M, FILTER_LEN) where M = len(far_end_buffer) - FILTER_LEN + 1
        # We expect M >= CHUNK; take the last CHUNK rows to align with current block.
        buf = self.far_end_buffer.astype(np.float32)
        try:
            windows = sliding_window_view(buf, FILTER_LEN)  # shape (M, FILTER_LEN)
        except Exception as e:
            # fallback: if sliding_window_view not available or shapes wrong, do no AEC
            print("sliding_window_view error:", e)
            return near_end_frame

        # Ensure we have at least CHUNK rows; if not, pad buffer (shouldn't happen because of initial zeros)
        if windows.shape[0] < CHUNK:
            # pad front of buffer with zeros and recompute
            pad_needed = (CHUNK - windows.shape[0])
            pad_arr = np.zeros((pad_needed, FILTER_LEN), dtype=np.float32)
            X = np.vstack([pad_arr, windows])
        else:
            X = windows[-CHUNK:, :]  # shape (CHUNK, FILTER_LEN)

        # Predicted echo y_hat for each sample in block: y = X @ w  (shape CHUNK)
        # w is (FILTER_LEN,), X is (CHUNK, FILTER_LEN)
        y_hat = X.dot(self.w)  # float32

        # error (residual) = near - y_hat
        e = near_f - y_hat  # shape (CHUNK,)

        # NLMS block update:
        # w <- w + mu * (X^T e) / (sum of squared X rows per coefficient + eps)
        # Common block NLMS normalizer: sum over all rows of squared elements (trace(X^T X)) -> scalar
        denom = np.sum(X * X) + EPS  # scalar
        # compute gradient = X^T e  (shape FILTER_LEN,)
        grad = X.T.dot(e)  # (FILTER_LEN,)
        self.w += (NLMS_MU / denom) * grad

        # output residual as int16 scaled/clipped
        out = e.astype(np.float32) * self.mic_volume_factor
        out_clipped = np.clip(out, -32768.0, 32767.0).astype(np.int16)
        return out_clipped

    def audio_callback(self, in_data, frame_count, time_info, status):
        try:
            # 1. Convert input bytes to numpy array
            near_end_frame = np.frombuffer(in_data, dtype=np.int16)

            # 2. Depending on AEC flag, process
            if self.aec_enabled:
                processed_frame = self.aec_process_nlms_block(near_end_frame)
            else:
                processed_frame = near_end_frame.astype(np.int16)
                # apply mic volume to exaggerate echo if desired
                if self.mic_volume_factor != 1.0:
                    processed_frame = (processed_frame.astype(np.float32) * self.mic_volume_factor).clip(-32768, 32767).astype(np.int16)

            # 3. Update far_end_buffer with what we are about to play (so next callbacks see this as reference)
            # Shift left by frame_count and append the new output at the end
            self.far_end_buffer = np.roll(self.far_end_buffer, -frame_count)
            # ensure types align
            self.far_end_buffer[-frame_count:] = processed_frame

            # 4. Return bytes to be played
            return processed_frame.tobytes(), pyaudio.paContinue

        except Exception as e:
            print(f"Lỗi trong Callback: {e}")
            return in_data, pyaudio.paAbort

    def start_stream(self):
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
        if self.is_running:
            self.stop_stream()
        else:
            self.start_stream()

    def check_status(self):
        if self.is_running and self.stream and not self.stream.is_active():
            print("Luồng âm thanh bị ngắt. Dừng lại.")
            self.stop_stream()

    def closeEvent(self, event):
        self.stop_stream()
        if self.p:
            self.p.terminate()
        event.accept()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = AECLoopbackApp()
    window.show()
    sys.exit(app.exec())
