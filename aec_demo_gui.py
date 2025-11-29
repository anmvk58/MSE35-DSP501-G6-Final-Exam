import sys
import pyaudio
import numpy as np
from pyaec import Aec
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QLabel, QPushButton, QCheckBox,
                             QComboBox, QGroupBox)
from PyQt6.QtCore import QThread, pyqtSignal, Qt

from nlms import SimpleNLMS


class AudioWorker(QThread):
    # Tín hiệu gửi trạng thái và lỗi về giao diện
    status_update = pyqtSignal(str)
    error_occurred = pyqtSignal(str)

    def __init__(self, input_device_index, output_device_index, use_aec=True):
        super().__init__()
        self.input_device_index = input_device_index
        self.output_device_index = output_device_index
        self.use_aec = use_aec
        self.is_running = True

        # Cấu hình âm thanh
        self.format = pyaudio.paInt16
        self.channels = 1
        self.rate = 16000
        self.frame_size = 160  # 10ms
        self.filter_length = int(self.rate * 0.2)  # Độ trễ lọc tối đa 0.2s

        # Cách 1: sử dụng thư viện pyaec: Khởi tạo AEC (đảm bảo nó nằm trong luồng run())
        # self.aec = Aec(self.frame_size, self.filter_length, self.rate, False)

        # Cách 2: Tự implement theo công thức nlms:
        self.nlms = SimpleNLMS(filter_length=self.filter_length, mu=0.1)

    def run(self):
        p = pyaudio.PyAudio()

        in_stream = None
        out_stream = None

        try:
            # Mở luồng Mic
            in_stream = p.open(format=self.format,
                               channels=self.channels,
                               rate=self.rate,
                               input=True,
                               input_device_index=self.input_device_index,
                               frames_per_buffer=self.frame_size)

            # Mở luồng Loa
            out_stream = p.open(format=self.format,
                                channels=self.channels,
                                rate=self.rate,
                                output=True,
                                output_device_index=self.output_device_index,
                                frames_per_buffer=self.frame_size)

            self.status_update.emit("Đang chạy Loopback...")

            # Buffer tham chiếu rỗng ban đầu (x mẫu * 2 bytes/mẫu)
            last_output_bytes = b'\x00' * self.frame_size * 2

            while self.is_running:
                try:
                    # 1. Đọc dữ liệu từ Mic (blocking operation)
                    in_data = in_stream.read(self.frame_size, exception_on_overflow=False)

                    # 2. Kiểm tra cờ dừng ngay sau khi thao tác blocking kết thúc
                    if not self.is_running:
                        break

                    in_samples = np.frombuffer(in_data, dtype=np.int16)
                    ref_samples = np.frombuffer(last_output_bytes, dtype=np.int16)

                    # 3. Xử lý AEC
                    if self.use_aec:
                        # # ============================
                        # # Cách 1: sử dụng thư viện pyaec
                        # processed_samples = self.aec.cancel_echo(in_samples, ref_samples)
                        # out_data = np.array(processed_samples, dtype=np.int16).tobytes()

                        # ============================
                        # Cách 2: sử dụng NLMS tự code:
                        # Thuật toán NLMS cần số thực, không phải số nguyên 16-bit.
                        mic_float = in_samples.astype(np.float32)
                        ref_float = ref_samples.astype(np.float32)

                        # Gọi hàm process của class tự viết
                        out_samples_float = self.nlms.process(mic_float, ref_float)
                        # Clip giá trị để tránh tràn số khi convert ngược lại int16
                        out_samples_float = np.clip(out_samples_float, -32768, 32767)

                        # Chuyển lại int16 để phát ra loa
                        out_data = out_samples_float.astype(np.int16).tobytes()

                    else:
                        out_data = in_data

                    # 4. Phát ra Loa
                    out_stream.write(out_data)

                    # 5. Lưu lại dữ liệu vừa phát để dùng cho vòng sau
                    last_output_bytes = out_data

                except IOError as e:
                    # Bỏ qua lỗi tràn bộ đệm (Overflow), tiếp tục vòng lặp
                    if "Input overflowed" in str(e):
                        continue
                    print(f"Lỗi vòng lặp audio: {e}")
                    break
                except Exception as e:
                    print(f"Lỗi vòng lặp audio không xác định: {e}")
                    break

            self.status_update.emit("Đang dọn dẹp...")

        except Exception as e:
            self.error_occurred.emit(f"Lỗi khởi tạo: {e}")

        finally:
            # Đảm bảo dọn dẹp tài nguyên PyAudio một cách an toàn
            if out_stream:
                try:
                    out_stream.stop_stream()
                except:
                    pass
                try:
                    out_stream.close()
                except:
                    pass
            if in_stream:
                try:
                    in_stream.stop_stream()
                except:
                    pass
                try:
                    in_stream.close()
                except:
                    pass

            p.terminate()
            self.status_update.emit("Đã dừng.")

    def stop(self):
        # Đặt cờ dừng để vòng lặp thoát an toàn
        self.is_running = False


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Python AEC Loopback")
        self.resize(400, 300)
        self.audio_thread = None
        self.pyaudio_instance = pyaudio.PyAudio()  # Khởi tạo một lần

        self.init_ui()

    def init_ui(self):
        # ... (Tạo giao diện như code trước) ...
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout()

        # --- Phần chọn thiết bị ---
        dev_group = QGroupBox("Cấu hình thiết bị")
        dev_layout = QVBoxLayout()

        dev_layout.addWidget(QLabel("Chọn Microphone:"))
        self.combo_input = QComboBox()
        self.populate_devices(self.combo_input, input=True)
        dev_layout.addWidget(self.combo_input)

        dev_layout.addWidget(QLabel("Chọn Loa (Speaker):"))
        self.combo_output = QComboBox()
        self.populate_devices(self.combo_output, input=False)
        dev_layout.addWidget(self.combo_output)

        dev_group.setLayout(dev_layout)
        layout.addWidget(dev_group)

        # --- Phần điều khiển ---
        ctrl_group = QGroupBox("Điều khiển")
        ctrl_layout = QVBoxLayout()

        self.chk_aec = QCheckBox("Bật khử vọng (AEC)")
        self.chk_aec.setChecked(True)
        self.chk_aec.setToolTip("Giúp giảm tiếng hú khi Mic gần Loa")
        ctrl_layout.addWidget(self.chk_aec)

        self.lbl_status = QLabel("Trạng thái: Sẵn sàng")
        self.lbl_status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_status.setStyleSheet("color: gray; font-weight: bold;")
        ctrl_layout.addWidget(self.lbl_status)

        self.btn_start = QPushButton("BẮT ĐẦU LOOPBACK")
        self.btn_start.setCheckable(True)
        self.btn_start.setStyleSheet("padding: 10px; font-weight: bold;")
        self.btn_start.clicked.connect(self.toggle_audio)
        ctrl_layout.addWidget(self.btn_start)

        ctrl_group.setLayout(ctrl_layout)
        layout.addWidget(ctrl_group)

        main_widget.setLayout(layout)

    def populate_devices(self, combobox, input=True):
        """Liệt kê các thiết bị âm thanh vào ComboBox"""
        try:
            info = self.pyaudio_instance.get_host_api_info_by_index(0)
            num_devices = info.get('deviceCount')

            for i in range(num_devices):
                device_info = self.pyaudio_instance.get_device_info_by_host_api_device_index(0, i)
                is_input = device_info.get('maxInputChannels') > 0
                is_output = device_info.get('maxOutputChannels') > 0

                if (input and is_input) or (not input and is_output):
                    combobox.addItem(f"{i}: {device_info.get('name')}", i)
        except Exception as e:
            # Xử lý nếu không tìm thấy thiết bị PyAudio
            combobox.addItem("Không tìm thấy thiết bị", -1)
            print(f"Lỗi liệt kê thiết bị: {e}")

    def toggle_audio(self):
        if self.btn_start.isChecked():
            # Bắt đầu
            in_idx = self.combo_input.currentData()
            out_idx = self.combo_output.currentData()

            if in_idx is None or out_idx is None or in_idx == -1 or out_idx == -1:
                self.lbl_status.setText("Trạng thái: Vui lòng chọn thiết bị!")
                self.btn_start.setChecked(False)
                return

            use_aec = self.chk_aec.isChecked()

            self.audio_thread = AudioWorker(in_idx, out_idx, use_aec)
            self.audio_thread.status_update.connect(self.update_status)
            self.audio_thread.error_occurred.connect(self.show_error)
            self.audio_thread.start()

            self.btn_start.setText("DỪNG")
            self.btn_start.setStyleSheet("background-color: #ffcccc; color: red; padding: 10px; font-weight: bold;")
            self.combo_input.setEnabled(False)
            self.combo_output.setEnabled(False)
            self.chk_aec.setEnabled(False)
        else:
            # Dừng
            if self.audio_thread and self.audio_thread.isRunning():
                # Báo cho luồng dừng vòng lặp
                self.audio_thread.stop()

                # CHỜ luồng kết thúc hoàn toàn (QUAN TRỌNG để tránh lỗi 0xC0000409)
                self.audio_thread.wait()

            self.btn_start.setText("BẮT ĐẦU LOOPBACK")
            self.btn_start.setStyleSheet("padding: 10px; font-weight: bold;")
            self.combo_input.setEnabled(True)
            self.combo_output.setEnabled(True)
            self.chk_aec.setEnabled(True)
            self.lbl_status.setText("Trạng thái: Sẵn sàng")

    def update_status(self, text):
        self.lbl_status.setText(f"Trạng thái: {text}")

    def show_error(self, err_msg):
        self.lbl_status.setText(f"Lỗi: {err_msg}")
        self.btn_start.setChecked(False)
        self.toggle_audio()  # Tự động đặt lại nút và trạng thái

    def closeEvent(self, event):
        if self.audio_thread and self.audio_thread.isRunning():
            self.audio_thread.stop()
            self.audio_thread.wait()
        self.pyaudio_instance.terminate()
        super().closeEvent(event)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())