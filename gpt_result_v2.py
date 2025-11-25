import sys
import numpy as np
import sounddevice as sd
from PyQt5 import QtWidgets, QtCore
from pyaec import Aec

SAMPLE_RATE = 16000
FRAME = 1024
DTYPE = "int16"


class AECRealtime:
    def __init__(self):
        self.running = False
        # self.aec = Aec(FRAME, SAMPLE_RATE)
        self.aec = Aec(FRAME, 128, SAMPLE_RATE, True)

        # double buffers cho render (far-end)
        self.render_buf = np.zeros(FRAME, dtype=np.int16)

        self.stream = sd.Stream(
            samplerate=SAMPLE_RATE,
            blocksize=FRAME,
            dtype=DTYPE,
            channels=1,
            callback=self.callback
        )

    def start(self):
        if not self.running:
            self.running = True
            self.stream.start()

    def stop(self):
        if self.running:
            self.running = False
            self.stream.stop()

    def callback(self, indata, outdata, frames, time_info, status):
        if not self.running:
            outdata[:] = np.zeros_like(outdata)
            return

        mic = indata[:, 0].astype(np.int16)
        far = self.render_buf

        # Gọi pyaec
        clean = self.aec.cancel_echo(mic, far)

        # Xuất ra loa
        outdata[:, 0] = clean

        # cập nhật render để dùng cho frame sau
        self.render_buf = clean.copy()


# ---------------- GUI -----------------

class MainWindow(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Realtime AEC (pyaec) — Mai Văn An")

        self.engine = AECRealtime()

        self.start_btn = QtWidgets.QPushButton("Start AEC")
        self.stop_btn = QtWidgets.QPushButton("Stop")

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.start_btn)
        layout.addWidget(self.stop_btn)
        self.setLayout(layout)

        self.start_btn.clicked.connect(self.start_aec)
        self.stop_btn.clicked.connect(self.stop_aec)

    def start_aec(self):
        self.engine.start()

    def stop_aec(self):
        self.engine.stop()


def main():
    app = QtWidgets.QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
