import numpy as np

class SimpleNLMS:
    def __init__(self, filter_length=256, mu=0.2, eps=1e-6):
        """
        :param filter_length: Số lượng trọng số (tap) của bộ lọc
        :param mu: Tốc độ học (Step size). 0 < mu < 2.
        :param eps: Hằng số nhỏ để tránh chia cho 0
        """
        self.filter_length = filter_length
        self.mu = mu
        self.eps = eps

        # W: Vector trọng số bộ lọc (Filter Weights), khởi tạo bằng 0
        self.w = np.zeros(filter_length)

        # x_buffer: Bộ đệm lịch sử tín hiệu tham chiếu (Reference History)
        # Lưu giữ 'filter_length' mẫu gần nhất từ Loa.
        self.x_buffer = np.zeros(filter_length)


    def process(self, mic_signal, ref_signal):
        """
        :param mic_signal: Tín hiệu từ Micro (d(n)) - bao gồm tiếng nói + vọng.
        :param ref_signal: Tín hiệu từ Loa (x(n)) - tín hiệu tham chiếu.
        :return: Tín hiệu lỗi (e(n)) - chính là giọng nói đã khử vọng.
        """
        n_samples = len(mic_signal)
        output_e = np.zeros(n_samples)  # Mảng chứa kết quả đầu ra

        # NLMS sample-by-sample
        for i in range(n_samples):
            # 1. Cập nhật bộ đệm tham chiếu (Shift & Insert)
            current_ref_sample = ref_signal[i]

            # Kỹ thuật: Dịch phải buffer 1 đơn vị, bỏ mẫu cũ nhất, thêm mẫu mới vào đầu
            self.x_buffer[1:] = self.x_buffer[:-1]
            self.x_buffer[0] = current_ref_sample

            # 2. Ước tính tiếng vọng
            # y(n) = w^T * x(n)
            y_est = np.dot(self.w, self.x_buffer)

            # 3. Tính sai số (Error Calculation)
            # e(n) = d(n) - y(n)
            # Tín hiệu mic hiện tại - Tiếng vọng ước tính
            e = mic_signal[i] - y_est

            # 4. Tính năng lượng tín hiệu tham chiếu (Power Norm)
            # ||x(n)||^2
            x_energy = np.dot(self.x_buffer, self.x_buffer)

            # 5. Cập nhật trọng số bộ lọc (Weight Update)
            # w(n+1) = w(n) + [mu / (||x||^2 + eps)] * e(n) * x(n)
            step_size = self.mu / (x_energy + self.eps)
            self.w = self.w + step_size * e * self.x_buffer

            # Lưu kết quả sạch vào output
            output_e[i] = e
        return output_e