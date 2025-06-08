import numpy as np

try:
    from scipy.signal import convolve2d
except Exception:  # pragma: no cover - scipy optional
    def convolve2d(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        kh, kw = kernel.shape
        pad_h, pad_w = kh // 2, kw // 2
        padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode="edge")
        output = np.zeros_like(image, dtype=float)
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                region = padded[i : i + kh, j : j + kw]
                output[i, j] = np.sum(region * kernel)
        return output


class LaneDetector:
    """Simple lane detection utility using Sobel edge filters."""

    def __init__(self, threshold: int = 50):
        self.threshold = threshold

    def detect(self, image: np.ndarray) -> np.ndarray:
        """Return an image keeping only lane edges."""
        if image.ndim == 3:
            gray = image.mean(axis=2)
        else:
            gray = image.astype(float)

        kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=float)
        ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=float)

        gx = convolve2d(gray, kx)
        gy = convolve2d(gray, ky)
        g = np.hypot(gx, gy)
        g *= 255.0 / max(g.max(), 1)
        edges = (g > self.threshold).astype(np.uint8) * 255

        # replicate edges to 3 channels to keep original shape
        if image.ndim == 3:
            edges = np.stack([edges] * 3, axis=-1)
        return edges
