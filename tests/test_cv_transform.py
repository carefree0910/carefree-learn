import torch
import unittest

import numpy as np

from PIL import Image
from cflearn.api.cv.data.transforms import ToRGB
from cflearn.api.cv.data.transforms import ToGray


class TestTransform(unittest.TestCase):
    def test_to_rgb(self) -> None:
        for c in [1, 3, 4]:
            arr = (np.random.random([64, 64, c]) * 255.0).astype(np.uint8)
            image = Image.fromarray(arr if c > 1 else arr[..., 0])
            tensor = torch.from_numpy(arr.transpose(2, 0, 1))
            to_rgb = ToRGB()
            rgb_arr = to_rgb(arr)
            rgb_image_arr = np.array(to_rgb(image))
            rgb_tensor_arr = to_rgb(tensor).numpy().transpose([1, 2, 0])
            self.assertTrue(np.allclose(rgb_arr, rgb_image_arr))
            self.assertTrue(np.allclose(rgb_arr, rgb_tensor_arr))

    def test_to_gray(self) -> None:
        arr = (np.random.random([64, 64, 3]) * 255.0).astype(np.uint8)
        image = Image.fromarray(arr)
        tensor = torch.from_numpy(arr.transpose(2, 0, 1))
        to_gray = ToGray()
        gray_arr = to_gray(arr)[..., 0]
        gray_image_arr = np.array(to_gray(image))
        gray_tensor_arr = to_gray(tensor)[0].numpy()
        self.assertTrue(np.allclose(gray_arr, gray_tensor_arr))
        diff = gray_image_arr.astype(np.float32) - gray_tensor_arr.astype(np.float32)
        self.assertTrue(np.abs(diff).max().item() < 2)


if __name__ == "__main__":
    unittest.main()
