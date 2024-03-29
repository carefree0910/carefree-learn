import numpy as np
import torch

from .model import handpose_model
from . import util

try:
    import cv2
except:
    cv2 = None
try:
    from scipy.ndimage import gaussian_filter
except:
    gaussian_filter = None
try:
    from skimage.measure import label
except:
    label = None


class Hand(object):
    def __init__(self, model_path, device="cpu"):
        if cv2 is None:
            raise RuntimeError("`cv2` is needed for `Body`")
        if gaussian_filter is None:
            raise RuntimeError("`scipy` is needed for `Body`")
        if label is None:
            raise RuntimeError("`scikit-image` is needed for `Hand`")
        self.model = handpose_model()
        model_dict = util.transfer(self.model, torch.load(model_path))
        self.model.load_state_dict(model_dict)
        self.model.eval()
        self.model.to(device)

    def __call__(self, oriImg, to):
        scale_search = [0.5, 1.0, 1.5, 2.0]
        # scale_search = [0.5]
        boxsize = 368
        stride = 8
        padValue = 128
        thre = 0.05
        multiplier = [x * boxsize / oriImg.shape[0] for x in scale_search]
        heatmap_avg = np.zeros((oriImg.shape[0], oriImg.shape[1], 22))
        # paf_avg = np.zeros((oriImg.shape[0], oriImg.shape[1], 38))

        for m in range(len(multiplier)):
            scale = multiplier[m]
            imageToTest = cv2.resize(
                oriImg, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC
            )
            imageToTest_padded, pad = util.padRightDownCorner(
                imageToTest, stride, padValue
            )
            im = (
                np.transpose(
                    np.float32(imageToTest_padded[:, :, :, np.newaxis]), (3, 2, 0, 1)
                )
                / 256
                - 0.5
            )
            im = np.ascontiguousarray(im)

            data = to(torch.from_numpy(im))
            # data = data.permute([2, 0, 1]).unsqueeze(0).float()
            with torch.no_grad():
                output = self.model(data).cpu().float().numpy()
                # output = self.model(data).numpy()q

            # extract outputs, resize, and remove padding
            heatmap = np.transpose(
                np.squeeze(output), (1, 2, 0)
            )  # output 1 is heatmaps
            heatmap = cv2.resize(
                heatmap, (0, 0), fx=stride, fy=stride, interpolation=cv2.INTER_CUBIC
            )
            heatmap = heatmap[
                : imageToTest_padded.shape[0] - pad[2],
                : imageToTest_padded.shape[1] - pad[3],
                :,
            ]
            heatmap = cv2.resize(
                heatmap,
                (oriImg.shape[1], oriImg.shape[0]),
                interpolation=cv2.INTER_CUBIC,
            )

            heatmap_avg += heatmap / len(multiplier)

        all_peaks = []
        for part in range(21):
            map_ori = heatmap_avg[:, :, part]
            one_heatmap = gaussian_filter(map_ori, sigma=3)
            binary = np.ascontiguousarray(one_heatmap > thre, dtype=np.uint8)
            # 全部小于阈值
            if np.sum(binary) == 0:
                all_peaks.append([0, 0])
                continue
            label_img, label_numbers = label(
                binary, return_num=True, connectivity=binary.ndim
            )
            max_index = (
                np.argmax(
                    [
                        np.sum(map_ori[label_img == i])
                        for i in range(1, label_numbers + 1)
                    ]
                )
                + 1
            )
            label_img[label_img != max_index] = 0
            map_ori[label_img == 0] = 0

            y, x = util.npmax(map_ori)
            all_peaks.append([x, y])
        return np.array(all_peaks)


if __name__ == "__main__":
    hand_estimation = Hand("../model/hand_pose_model.pth")

    # test_image = '../images/hand.jpg'
    test_image = "../images/hand.jpg"
    oriImg = cv2.imread(test_image)  # B,G,R order
    peaks = hand_estimation(oriImg)
    canvas = util.draw_handpose(oriImg, peaks, True)
    cv2.imshow("", canvas)
    cv2.waitKey(0)
