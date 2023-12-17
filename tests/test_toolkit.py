import torch
import cflearn
import unittest

import torch.nn.functional as F

from cflearn.toolkit import *
from torch import Tensor
from pathlib import Path
from cftool.array import corr
from cftool.array import allclose
from unittest.mock import patch
from safetensors.torch import save_file


class TestToolkit(unittest.TestCase):
    def test_auto_num_layers(self) -> None:
        for img_size in [3, 7, 11, 23, 37, 53]:
            for min_size in [1, 2, 4, 8]:
                if min_size > img_size:
                    continue
                num_layers = auto_num_layers(img_size, min_size, None)
                if num_layers == 0:
                    self.assertTrue(img_size < 2 * min_size)

    def test_corr(self) -> None:
        pred = torch.randn(100, 5)
        target = torch.randn(100, 5)
        weights = torch.zeros(100, 1)
        weights[:30] = weights[-30:] = 1.0
        corr00 = corr(pred, pred, weights)
        corr01 = corr(pred, target, weights)
        corr02 = corr(target, pred, weights)
        w_pred = pred[list(range(30)) + list(range(70, 100))]
        w_target = target[list(range(30)) + list(range(70, 100))]
        corr10 = corr(w_pred, w_pred)
        corr11 = corr(w_pred, w_target)
        corr12 = corr(w_target, w_pred)
        self.assertTrue(allclose(corr00, corr10, atol=1.0e-5))
        self.assertTrue(allclose(corr01, corr11, corr02.t(), corr12.t(), atol=1.0e-5))

    def test_get_torch_device(self) -> None:
        cpu = torch.device("cpu")
        cuda1 = torch.device("cuda:1")
        self.assertTrue(get_torch_device(None) == cpu)
        self.assertTrue(get_torch_device("cpu") == cpu)
        self.assertTrue(get_torch_device(1) == cuda1)
        self.assertTrue(get_torch_device("1") == cuda1)
        self.assertTrue(get_torch_device("cuda:1") == cuda1)
        self.assertTrue(get_torch_device(cuda1) == cuda1)

    def test_seed(self) -> None:
        seed_everything(123)
        s0 = new_seed()
        seed_everything(123)
        s1 = new_seed()
        self.assertEqual(s0, s1)

    def test_seed_within_range(self) -> None:
        seed = 42
        with patch("cflearn.toolkit.random.seed") as mock_random_seed, patch(
            "cflearn.toolkit.np.random.seed"
        ) as mock_np_seed, patch(
            "cflearn.toolkit.torch.manual_seed"
        ) as mock_torch_seed, patch(
            "cflearn.toolkit.torch.cuda.manual_seed_all"
        ) as mock_cuda_seed_all:
            result = seed_everything(seed)
        self.assertEqual(result, seed)
        mock_random_seed.assert_called_once_with(seed)
        mock_np_seed.assert_called_once_with(seed)
        mock_torch_seed.assert_called_once_with(seed)
        mock_cuda_seed_all.assert_called_once_with(seed)

    def test_seed_outside_range(self) -> None:
        for seed in [min_seed_value - 1, max_seed_value + 1]:
            new_seed = 42
            with patch(
                "cflearn.toolkit.new_seed", return_value=new_seed
            ) as mock_new_seed, patch(
                "cflearn.toolkit.random.seed"
            ) as mock_random_seed, patch(
                "cflearn.toolkit.np.random.seed"
            ) as mock_np_seed, patch(
                "cflearn.toolkit.torch.manual_seed"
            ) as mock_torch_seed, patch(
                "cflearn.toolkit.torch.cuda.manual_seed_all"
            ) as mock_cuda_seed_all:
                result = seed_everything(seed)
            self.assertEqual(result, new_seed)
            mock_new_seed.assert_called_once()
            mock_random_seed.assert_called_once_with(new_seed)
            mock_np_seed.assert_called_once_with(new_seed)
            mock_torch_seed.assert_called_once_with(new_seed)
            mock_cuda_seed_all.assert_called_once_with(new_seed)

    def test_get_file_info(self) -> None:
        text = "This is a test file."
        test_file = Path("test_file.txt")
        test_file.write_text(text)

        file_info = get_file_info(test_file)

        self.assertIsInstance(file_info, FileInfo)
        self.assertEqual(file_info.sha, hashlib.sha256(text.encode()).hexdigest())
        self.assertEqual(file_info.st_size, len(text))

        test_file.unlink()

    def test_check_sha_with_matching_hash(self) -> None:
        path = Path("test_file.txt")
        path.write_text("This is a test file.")
        tgt_sha = hashlib.sha256(path.read_bytes()).hexdigest()

        result = check_sha_with(path, tgt_sha)

        self.assertTrue(result)

        path.unlink()

    def test_check_sha_with_non_matching_hash(self) -> None:
        path = Path("test_file.txt")
        path.write_text("This is a test file.")
        tgt_sha = "0"

        result = check_sha_with(path, tgt_sha)

        self.assertFalse(result)

        path.unlink()

    def test_get_download_path_info(self) -> None:
        ckpt_root = get_download_root(DownloadDtype.CHECKPOINTS)
        download_path_info = get_download_path_info(DownloadDtype.CHECKPOINTS, "ldm.sd")
        self.assertEqual(download_path_info.download_root, ckpt_root)
        self.assertEqual(download_path_info.download_path, ckpt_root / "ldm.sd.pt")

    def test_download_existing_file(self) -> None:
        dtype = DownloadDtype.CHECKPOINTS
        tag = "lpips"
        target = get_download_path_info(dtype, tag).download_path
        existing = target.is_file()
        downloaded = download(dtype, tag)
        self.assertEqual(target, downloaded)
        if not existing:
            downloaded.unlink()

    def test_specific_download(self) -> None:
        download_checkpoint("lpips")
        download_json("sd_mapping")

    def test_show_or_return(self) -> None:
        try:
            import matplotlib.pyplot as plt

            plt.figure()
            self.assertIsInstance(show_or_return(True), np.ndarray)
            plt.close()
        except:
            with self.assertRaises(RuntimeError):
                show_or_return(True)

    def _test_decay(self, num: int, decay: str, expected: np.ndarray) -> None:
        ws = WeightsStrategy(decay)
        weights = ws(num)
        self.assertIsInstance(weights, np.ndarray)
        self.assertEqual(weights.shape, (num,))  # type: ignore
        np.testing.assert_allclose(weights, expected)  # type: ignore

    def test_linear_decay(self) -> None:
        expected_weights = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        self._test_decay(10, "linear_decay", expected_weights)

    def test_radius_decay(self) -> None:
        num = 10
        expected_weights = np.sin(np.arccos(1.0 - np.linspace(0, 1, num + 1)[1:]))
        self._test_decay(num, "radius_decay", expected_weights)

    def test_log_decay(self) -> None:
        num = 10
        excepted_weights = np.log(np.arange(num) + np.e)
        self._test_decay(num, "log_decay", excepted_weights)

    def test_sigmoid_decay(self) -> None:
        num = 10
        expected_weights = 1.0 / (1.0 + np.exp(-np.linspace(-5.0, 5.0, num)))
        self._test_decay(num, "sigmoid_decay", expected_weights)

    def test_no_decay(self) -> None:
        ws = WeightsStrategy(None)
        num = 10
        weights = ws(num)
        self.assertIsNone(weights)

    def test_decay_visualize(self) -> None:
        ws = WeightsStrategy("linear_decay")
        try:
            import matplotlib.pyplot

            ws.visualize()
        except:
            with self.assertRaises(RuntimeError):
                ws.visualize()

    def test_sdp_attn_with_xformers(self) -> None:
        q = torch.randn(3, 32, 64)
        k = torch.randn(3, 32, 64)
        v = torch.randn(3, 32, 64)
        training = True
        mask = None
        dropout = None

        with patch(
            "cflearn.toolkit.try_run_xformers_sdp_attn",
            return_value=torch.randn(3, 32, 64),
        ):
            output = sdp_attn(q, k, v, training, mask, dropout)

        self.assertIsInstance(output, Tensor)
        self.assertEqual(output.shape, (3, 32, 64))

    def test_sdp_attn_without_xformers(self) -> None:
        q = torch.randn(3, 32, 64)
        k = torch.randn(3, 32, 64)
        v = torch.randn(3, 32, 64)
        training = True
        mask = torch.randn(3, 32, 32)
        dropout = 0.5

        with patch(
            "cflearn.toolkit.try_run_xformers_sdp_attn",
            return_value=None,
        ):
            output = sdp_attn(q, k, v, training, mask, dropout)

        self.assertIsInstance(output, Tensor)
        self.assertEqual(output.shape, (3, 32, 64))

    def test_sdp_attn_without_xformers_and_pt2_sdp_attn(self) -> None:
        q = torch.randn(3, 32, 64)
        k = torch.randn(3, 32, 64)
        v = torch.randn(3, 32, 64)
        training = True
        mask = torch.randint(0, 2, (3, 32, 32)).bool()
        dropout = 0.0

        with patch(
            "cflearn.toolkit.try_run_xformers_sdp_attn",
            return_value=None,
        ), patch.object(
            F,
            "scaled_dot_product_attention",
            None,
        ):
            output = sdp_attn(q, k, v, training, mask, dropout)

        self.assertIsInstance(output, Tensor)
        self.assertEqual(output.shape, (3, 32, 64))
        expected = F.scaled_dot_product_attention(q, k, v, mask, dropout)
        torch.testing.assert_close(output, expected)

    def test_get_tensors_from_safetensors(self) -> None:
        path = Path("example.safetensors")
        expected_tensors = {
            "tensor1": torch.tensor([1, 2, 3]),
            "tensor2": torch.tensor([4, 5, 6]),
        }
        save_file(expected_tensors, path)

        tensors = get_tensors(path)

        self.assertIsInstance(tensors, dict)
        torch.testing.assert_close(tensors, expected_tensors)
        path.unlink()

    def test_get_tensors_from_pt(self) -> None:
        file_path = "example.pt"
        expected_tensors = {
            "tensor1": torch.tensor([1, 2, 3]),
            "tensor2": torch.tensor([4, 5, 6]),
        }
        torch.save(expected_tensors, file_path)

        tensors = get_tensors(file_path)

        self.assertIsInstance(tensors, dict)
        torch.testing.assert_close(tensors, expected_tensors)
        os.remove(file_path)

    def test_get_tensors_from_state_dict(self) -> None:
        state_dict = {
            "state_dict": {
                "tensor1": torch.tensor([1, 2, 3]),
                "tensor2": torch.tensor([4, 5, 6]),
            }
        }
        expected_tensors = {
            "tensor1": torch.tensor([1, 2, 3]),
            "tensor2": torch.tensor([4, 5, 6]),
        }

        tensors = get_tensors(state_dict)

        self.assertIsInstance(tensors, dict)
        torch.testing.assert_close(tensors, expected_tensors)

    def test_get_tensors_from_dict(self) -> None:
        tensor_dict = {
            "tensor1": torch.tensor([1, 2, 3]),
            "tensor2": torch.tensor([4, 5, 6]),
        }
        expected_tensors = {
            "tensor1": torch.tensor([1, 2, 3]),
            "tensor2": torch.tensor([4, 5, 6]),
        }

        tensors = get_tensors(tensor_dict)

        self.assertIsInstance(tensors, dict)
        torch.testing.assert_close(tensors, expected_tensors)

    def test_fix_denormal_states(self) -> None:
        states = {
            "a": torch.tensor([1.0, 2.0, 1.0e-33]),
            "b": torch.tensor([4.0, 5.0, 6.0]),
        }
        expected_states = {
            "a": torch.tensor([1.0, 2.0, 0.0]),
            "b": torch.tensor([4.0, 5.0, 6.0]),
        }

        new_states = fix_denormal_states(states)

        self.assertIsInstance(new_states, dict)
        torch.testing.assert_close(new_states, expected_states)

    def test_has_batch_norms_with_batch_norm_layers(self) -> None:
        m = nn.Sequential(nn.Linear(10, 2), nn.BatchNorm1d(2))

        result = has_batch_norms(m)

        self.assertTrue(result)

    def test_has_batch_norms_without_batch_norm_layers(self) -> None:
        m = nn.Sequential(nn.Linear(10, 2), nn.ReLU())

        result = has_batch_norms(m)

        self.assertFalse(result)

    def test_sorted_param_diffs(self) -> None:
        m1 = nn.Linear(10, 2)
        m2 = nn.Linear(10, 2)

        diffs = sorted_param_diffs(m1, m2)

        self.assertIsInstance(diffs, Diffs)
        self.assertEqual(diffs.names1, ["weight", "bias"])
        self.assertEqual(diffs.names2, ["weight", "bias"])
        self.assertEqual(len(diffs.diffs), 2)
        self.assertIsInstance(diffs.diffs[0], Tensor)
        self.assertIsInstance(diffs.diffs[1], Tensor)

    def test_sorted_param_diffs_with_different_lengths(self) -> None:
        m1 = nn.Linear(10, 2)
        m2 = nn.Sequential(nn.Linear(10, 2), nn.Linear(2, 2))

        with self.assertRaises(ValueError):
            sorted_param_diffs(m1, m2)

    def test_initialize_builtin_method(self) -> None:
        initializer = Initializer()
        m = nn.Linear(10, 2)
        param = m.weight

        initializer.initialize(param, "xavier_uniform")

        self.assertIsInstance(param, Tensor)

    def test_initialize_custom_method(self) -> None:
        initializer = Initializer()
        m = nn.Linear(10, 2)
        param = m.weight

        @Initializer.register("custom")
        def custom(self: Initializer, param: nn.Parameter) -> None:
            with torch.no_grad():
                param.data.fill_(1.0)

        initializer.initialize(param, "custom")

        self.assertIsInstance(param, Tensor)
        torch.testing.assert_close(param, torch.ones_like(param))

    def test_xavier_uniform(self) -> None:
        initializer = Initializer()
        m = nn.Linear(10, 2)
        param = m.weight

        initializer.xavier_uniform(param)

        self.assertIsInstance(param, Tensor)

    def test_xavier_normal(self) -> None:
        initializer = Initializer()
        m = nn.Linear(10, 2)
        param = m.weight

        initializer.xavier_normal(param)

        self.assertIsInstance(param, Tensor)

    def test_normal(self) -> None:
        initializer = Initializer()
        m = nn.Linear(10, 2)
        param = m.weight

        initializer.normal(param)

        self.assertIsInstance(param, Tensor)

    def test_truncated_normal(self) -> None:
        initializer = Initializer()
        m = nn.Linear(10, 2)
        param = m.weight

        initializer.truncated_normal(param)

        self.assertIsInstance(param, Tensor)

    def test_orthogonal(self) -> None:
        initializer = Initializer()
        m = nn.Linear(10, 2)
        param = m.weight

        initializer.orthogonal(param)

        self.assertIsInstance(param, Tensor)

    def test_onnx(self) -> None:
        try:
            import onnx
            from onnxruntime import InferenceSession
        except:
            return

        config = cflearn.DLConfig(
            module_name="linear",
            module_config=dict(input_dim=10, output_dim=1),
            loss_name="mse",
        )
        model = cflearn.IDLModel.from_config(config)
        model.m.net.weight.data.fill_(1)
        model.m.net.bias.data.zero_()

        onnx_path = "model.onnx"
        net = np.random.random((10, 10))
        inputs = {"input": net}
        model.to_onnx(onnx_path, np_batch_to_tensor(inputs))
        onnx = ONNX(onnx_path)

        outputs = onnx.predict(inputs)
        predictions = outputs[cflearn.PREDICTIONS_KEY]

        self.assertIsInstance(outputs, dict)
        self.assertEqual(len(outputs), 1)
        expected = net.sum(axis=1, keepdims=True)
        np.testing.assert_allclose(predictions, expected, rtol=1.0e-5, atol=1.0e-5)

        os.remove(onnx_path)

    def test_to_2d_with_none(self) -> None:
        arr = None

        result = to_2d(arr)

        self.assertIsNone(result)

    def test_to_2d_with_string(self) -> None:
        arr = "string"

        result = to_2d(arr)

        self.assertIsNone(result)

    def test_to_2d_with_1d_array(self) -> None:
        arr = np.array([1, 2, 3])
        expected_result = np.array([[1], [2], [3]])

        result = to_2d(arr)

        self.assertIsInstance(result, np.ndarray)
        np.testing.assert_array_equal(result, expected_result)  # type: ignore

    def test_to_2d_with_2d_array(self) -> None:
        arr = np.array([[1, 2], [3, 4]])
        expected_result = np.array([[1, 2], [3, 4]])

        result = to_2d(arr)

        self.assertIsInstance(result, np.ndarray)
        np.testing.assert_array_equal(result, expected_result)  # type: ignore

    def test_to_2d_with_1d_list(self) -> None:
        arr = [1, 2, 3]
        expected_result = [[1], [2], [3]]

        result = to_2d(arr)  # type: ignore

        self.assertIsInstance(result, list)
        np.testing.assert_array_equal(result, expected_result)  # type: ignore

    def test_to_2d_with_2d_list(self) -> None:
        arr = [[1], [2], [3]]
        expected_result = [[1], [2], [3]]

        result = to_2d(arr)  # type: ignore

        self.assertIsInstance(result, list)
        np.testing.assert_array_equal(result, expected_result)  # type: ignore

    def test_slerp(self) -> None:
        x1 = torch.tensor([[1.0, 0.0]])
        x2 = torch.tensor([[0.0, 1.0]])
        r1 = 0.5

        result = slerp(x1, x2, r1)

        self.assertIsInstance(result, torch.Tensor)
        self.assertEqual(result.shape, (1, 2))
        torch.testing.assert_close(result, torch.tensor([[0.7071, 0.7071]]))

    def test_mean_std(self) -> None:
        latent_map = torch.rand(1, 3, 4, 4)

        mean, std = mean_std(latent_map)
        expected_mean = latent_map.mean(dim=[0, 2, 3], keepdim=True)
        expected_std = latent_map.std(dim=[0, 2, 3], keepdim=True)

        self.assertIsInstance(mean, Tensor)
        self.assertIsInstance(std, Tensor)
        torch.testing.assert_close(mean, expected_mean, atol=1.0e-4, rtol=1.0e-4)
        torch.testing.assert_close(std, expected_std, atol=1.0e-4, rtol=1.0e-4)

    def test_adain_with_params(self) -> None:
        src = torch.rand(1, 3, 4, 4)
        mean = torch.rand(1, 3, 1, 1)
        std = torch.rand(1, 3, 1, 1)

        result = adain_with_params(src, mean, std)

        self.assertIsInstance(result, Tensor)
        self.assertEqual(result.shape, (1, 3, 4, 4))

    def test_adain_with_tensor(self) -> None:
        src = torch.rand(1, 3, 4, 4)
        tgt = torch.rand(1, 3, 4, 4)

        result = adain_with_tensor(src, tgt)

        self.assertIsInstance(result, Tensor)
        self.assertEqual(result.shape, (1, 3, 4, 4))


if __name__ == "__main__":
    unittest.main()
