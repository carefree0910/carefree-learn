import os
import unittest


file_folder = os.path.dirname(__file__)
examples_folder = os.path.join(file_folder, os.pardir, "examples", "cv")


class TestExample(unittest.TestCase):
    def test_mnist_clf(self) -> None:
        folder = os.path.join(examples_folder, "classification")
        self.assertEqual(
            os.system(f"python {os.path.join(folder, 'mnist_clf.py')} --ci 1"),
            0,
        )

    def test_products_10k_clf(self) -> None:
        folder = os.path.join(examples_folder, "classification")
        self.assertEqual(
            os.system(f"python {os.path.join(folder, 'products-10k_clf.py')} --ci 1"),
            0,
        )

    def test_mnist_gan(self) -> None:
        folder = os.path.join(examples_folder, "gan")
        self.assertEqual(
            os.system(f"python {os.path.join(folder, 'mnist_gan.py')} --ci 1"),
            0,
        )

    def test_mnist_siren_gan(self) -> None:
        folder = os.path.join(examples_folder, "gan")
        self.assertEqual(
            os.system(f"python {os.path.join(folder, 'mnist_siren_gan.py')} --ci 1"),
            0,
        )

    def test_mnist_pixel_cnn(self) -> None:
        folder = os.path.join(examples_folder, "generator")
        self.assertEqual(
            os.system(f"python {os.path.join(folder, 'mnist_pixel_cnn.py')} --ci 1"),
            0,
        )

    def test_products_10k_seg(self) -> None:
        folder = os.path.join(examples_folder, "segmentation")
        self.assertEqual(
            os.system(f"python {os.path.join(folder, 'u2net_finetune.py')} --ci 1"),
            0,
        )
        self.assertEqual(
            os.system(f"python {os.path.join(folder, 'u2net_cascade.py')} --ci 1"),
            0,
        )

    def test_dino(self) -> None:
        folder = os.path.join(examples_folder, "ssl")
        self.assertEqual(
            os.system(f"python {os.path.join(folder, 'dino.py')} --ci 1"),
            0,
        )

    def test_adain(self) -> None:
        folder = os.path.join(examples_folder, "style_transfer")
        self.assertEqual(
            os.system(f"python {os.path.join(folder, 'adain.py')} --ci 1"),
            0,
        )

    def test_mnist_vae(self) -> None:
        folder = os.path.join(examples_folder, "vae")
        self.assertEqual(
            os.system(f"python {os.path.join(folder, 'mnist_vae.py')} --ci 1"),
            0,
        )

    def test_mnist_vae2d(self) -> None:
        folder = os.path.join(examples_folder, "vae")
        self.assertEqual(
            os.system(f"python {os.path.join(folder, 'mnist_vae2d.py')} --ci 1"),
            0,
        )

    def test_mnist_siren_vae(self) -> None:
        folder = os.path.join(examples_folder, "vae")
        self.assertEqual(
            os.system(f"python {os.path.join(folder, 'mnist_siren_vae.py')} --ci 1"),
            0,
        )

    def test_mnist_vq_vae(self) -> None:
        folder = os.path.join(examples_folder, "vae")
        self.assertEqual(
            os.system(f"python {os.path.join(folder, 'mnist_vq_vae.py')} --ci 1"),
            0,
        )
        cmd = f"python {os.path.join(folder, 'mnist_vq_vae_inference.py')} --ci 1"
        self.assertEqual(os.system(cmd), 0)


if __name__ == "__main__":
    TestExample().test_mnist_vq_vae()
