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

    def test_mnist_vae(self) -> None:
        folder = os.path.join(examples_folder, "vae")
        self.assertEqual(
            os.system(f"python {os.path.join(folder, 'mnist_vae.py')} --ci 1"),
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

    def test_mnist_gan(self) -> None:
        folder = os.path.join(examples_folder, "gan")
        self.assertEqual(
            os.system(f"python {os.path.join(folder, 'mnist_gan.py')} --ci 1"),
            0,
        )

    def test_mnist_ae(self) -> None:
        try:
            import albumentations
        except:
            return
        folder = os.path.join(examples_folder, "ae")
        self.assertEqual(
            os.system(f"python {os.path.join(folder, 'mnist_ae_kl.py')} --ci 1"),
            0,
        )
        self.assertEqual(
            os.system(f"python {os.path.join(folder, 'mnist_ae_vq.py')} --ci 1"),
            0,
        )

    def test_mnist_diffusion(self) -> None:
        try:
            import albumentations
        except:
            return
        folder = os.path.join(examples_folder, "diffusion")
        self.assertEqual(
            os.system(f"python {os.path.join(folder, 'mnist_ddpm.py')} --ci 1"),
            0,
        )
        self.assertEqual(
            os.system(f"python {os.path.join(folder, 'mnist_ldm.py')} --ci 1"),
            0,
        )
        self.assertEqual(
            os.system(f"python {os.path.join(folder, 'mnist_ldm_vq.py')} --ci 1"),
            0,
        )


if __name__ == "__main__":
    unittest.main()
