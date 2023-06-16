"""Test model learning"""
import io
import unittest
import PIL.Image
import requests
from requests import request


class MyTestCase(unittest.TestCase):
    """Test class"""

    def test_home(self):
        """test home page /"""
        response = requests.request("GET", "http://localhost:1800/", timeout=10)
        sample = response.content.decode()
        self.assertEqual(sample, "Home page")  # add assertion here

    def test_classify(self):
        """test /classify/imagenet page"""
        img = PIL.Image.open("data/raw/dog.jpg")
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG")

        with buffer as buf:
            buffer.seek(0)
            response = request(
                "POST", "http://localhost:1800/classify/imagenet", data=buf, timeout=10
            )

        out = response.content.decode("utf-8")
        print(out)
        expected = "Пембрук"

        self.assertIn(expected, out)

    def test_binary_classify(self):
        img = PIL.Image.open("data/raw/cat.jpg")
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG")

        with buffer as buf:
            buffer.seek(0)
            response = request(
                "POST", "http://localhost:1800/classify/binary", data=buf
            )

        out = response.content.decode("utf-8")
        # print(out)
        expected = "Cat"
        self.assertEqual(expected, out)


if __name__ == "__main__":
    unittest.main()
