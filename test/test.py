"""Test for image classifire"""
import io
import unittest
import PIL.Image
import requests
from requests import request


class MyTestCase(unittest.TestCase):
    """Test Class"""

    def test_home(self):
        """test home page /"""
        response = requests.request(
            "GET", "http://localhost:1800/", timeout=10
            )
        sample = response.content.decode()
        self.assertEqual(sample, "Home page")  # add assertion here

    def test_classify(self):
        """test classify page"""
        img = PIL.Image.open("/Users/leonid/fa/pabd/pabd_cv/data/dog.jpg")
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG")

        with buffer as buf:
            buffer.seek(0)
            response = request(
                "POST", "http://localhost:1800/classify", data=buf, timeout=10
            )

        out = response.content.decode("utf-8")
        expected = "Пембрук"

        self.assertIn(expected, out)


if __name__ == "__main__":
    unittest.main()
