import unittest
from src.core.image_reader import ImageReader

class TestImageReader(unittest.TestCase):

    def setUp(self):
        self.image_reader = ImageReader()

    def test_read_data(self):
        # Assuming a sample image path for testing
        image_path = 'path/to/sample/image.jpg'
        data = self.image_reader.read_data(image_path)
        self.assertIsNotNone(data)

    def test_get_details(self):
        # Assuming a sample image path for testing
        image_path = 'path/to/sample/image.jpg'
        self.image_reader.read_data(image_path)
        details = self.image_reader.get_details()
        self.assertIn('size', details)
        self.assertIn('memory', details)

if __name__ == '__main__':
    unittest.main()