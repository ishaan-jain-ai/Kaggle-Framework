import unittest
from src.core.text_reader import TextReader

class TestTextReader(unittest.TestCase):

    def setUp(self):
        self.reader = TextReader()

    def test_read_data(self):
        # Assuming we have a sample text file for testing
        data = self.reader.read_data('sample.txt')
        self.assertIsInstance(data, str)  # Check if the data is read as a string

    def test_get_details(self):
        # Assuming we have a sample text file for testing
        self.reader.read_data('sample.txt')
        details = self.reader.get_details()
        self.assertIn('size', details)  # Check if size is in details
        self.assertIn('memory', details)  # Check if memory is in details

if __name__ == '__main__':
    unittest.main()