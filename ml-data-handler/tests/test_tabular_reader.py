import unittest
from src.core.tabular_reader import TabularReader

class TestTabularReader(unittest.TestCase):

    def setUp(self):
        self.reader = TabularReader()

    def test_read_data(self):
        # Assuming a method to set the file path for testing
        self.reader.set_file_path('test_data.csv')  # Replace with actual test data file
        data = self.reader.read_data()
        self.assertIsNotNone(data)
        self.assertIsInstance(data, list)  # Assuming data is returned as a list of rows

    def test_get_details(self):
        self.reader.set_file_path('test_data.csv')  # Replace with actual test data file
        self.reader.read_data()  # Read data first to populate details
        details = self.reader.get_details()
        self.assertIn('num_columns', details)
        self.assertIn('size', details)
        self.assertIn('memory', details)

if __name__ == '__main__':
    unittest.main()