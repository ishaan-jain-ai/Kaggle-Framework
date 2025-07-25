class TextReader(BaseReader):
    def __init__(self, file_path):
        self.file_path = file_path
        self.content = None

    def read_data(self):
        with open(self.file_path, 'r') as file:
            self.content = file.read()

    def get_details(self):
        if self.content is None:
            return None
        num_lines = len(self.content.splitlines())
        size = len(self.content)
        return {
            'num_lines': num_lines,
            'size': size
        }