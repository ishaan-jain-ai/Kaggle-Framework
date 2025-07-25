class ImageReader(BaseReader):
    def __init__(self, file_path):
        self.file_path = file_path
        self.image = None

    def read_data(self):
        from PIL import Image
        self.image = Image.open(self.file_path)
        return self.image

    def get_details(self):
        if self.image is not None:
            width, height = self.image.size
            size = self.image.tell()  # This may not work as expected; consider using os.path.getsize
            return {
                "width": width,
                "height": height,
                "size": size,
                "format": self.image.format
            }
        else:
            raise ValueError("Image not read. Please call read_data() first.")