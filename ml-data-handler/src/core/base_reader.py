class BaseReader:
    def read_data(self):
        raise NotImplementedError("Subclasses should implement this method.")

    def get_details(self):
        raise NotImplementedError("Subclasses should implement this method.")