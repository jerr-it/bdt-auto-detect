
from probstructs import CountMinSketch


def create_instance(filename, depth, width):
    cms = CMSketch(filename, depth, width)
    cms.load_from_file(filename)
    return cms


class CMSketch(CountMinSketch):
    def __init__(self, filename: str, depth, width):
        super().__init__(depth, width)
        self.depth = depth
        self.width = width
        self.filename = filename

    def __reduce__(self):
        self.save_to_file(self.filename)
        return (create_instance, (self.filename, self.depth, self.width))