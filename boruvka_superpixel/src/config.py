class Feature(object):
    def __init__(self, name, root_folder, prefactors):
        self.NAME = name
        self.ROOT_FOLDER = root_folder
        self.PREFACTORS = prefactors
        
class Config(object):
    ROOT_FOLDER = None
    OUT_ROOT_FOLDER = None
    PROCESS_NUMBER = 10
    SAVE_VIDEO = True

    def display(self):
        """Display Configuration values."""
        print("\nConfigurations:")
        for a in dir(self):
            if not a.startswith("__") and not callable(getattr(self, a)):
                print("{:30} {}".format(a, getattr(self, a)))
        print("\n")