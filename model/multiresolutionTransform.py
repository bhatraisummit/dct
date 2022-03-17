import multiresolution


class MultiResolution:
    def __init__(self):
        pass

    def __call__(self, pic):
        multiresolution_img = multiresolution.multiresolution_dct(pic)
        return multiresolution_img

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"