import os

from PIL import Image

import ui_backend
if __name__ == "__main__":
    i2l = ui_backend.Image2Latex()
    [print(x + ": " + i2l.predict(Image.open("testims/" + x))) for x in os.listdir("testims")]

