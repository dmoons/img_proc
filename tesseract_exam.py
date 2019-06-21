# -*-encoding:utf-8-*-
# pip3 install pytesseract
# pip3 install PILLOW

import pytesseract
from PIL import Image

image = Image.open("images/code.png")
image = image.convert('L')
#image.show()
code = pytesseract.image_to_string(image, lang='eng', config='--psm 7')
print("code = ", code)
