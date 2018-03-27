from numpy import random
import matplotlib
from showit import image, tile
import matplotlib.pyplot as plt
matplotlib.use('Agg')

def test_image():
	im = random.randn(25, 25)
	image(im)

def test_tile():
	im = random.randn(5, 25, 25)
	tile(im)

im = random.rand(25, 25, 3)
print('aaa')
i = image(im)
plt.show()
ims = random.rand(9, 25, 25, 3)
tile(ims)