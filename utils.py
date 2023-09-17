import matplotlib.pyplot as plt
from PIL import Image

def show_image(images , figsize=(20,10), columns = 3):
   plt.figure(figsize=figsize)
   for i, image in enumerate(images):
      plt.subplot(int(len(images) / columns + 1), columns, i + 1)
      x=Image.open(image)
      plt.title(image.split("/")[4][0],fontsize=20,color="black")
      plt.imshow(x)
   plt.show()