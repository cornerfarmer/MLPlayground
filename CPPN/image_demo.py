import numpy as np
import matplotlib.cm as cm
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

image_data = np.random.uniform(0, 1, size=(255, 255)).astype(np.float32)
plt.subplot(1, 1, 1)
y_dim = image_data.shape[0]
x_dim = image_data.shape[1]
plt.imshow(image_data.reshape(y_dim, x_dim), cmap='gray', interpolation='nearest')
plt.axis('off')
plt.show()