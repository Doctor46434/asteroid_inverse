import numpy as np
from PIL import Image

# Experiment specifications
imagename = './2024on/2024On_new_20.png'
target_path = './2024on_log/2024On_new_20.png'
# imagename = 'cameraman256.png'

# Load noise-free image
y = Image.open(imagename)
y = y.convert('L')
y = np.array(y)
# Possible noise types to be generated 'gw', 'g1', 'g2', 'g3', 'g4', 'g1w',

log_noisy_image = np.log1p(y)

log_noisy_image = log_noisy_image/np.max(log_noisy_image)*255
log_noisy_image = log_noisy_image.astype(np.uint8) 

image = Image.fromarray(log_noisy_image)

image.save(target_path, 'PNG')

