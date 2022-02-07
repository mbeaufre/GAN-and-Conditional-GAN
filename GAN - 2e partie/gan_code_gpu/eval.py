### Evaluate your cGAN
from discriminator import *
from generator import *
from datasets import *
from utils import *

load_model(epoch=200)

# switching mode
generator.eval()

# show a sample evaluation image on the training base
image, mask = next(iter(dataloader))
output = generator(mask.type(Tensor))
output = output.view(16, 3, 256, 256)
output = output.cpu().detach()
for i in range(8):
    image_plot = reverse_transform(image[i])
    output_plot = reverse_transform(output[i])
    mask_plot = reverse_transform(mask[i])
    plot2x3Array(mask_plot,image_plot,output_plot)

# show a sample evaluation image on the validation dataset
image, mask = next(iter(val_dataloader))
output = generator(mask.type(Tensor))
output = output.view(8, 3, 256, 256)
output = output.cpu().detach()
for i in range(8):
    image_plot = reverse_transform(image[i])
    output_plot = reverse_transform(output[i])
    mask_plot = reverse_transform(mask[i])
    plot2x3Array(mask_plot,image_plot,output_plot)

"""<font color='red'>**Question 4**</font>                                                                  
Compare results for 100 and 200 epochs
"""

# TO DO : Your code here to load and evaluate with a few samples
#         a model after 100 epochs

# And finally :
if cuda:
    torch.cuda.empty_cache()