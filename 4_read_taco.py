import numpy as np
import tacoreader
import rasterio as rio
import matplotlib.pyplot as plt

dataset = tacoreader.load("/data/USERS/shollend/taco/austria.0000.part.taco")

# Read a sample row
sample_idx = 1423
row = dataset.read(sample_idx)
print(dataset.iloc[sample_idx])
row_id = dataset.iloc[sample_idx]["tortilla:id"]
row_lr = row.iloc[0]
row_hr = row.iloc[1]
row_mask = row.iloc[2]

# Retrieve the data
lr, hr, mask = row.read(0), row.read(1), row.read(2)
with rio.open(lr) as src_lr, rio.open(hr) as src_hr, rio.open(mask) as src_mask:
    lr_data = src_lr.read([4, 3, 2]) #
    hr_data = src_hr.read([1, 2, 3]) #
    mask_data = src_mask.read()
    print(np.unique(mask_data))

# Display
fig, ax = plt.subplots(1, 3, figsize=(30, 5.5))
ax[0].imshow(lr_data.transpose(1, 2, 0) / 1000)
ax[0].set_title(f'LR_{row_id}')
ax[0].axis('off')

ax[1].imshow(hr_data.transpose(1, 2, 0) / 300) 
ax[1].set_title(f'HR_{row_id}')
ax[1].axis('off')

ax[2].imshow(mask_data.transpose(1, 2, 0))
ax[2].set_title(f'Mask_{row_id}')
ax[2].axis('off')

plt.tight_layout()
#plt.show()