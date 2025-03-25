import tacoreader
import rasterio as rio
import matplotlib.pyplot as plt

dataset = tacoreader.load("taco_example/austria.taco")

# Read a sample row
sample_idx = 0
row = dataset.read(sample_idx)
row_id = dataset.iloc[sample_idx]["tortilla:id"]
row_lr = row.iloc[0]
row_hr = row.iloc[1]

# Retrieve the data
lr, hr = row.read(0), row.read(1)
with rio.open(lr) as src_lr, rio.open(hr) as src_hr:
    lr_data = src_lr.read([1, 2, 3]) # 
    hr_data = src_hr.read([1, 2, 3]) # 

# Display
fig, ax = plt.subplots(1, 2, figsize=(10, 5.5))
ax[0].imshow(lr_data.transpose(1, 2, 0) / 1000)
ax[0].set_title(f'LR_{row_id}')
ax[0].axis('off')
ax[1].imshow(hr_data.transpose(1, 2, 0) / 300) 
ax[1].set_title(f'HR_{row_id}')
ax[1].axis('off')
plt.tight_layout()
plt.show()