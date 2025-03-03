# Notes
#### Load images from a folder
```python
batch = []
data_dir = './data_dir/'
filenames = [name for name in os.listdir(data_dir) if name.endswith('.jpg') or name.endswith('.png')]
for i, filename in enumerate(filenames):
    img_arr = imageio.imread(os.path.join(data_dir, filename))
    img_t = torch.from_numpy(img_arr)
    img_t = img_t.permute(2, 0, 1)  # HWC to CHW
    img_t = img_t[:3] # take only the first 3 channels: CHW
    batch.append(img_t)
```

#### Normalize images
```python
n_channels = batch.shape[1] # NCHW
for c in range(n_channels):
    mean = torch.mean(batch[:, c])
    std = torch.std(batch[:, c])
    batch[:, c] = (batch[:, c] - mean) / std
```
