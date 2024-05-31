from PIL import Image, ImageDraw
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import os
import shutil
import random
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import torch
from torchvision import transforms
from onedrivedownloader import download

class Dot:
  def __init__(self, x, y, radius):
    self.x = x
    self.y = y
    self.radius = radius

  def __repr__(self):
    return f"Dot at ({self.x}, {self.y}) with radius {self.radius}"

  def has_min_distance(self, other_dot, min_distance):
    distance = ((self.x - other_dot.x) ** 2 + (self.y - other_dot.y) ** 2) ** 0.5
    radius_sum = self.radius + other_dot.radius
    return distance >= radius_sum + min_distance

  def __eq__(self, other_dot):
    return self.x == other_dot.x and self.y == other_dot.y and self.radius == other_dot.radius

  def draw(self, draw):
    draw.ellipse((self.x - self.radius, self.y - self.radius, self.x + self.radius, self.y + self.radius), fill=(255, 0, 0))

class ImageDataset(Dataset):
  def __init__(self, root_dir, transform=None, size=None):
    self.root_dir = root_dir
    self.transform = transform
    self.files = os.listdir(self.root_dir)
    if size:
      if len(self.files) < size:
        raise Exception(f'Only found {len(self.files)} files in root directory, but the requested dataset size is {size}')
      self.files = random.sample(self.files, size)

  def __getitem__(self, idx):
    filename = self.files[idx]
    img_path = os.path.join(self.root_dir, filename)
    img = Image.open(img_path)
    img = self.transform(img) if self.transform is not None else img

    dot_count = torch.tensor([int(filename.split('_')[0])])

    return img, dot_count

  def __len__(self):
    return len(self.files)

def test_dot():
  d1 = Dot(10, 10, 5)
  d2 = Dot(20, 20, 8)
  d3 = Dot(30, 30, 6)

  img = Image.new('RGB', (50, 50), 'white')
  draw = ImageDraw.Draw(img)
  d1.draw(draw)
  d2.draw(draw)
  d3.draw(draw)
  plt.imshow(img)
  plt.show()

  assert d1.has_min_distance(d2, 1)
  assert not d2.has_min_distance(d3, 1)


def generate_image(image_size, dots_range, radius_range, min_distance, inserts_path=None):
  img = Image.new('RGB', image_size, 'white')

  if inserts_path:
    if os.path.exists(inserts_path):
      inserts_filenames = [filename for filename in os.listdir(inserts_path) if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif'))]
      n_inserts = random.choice([3, 4, 5])
      inserts_chosen = random.sample(inserts_filenames, n_inserts)
      for insert_path in inserts_chosen:
        insert = Image.open(os.path.join(inserts_path, insert_path))
        max_size = min(*image_size) // random.choice([2, 3, 4])
        insert.thumbnail((max_size, max_size))
        x_size, y_size = insert.size
        x = random.randint(0, image_size[0])
        y = random.randint(0, image_size[1])
        img.paste(insert, (x, y, x + x_size, y + y_size), mask=insert)
    else:
      print(f'WARNING: provided inserts path {inserts_path} does not exists')

  draw = ImageDraw.Draw(img)
  dots = []

  num_dots = random.randint(dots_range[0], dots_range[1])

  for _ in range(num_dots):
    x = random.randint(0, image_size[0])
    y = random.randint(0, image_size[1])
    radius = random.randint(radius_range[0], radius_range[1])
    dot = Dot(x, y, radius)

    # Check if the new dot would overlap with any existing dots
    too_close = False
    for other_dot in dots:
      if not dot.has_min_distance(other_dot, min_distance):
        too_close = True # Skip this dot if it is too close
        break

    if too_close:
      continue

    dot.draw(draw)
    dots.append(dot)

  return img, len(dots)

def generate_data(size, path, delete_existing=True, image_size=(32, 32), dots_range=(1, 50), radius_range=(2, 4), min_distance=5, inserts_path=None):
  path = Path(path)
  if os.path.exists(path) and delete_existing:
    shutil.rmtree(path)
  path.mkdir(parents=True, exist_ok=True)

  for i in tqdm(range(size)):
    img, n_dots = generate_image(image_size=image_size, dots_range=dots_range, radius_range=radius_range, min_distance=min_distance, inserts_path=inserts_path)
    img.save(f'{path}/{n_dots}_{i}.png')

def download_data_dots(root='data_dots'):
  url = 'https://goforeoy-my.sharepoint.com/:u:/g/personal/sebastian_landl_gofore_com/EbxvzSlpQsRNoUH740Ip67cBzpbE5UCAj1FtMovm9liJOg?e=DkhOht'
  filename = 'data_dots.zip'

  if os.path.exists(root):
    print(f'Folder {root} already exists. Not downloading.')
    return
  download(url, filename=filename, unzip=True, unzip_path=root, clean=True)

def prepare_data(path, size=1000, transform=None, batch_size=128, shuffle=True):
  if transform is None:
    transform = transforms.ToTensor()

  dataset = ImageDataset(root_dir=path, transform=transform, size=size)
  dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle)
  return dataloader, dataset

if __name__ == '__main__':
  download_data_dots()
