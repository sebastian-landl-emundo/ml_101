from PIL import Image, ImageDraw
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import os
import shutil
import random
from pathlib import Path

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
