import torch
from pytorch3d.structures import Meshes
from pytorch3d.io import load_objs_as_meshes
from pytorch3d.renderer import (
    look_at_view_transform,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    SoftPhongShader,
    TexturesVertex
)
import matplotlib.pyplot as plt

# Assume 'model.obj' is your 3D model file.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = 'wx_origin.obj'
mesh = load_objs_as_meshes([model_path], device=device)

# Set the camera position.
R, T = look_at_view_transform(2.7, 0, 180)

# Define the settings for rasterization and shading. 
raster_settings = RasterizationSettings(
    image_size=512, 
    blur_radius=0.0, 
    faces_per_pixel=1, 
)

# Create a Phong renderer by composing a rasterizer and a shader. 
renderer = MeshRenderer(
    rasterizer=MeshRasterizer(
        cameras=None,
        raster_settings=raster_settings
    ),
    shader=SoftPhongShader(
        device=device,
        cameras=None,
    )
)



# Render the image.
rendered_image = renderer(mesh)

# Convert the rendered image to numpy for visualization
plt.figure(figsize=(10, 10))
plt.imshow(rendered_image[0, ..., :3].cpu().numpy())
plt.axis("off")
plt.show()