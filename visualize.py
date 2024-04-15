import os
import torch
import matplotlib.pyplot as plt

# Util function for loading meshes
from pytorch3d.io import load_objs_as_meshes, load_obj

# Data structures and functions for rendering
from pytorch3d.structures import Meshes
from pytorch3d.vis.plotly_vis import AxisArgs, plot_batch_individually, plot_scene
from pytorch3d.vis.texture_vis import texturesuv_image_matplotlib
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras,
    PointLights,
    AmbientLights,
    DirectionalLights,
    Materials,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    SoftPhongShader,
    TexturesUV,
    TexturesVertex,
)

from PIL import Image
import numpy as np
from pathlib import Path
import click
import pathlib

# Set batch size - this is the number of different viewpoints from which we want to render the mesh.

# Load the OBJ file and texture image


def generate_views(base_path, batch_size: int = 10, device: str = "cuda", distance:float=2.7):

    base_path = Path(base_path)

    # Define the directory path
    directory_path = Path(base_path)

    # Get a list of all files with the .obj extension in the directory
    obj_files = [file for file in directory_path.iterdir() if file.suffix == '.obj']

    # Check if any .obj files were found
    if obj_files:
        # Get the first .obj file
        first_obj_file = obj_files[0]
        print("First .obj file found:", first_obj_file)
    else:
        print("No .obj files found in the directory")

    # Find the object file *.obj
    model_file = base_path / first_obj_file #"textured.obj"
    texture_file = base_path / "textured.png"

    mesh = load_objs_as_meshes([model_file], device=device)
    # texture = load_texture(texture_file)

    # Initialize a camera.
    # With world coordinates +Y up, +X left and +Z in, the front of the cow is facing the -Z direction.
    # So we move the camera by 180 in the azimuth direction so it is facing the front of the cow.
    R, T = look_at_view_transform(2.0, 10, 45 + 180)
    cameras = FoVPerspectiveCameras(device=device, R=R, T=T)

    # Define the settings for rasterization and shading. Here we set the output image to be of size
    # 512x512. As we are rendering images for visualization purposes only we will set faces_per_pixel=1
    # and blur_radius=0.0. We also set bin_size and max_faces_per_bin to None which ensure that
    # the faster coarse-to-fine rasterization method is used. Refer to rasterize_meshes.py for
    # explanations of these parameters. Refer to docs/notes/renderer.md for an explanation of
    # the difference between naive and coarse-to-fine rasterization.
    raster_settings = RasterizationSettings(
        image_size=2048,
        blur_radius=0.0,
        faces_per_pixel=1,
    )

    light_distance = 10
    locations = torch.tensor(
        [
            [
                [0.0, 0.0, -light_distance],
            ]
        ],
        device=device,
    )
    print("contiguous", locations.is_contiguous())

    # Place a point light in front of the object. As mentioned above, the front of the cow is facing the
    # -z direction.
    lights = PointLights(device=device, location=locations)
    lights = AmbientLights(device=device, ambient_color=[0.5, 0.5, 0.5])

    # Create a Phong renderer by composing a rasterizer and a shader. The textured Phong shader will
    # interpolate the texture uv coordinates for each vertex, sample from a texture image and
    # apply the Phong lighting model
    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings),
        shader=SoftPhongShader(device=device, cameras=cameras, lights=lights),
    )

    # Create a batch of meshes by repeating the cow mesh and associated textures.
    # Meshes has a useful `extend` method which allows us do this very easily.
    # This also extends the textures.
    meshes = mesh.extend(batch_size)

    # Get a batch of viewing angles.
    elev = torch.linspace(0, 180, batch_size, device=device)
    azim = torch.linspace(-180, 180, batch_size, device=device)

    # All the cameras helper methods support mixed type inputs and broadcasting. So we can
    # view the camera from the same distance and specify dist=2.7 as a float,
    # and then specify elevation and azimuth angles for each viewpoint as tensors.
    R, T = look_at_view_transform(dist=distance, elev=elev, azim=azim)
    cameras = FoVPerspectiveCameras(device=device, R=R, T=T)

    # Move the light back in front of the cow which is facing the -z direction.
    # lights.location = torch.tensor([[0.0, 0.0, -3.0]], device=device)

    images = renderer(meshes, cameras=cameras, lights=lights)

    for index, image in enumerate(images):
        plt.figure(num=index, figsize=(10, 10))
        plt.imshow(images[index, ..., :3].cpu().numpy())
        plt.axis("off")

    plt.show()


@click.command()
@click.option("--base-path", help="Base path where obj file is located.")
@click.option("--batch-size", default=10, help="Number of images to draw")
@click.option("--distance", default=2.7, help="Distance of camera from origin")
def run(base_path: str, batch_size: int, distance:float):
    generate_views(
        base_path=base_path,
        batch_size=batch_size,
        distance=distance,
    )


if __name__ == "__main__":
    run()
