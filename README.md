# Modified version using Poetry
Since torch==2.0.1 has an error that prevents proper poetry installation of cuda libraries, after
running
```
poetry install
```
run
```
poetry shell
```
to bring up the correct env, and then run
```
pip install torch==2.0.1
```
and that should do it. Later maybe I'll try getting this to work with a later version of torch
# SyncMVD

Official [Pytorch](https://pytorch.org/) & [Diffusers](https://github.com/huggingface/diffusers) implementation of the paper:



**[Text-Guided Texturing by Synchronized Multi-View Diffusion](https://arxiv.org/pdf/2311.12891)**

<!-- Authors: Yuxin Liu, Minshan Xie, Hanyuan Liu, Tien-Tsin Wong -->

<img src=assets/teaser.jpg width=768>

**SyncMVD** can generate texture for a 3D object from a text prompt using a **Sync**hronized **M**ulti-**V**iew **D**iffusion approach.
The method shares the denoised content among different views in each denoising step to ensure texture consistency and avoid seams and fragmentation (fig a).



<table style="table-layout: fixed; width: 100%;">
        <col style="width: 25%;">
        <col style="width: 25%;">
        <col style="width: 25%;">
        <col style="width: 25%;">
  <tr>
  <td>
    <img src=assets/gif/bird.gif width="170">
  </td>
  <td>
    <img src=assets/gif/david.gif width="170">
  </td>
  <td>
    <img src=assets/gif/dog.gif width="170">
  </td>
  <td>
    <img src=assets/gif/doll.gif width="170">
  </td>
  </tr>
  <tr style="vertical-align: text-top;">
    <td style="font-family:courier">"Photo of a beautiful magpie."</td>
    <td style="font-family:courier">"Publicity photo of a 60s movie, full color."</td>
    <td style="font-family:courier">"A cute shiba inu dog."</td>
    <td style="font-family:courier">"A cute Hatsune Miku plush doll, wearing beautiful dress."</td>
  </tr>
   <tr>
  <td>
    <img src=assets/gif/gloves.gif width="170" >
  </td>
  <td>
    <img src=assets/gif/hamburger.gif width="170" >
  </td>
  <td>
    <img src=assets/gif/house.gif width="170" >
  </td>
  <td>
    <img src=assets/gif/luckycat.gif width="170">
  </td>
  </tr>
  <tr style="vertical-align: text-top;">
    <td style="font-family:courier">"A photo of a robot hand with mechanical joints."</td>
    <td style="font-family:courier">"Photo of a hamburger."</td>
    <td style="font-family:courier">"Photo of a lowpoly fantasy house from warcraft game, lawn."</td>
    <td style="font-family:courier">"Blue and white pottery style lucky cat with intricate patterns."</td>
  </tr>

  <tr>
  <td>
    <img src=assets/gif/mask.gif width="170" >
  </td>
  <td>
    <img src=assets/gif/Moai.gif width="170" >
  </td>
  <td>
    <img src=assets/gif/sneakers.gif width="170">
  </td>
  <td>
    <img src=assets/gif/teddybear.gif width="170">
  </td>
  </tr>
  <tr style="vertical-align: text-top;">
    <td style="font-family:courier">"A Japanese demon mask."</td>
    <td style="font-family:courier">"Photo of James Harden."</td>
    <td style="font-family:courier">"A photo of a gray and black Nike Airforce high top sneakers."</td>
    <td style="font-family:courier">"Teddy bear wearing superman costume."</td>
  </tr>
</table>

## Installation :wrench:
The program is developed and tested on Linux system with Nvidia GPU. If you find compatibility issues on Windows platform, you can also consider using [WSL](https://learn.microsoft.com/en-us/windows/wsl/install).

To install, first clone the repository and install the basic dependencies
```bash
git clone https://github.com/LIU-Yuxin/SyncMVD.git
cd SyncMVD
conda create -n syncmvd python=3.8
conda activate syncmvd
pip install -r requirements.txt
```
Then install PyTorch3D through the following URL (change the respective Python, CUDA and PyTorch version in the link for the binary compatible with your setup), or install according to official [installation guide](https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md)
```bash
pip install https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py38_cu117_pyt200/download.html
```
The pretrained models will be downloaded automatically on demand, including:
- [runwayml/stable-diffusion-v1-5](https://huggingface.co/runwayml/stable-diffusion-v1-5)
- [lllyasviel/control_v11f1p_sd15_depth](lllyasviel/control_v11f1p_sd15_depth)
- [lllyasviel/control_v11p_sd15_normalbae](https://huggingface.co/lllyasviel/control_v11p_sd15_normalbae) 

## Data :floppy_disk:
The current program based on [PyTorch3D](https://github.com/facebookresearch/pytorch3d) library requires a input `.obj` mesh with `.mtl` material and related textures to read the original UV mapping of the object, which may require manual cleaning. Alternatively the program also support auto unwrapping based on [XAtlas](https://github.com/jpcy/xatlas) to load mesh that does not met the above requirements. The program also supports loading `.glb` mesh, but it may not be stable as its a PyTorch3D experiment feature.

To avoid unexpected artifact, the object being textured should avoid flipped face normals and overlapping UV, and keep the number of triangle faces within around 40,000. You can try [Blender](https://www.blender.org/) for manual mesh cleaning and processing, or its python scripting for automation.

You can try out the method with the following pre-processed meshes and configs:
- [Face - "Portrait photo of Kratos, god of war."](data/face/config.yaml) (by [2on](https://sketchfab.com/3d-models/face-ffde29cb64584cf1a939ac2b58d0a931))
- [Sneaker - "A photo of a camouflage military boot."](data/sneaker/config.yaml) (by [gianpego](https://sketchfab.com/3d-models/air-jordan-1-1985-2614cef9a3724ec5852144446fbb726f))

## Inference :rocket:
```bash
python run_experiment.py --config {your config}.yaml
```
Refer to [config.py](src/configs.py) for the list of arguments and settings you can adjust. You can change these settings by including them in a `.yaml` config file or passing the related arguments in command line; values specified in command line will overwrite those in config files.

When no output path is specified, the generated result will be placed in the same folder as the config file by default.

## License :scroll:
The program licensed under [MIT License](LICENSE).

## Citation :memo:
```bibtex
@article{liu2023text,
  title={Text-Guided Texturing by Synchronized Multi-View Diffusion},
  author={Liu, Yuxin and Xie, Minshan and Liu, Hanyuan and Wong, Tien-Tsin},
  journal={arXiv preprint arXiv:2311.12891},
  year={2023}
}
```
