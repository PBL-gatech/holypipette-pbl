# PatcherBot-Agent


This code has its basis in [HolyPipette](https://github.com/romainbrette/holypipette). Many 
Thanks to the original contributors.

See the [documentation](docs/index.rst) for installation and usage.

## Conda setup and install order
1. `conda create -n patcherbot python=3.10.11`
2. `conda activate patcherbot`
3. `pip install -r requirements_runtime.txt` (core runtime deps)
4. Ensure LightGlue, SAM2, and robomimic are already cloned into `patcherbot/deepLearning/...` (matching the paths above).
5. `pip install -e patcherbot/deepLearning/cellModel/LightGlue`
6. `pip install -e patcherbot/deepLearning/cellModel/sam2`
7. `pip install -e patcherbot/deepLearning/patchModel/robomimic`
8. `pip install -e .`

