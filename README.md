# Roger's Deep Learning Tools of the Trade

This codebase should be as generic as possible. Implementation of fancy ideas should be built
in fork of this repo rather than working in the repo itself to allow rapid development.

Dependencies
- PyTorch (torch, torchvision)
- yacs
- OpenCV
- PIL

TODOs
- Support for better Logging/Timing (tensorboard?)
- Pack the modules folder as a package to get rid of sys.path tricks
- Add engine (or core trainer) to abstract trainer for different tasks
- Adapt registry (as in FAIR detectron 2) and use decorator for better style
- Add better print statement for train summary between epochs
- Support for more components
    - Dataset
        - ADE20K
        - On COCO, add an option to disable penalizing crowded segmentation
    - Loss
        - Focal Loss
        - More Imbalanced Loss/or combiner
- Update README docs in each folder
- Add model saving logic
    - Computation Graph + Weights
    - Just weights
    - Feature Extraction
- Add transforms foolproof sanity checker
    - consistency of normalization in train/test set
    - normalization should only happen at the end of transforms
    - crop_size/input_size consistency check

Future TODOs
- Add closed-loop experiment logic
    - Use deterministic CUDA Ops from [PyTorch Reproducibility](https://pytorch.org/docs/stable/notes/randomness.html)
    - Fix seeds for all random Ops from [PyTorch Reproducibility](https://pytorch.org/docs/stable/notes/randomness.html)
    - Save entire EXP folders/config files to backup location
- Add pretrained weights loading logic and backbone freezing logic (support for fine-tuning)
- Support for Detection
- Support for 3D CV task
