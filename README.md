# Extendible continual few-shot segmentation framework

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
- Adapt registry (as in FAIR detectron 2) and use decorator for better style
- Use logger instead for print for train summary between epochs
- Support for more components
    - Loss
        - Focal Loss
        - More Imbalanced Loss/or combiner
- Update README docs in each folder
- Add transforms foolproof sanity checker
    - consistency of normalization in train/test set
    - normalization should only happen at the end of transforms
    - crop_size/input_size consistency check
- Use registry + decorator to eliminate all dispatcher
    - config files will need to be modified accordingly
- Integrate reproducibility package such as Sacred
- Support for Detection
- Support for 3D CV task
- Implement some nice features from the MIT ADE20K codebase
    - Batch random resizing augmentation
    - encoder/decoder design
    - deep supervision
