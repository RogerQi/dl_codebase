# Roger's Deep Learning Tools of the Trade

Dependencies
- PyTorch (torch, torchvision)
- yacs
- OpenCV
- PIL

TODOs
- Implement naive CVAE and test on MNIST
- Support for 2D segmentation task
- Add more options for classifier/dataset/loss/network
    - Classifier
        - Dense
    - Dataset
        - Cifar10
        - ImageNet
    - Loss
    - Network
        - Resnet
- Update README docs in each folder
- Add model saving logic
    - Computation Graph + Weights
    - Just weights
    - Feature Extraction


Future TODOs
- Support for better Logging/Timing (tensorboard?)
- Add closed-loop experiment logic
    - Use deterministic CUDA Ops
    - Fix seeds for all random Ops
    - Save entire EXP folders/config files to backup location
- Add pretrained weights loading logic and backbone freezing logic (support for fine-tuning)
- Support for Detection
- Support for 3D CV task
