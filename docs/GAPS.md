# Prepare Dataset

Before preparing dataset, you first need to determine dataset root. You can set dataset root by setting a system-wide environment variable $DATASET_ROOT. If the environmental variable is not set, by default, it uses `/data`. For more details you can refer to https://github.com/RogerQi/dl_codebase/blob/roger/submission/modules/utils/misc.py#L10.

In the following instructions, we will assume that the data root is `/data`.

Main experiments of GAPS are done on two datasets: pascal-5<sup>i</sup> and coco-20<sup>i</sup>.

## Pascal-5<sup>i</sup>

Pascal segmentation datasets usually contain two sets of datasets - the original segmentation mask accompanying Pascal VOC 2012 semantic segmentation challenge, and a set of additional annotations supplemented by Berkeley SBD project.

Fortunately, torchvision has routines for conveniently downloading both of these two sets. The easiest way to download these two datasets are running examples at https://github.com/RogerQi/pascal-5i/blob/main/examples.ipynb

## COCO-20i<sup>i</sup>

TBD

## Pretrained Models

Like many other few-shot/incremental/general segmentation works, GAPS is trained from ImageNet pretrained weights. In particular, for fair comparisons with existing works, we follow their implementations and also use ResNet-101. The pretrained weights can be downloaded from https://download.pytorch.org/models/resnet101-5d3b4d8f.pth [Other ResNet weights can be found here](https://pytorch.org/vision/0.8/_modules/torchvision/models/resnet.html).

By default, the model loads weights from `/data/pretrained_model/resnet101-5d3b4d8f.pth` as defined [here](https://github.com/RogerQi/dl_codebase/blob/roger/submission/configs/fs_incremental/pascal5i_base.yaml#L16).

## Running GAPS

As described in our paper, learning in GAPS are divided into two stages: base learning stage and incremental learning stage. Take Pascal-5-3 as an example. To run the base learning stage, the command line to invoke is

```
cd dl_codebase # run from project root
python3 main/train.py --cfg configs/fs_incremental/pascal5i_split3_5shot.yaml
```

If you want to skip the prolonged base learning stage, you can find weights trained from the base stage at TBD.

After base learning stage, it will generate a weight named `GIFS_pascal_voc_split3_final.pt` at the project root. To perform incremental learning and testing, the command line to be invoked is

```
python3 main/test.py --cfg configs/fs_incremental/pascal5i_split3_5shot.yaml --load GIFS_pascal_voc_split3_final.py
```

and you should see the results.
