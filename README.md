# fast-neural-style-keras

This is a fast neural style transfer implement with Keras 2 (Tensorflow backend). For more detail please refer to  [Perceptual Losses for Real-Time Style Transfer and Super-Resolution](https://arxiv.org/abs/1603.08155)


## How to use

There are some pretrained models in pretrained/ - so you can use the command below to transform your image:

``` shell
python transform.py -i image/content/101.jpg -s la_muse -b 0.1 -o out
```

## How to train a new model

You'll first need an image dataset on which to train, perhaps MS COCO, which will need to be in a subdirectory of images/train. ``train.py`` uses ``ImageDataGenerator.flow_from_directory()`` with ``class_mode=None`` so the images can all be within the one subdirectory.

The image that you want to copy the style from needs to be in ``images/style`` and needs to have a ``jpg`` extension.

With all that in place:

``` shell
python train.py -s your_style_image_name_without_extension
```

## Some Examples
<img src="images/content/101.jpg" width="50%"><img src="images/content/tubingen.jpg" width="50%">
<img src="images/generated/des_glaneuses_101_output.png" width="50%"><img src="images/generated/des_glaneuses_tubingen_output.png" width="50%">
<img src="images/generated/starry_output.png" width="50%"><img src="images/generated/la_muse_tubingen_output.png" width="50%">
<img src="images/generated/wave_crop_output.png">

