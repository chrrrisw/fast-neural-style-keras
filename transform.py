#!/usr/bin/env python

# Standard imports
import argparse
import os
import time

# Third party imports
from keras.optimizers import Adam
from scipy.misc import imsave
from scipy.ndimage.filters import median_filter
from skimage import color
import numpy as np

# Local imports
from img_util import preprocess_reflect_image, crop_image
from loss import dummy_loss
import nets

PRETRAINED_DIR = os.path.join(os.path.dirname(__file__), "pretrained")


def get_pretrained():
    print("The following pre-trained networks are available:")
    model_files = [mf for mf in os.listdir(PRETRAINED_DIR) if mf.endswith(".h5")]
    for i, mf in enumerate(model_files):
        print(f"{i:4d}: {mf}")
    result = input("Enter model number:")
    return model_files[int(result)]


# from 6o6o's fork. https://github.com/6o6o/chainer-fast-neuralstyle/blob/master/generate.py
def original_colors(original, stylized, original_color):
    # Histogram normalization in v channel
    ratio = 1.0 - original_color

    hsv = color.rgb2hsv(original / 255)
    hsv_s = color.rgb2hsv(stylized / 255)

    hsv_s[:, :, 2] = (ratio * hsv_s[:, :, 2]) + (1 - ratio) * hsv[:, :, 2]
    img = color.hsv2rgb(hsv_s)
    return img


def blend(original, stylized, alpha):
    """Simple alpha blend of original with the styled output."""
    return alpha * original + (1 - alpha) * stylized


def median_filter_all_colours(im_small, window_size):
    """
    Applies a median filer to all colour channels
    """
    ims = []
    for d in range(3):
        im_conv_d = median_filter(im_small[:, :, d], size=(window_size, window_size))
        ims.append(im_conv_d)

    im_conv = np.stack(ims, axis=2).astype("uint8")

    return im_conv


def main(args):
    if args.style is not None:
        style = args.style
    else:
        style = get_pretrained()

    # img_width = img_height =  args.image_size
    output_file = args.output
    input_file = args.input
    original_color = args.original_color
    blend_alpha = args.blend
    media_filter = args.media_filter

    aspect_ratio, x = preprocess_reflect_image(input_file, size_multiple=4)

    img_width = img_height = x.shape[1]
    net = nets.image_transform_net(img_width, img_height)
    model = nets.loss_net(net.output, net.input, img_width, img_height, "", 0, 0)

    # model.summary()

    model.compile(
        Adam(), dummy_loss
    )  # Dummy loss since we are learning from regularizes

    model.load_weights(os.path.join(PRETRAINED_DIR, style), by_name=False)

    t1 = time.time()
    y = net.predict(x)[0]
    y = crop_image(y, aspect_ratio)

    print("process: %s" % (time.time() - t1))

    ox = crop_image(x[0], aspect_ratio)

    y = median_filter_all_colours(y, media_filter)

    if blend_alpha > 0:
        y = blend(ox, y, blend_alpha)

    if original_color > 0:
        y = original_colors(ox, y, original_color)

    imsave("%s_output.png" % output_file, y)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Real-time style transfer",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--input",
        "-i",
        default=None,
        required=True,
        type=str,
        help="The content image filename.",
    )

    parser.add_argument(
        "--style",
        "-s",
        type=str,
        default=None,
        metavar="FILENAME",
        help="""Style image filename, assumes that the file exists in the
        ./pretrained directory. If no style is specified, you will be asked
        to choose from a list.""",
    )

    parser.add_argument(
        "--output",
        "-o",
        default=None,
        required=True,
        type=str,
        help="output file name without extension",
    )

    parser.add_argument(
        "--original_color", "-c", default=0, type=float, help="0~1 for original color"
    )

    parser.add_argument(
        "-b",
        "--blend",
        default=0,
        type=float,
        help="Blend with original image. This value is the alpha value for the original [0-1]",
    )

    parser.add_argument(
        "--media_filter", "-f", default=3, type=int, help="media_filter size"
    )
    parser.add_argument("--image_size", default=256, type=int)

    args = parser.parse_args()

    main(args)
