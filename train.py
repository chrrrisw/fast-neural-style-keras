from loss import dummy_loss

from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator

from scipy.misc import imsave
import time
import numpy as np
import argparse
import os

import nets


def save_img(i, x, style, is_val=False):
    # save current generated image
    img = x  # deprocess_image(x)
    if is_val:
        # img = ndimage.median_filter(img, 3)

        fname = "images/output/%s_%d_val.png" % (style, i)
    else:
        fname = "images/output/%s_%d.png" % (style, i)
    imsave(fname, img)
    print("Image saved as", fname)


def get_style_img_path(style):
    return "images/style/" + style + ".jpg"


def main(args):
    style_weight = args.style_weight
    content_weight = args.content_weight
    tv_weight = args.tv_weight
    style = args.style
    img_width = img_height = args.image_size

    style_image_path = get_style_img_path(style)

    net = nets.image_transform_net(img_width, img_height, tv_weight)
    model = nets.loss_net(
        net.output,
        net.input,
        img_width,
        img_height,
        style_image_path,
        content_weight,
        style_weight,
    )
    model.summary()

    # TODO: Cope with other file extensions
    num_images = 0
    for root, dirs, files in os.walk(args.train_image_path, followlinks=True):
        # print(root, dirs)
        num_images += len([f for f in files if f.endswith("jpg")])

    max_images = num_images * 2
    print("Number of images to be processed:", max_images)

    train_batchsize = 1

    # learning_rate = 1e-3  # 1e-3
    optimizer = Adam()  # Adam(lr=learning_rate,beta_1=0.99)

    model.compile(
        optimizer, dummy_loss
    )  # Dummy loss since we are learning from regularizes

    datagen = ImageDataGenerator()

    dummy_y = np.zeros(
        (train_batchsize, img_width, img_height, 3)
    )  # Dummy output, not used since we use regularizers to train

    # TODO: Start where we left off
    # model.load_weights(style + '_weights.h5', by_name=False)
    if args.resume[0] is not None:
        print(f"Resuming from {args.resume[1]}")
        model.load_weights(args.resume[0])
        skip_to = int(args.resume[1])
    else:
        skip_to = 0

    image_count = 0
    t1 = time.time()

    try:
        pass
        for x in datagen.flow_from_directory(
            args.train_image_path,
            class_mode=None,
            batch_size=train_batchsize,
            target_size=(img_width, img_height),
            shuffle=False,
        ):
            # We're done
            if image_count > max_images:
                break

            # Skip training until image_count >= skip_to
            if image_count < skip_to:
                image_count += train_batchsize
                if image_count % 1000 == 0:
                    print("skip to: %d" % image_count)

                continue

            hist = model.train_on_batch(x, dummy_y)

            if image_count % args.save_interval == 0:
                print(image_count, hist, (time.time() - t1))
                val_x = net.predict(x)
                save_img(image_count, x[0], style)
                save_img(image_count, val_x[0], style, True)
                model.save_weights(style + "_weights.h5")
                t1 = time.time()

            image_count += train_batchsize

    except KeyboardInterrupt:
        print("Stopped by user.")

    # Save the final model
    model.save_weights(f"{style}_weights_final_{image_count}.h5")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Real-time style transfer")

    parser.add_argument(
        "--style",
        "-s",
        type=str,
        required=True,
        help="style image file name without extension",
    )

    # parser.add_argument(
    #     "--output",
    #     "-o",
    #     default=None,
    #     type=str,
    #     help="output model file path without extension",
    # )

    parser.add_argument(
        "--tv_weight",
        default=1e-6,
        type=float,
        help="weight of total variation regularization according to the paper to be set between 10e-4 and 10e-6.",
    )
    parser.add_argument("--content_weight", default=1.0, type=float)
    parser.add_argument("--style_weight", default=4.0, type=float)
    parser.add_argument("--image_size", default=256, type=int)
    parser.add_argument("--train_image_path", default="images/train/", type=str)
    parser.add_argument("--save_interval", default=50, type=int)
    parser.add_argument("--resume", nargs=2, default=[None, 0], help="Model name and iteration number")

    args = parser.parse_args()
    main(args)
