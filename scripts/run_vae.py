import cv2
import numpy as np
import rorlkit.torch.pytorch_util as ptu

from rorlkit.launchers.vae_experiments import prepare_vae_dataset
from rorlkit.util.io import load_local_or_remote_file
from rorlkit.torch.vae.conv_vae import imsize48_default_architecture


def reconstruct_img(vae, flat_img):
    latent_distribution_params = vae.encode(ptu.from_numpy(flat_img.reshape(1, -1)).cuda())
    reconstructions, _ = vae.decode(latent_distribution_params[0])
    imgs = ptu.get_numpy(reconstructions)
    imgs = imgs.reshape(
        1, vae.input_channels, vae.imsize, vae.imsize
    ).transpose(0, 3, 2, 1)  # BCWH -> BHWC
    img = cv2.cvtColor(imgs[0], cv2.COLOR_RGB2BGR)
    return img


def normalize_image(image, dtype=np.float64):
    assert image.dtype == np.uint8
    return dtype(image) / 255.0


if __name__ == "__main__":
    variant = dict(
        imsize=48,
        test_p=0.9,
        # exp1
        data_path='/home/dzp/samples_img.npy',
        flatted_data=False,
        vae_path='../models/vae/vae.pkl',
        # exp2
        # data_path='/home/dzp/Sawyer.npy',
        # flatted_data=True,
        # vae_path='../models/vae/sawyer_door.pkl'
    )

    train_data, test_data = prepare_vae_dataset(variant)
    vae_path = variant.get("vae_path")
    vae = load_local_or_remote_file(vae_path) if type(vae_path) is str else vae_path

    for i in range(len(train_data)):
        n_img = normalize_image(train_data[i])
        # n_img = ((n_img - data_mean) + 1.) / 2.  No need to do this

        r_img = reconstruct_img(vae, n_img)
        cv2.imshow('reconstructed train image', r_img)
        cv2.waitKey(50)

    for i in range(len(test_data)):
        n_img = normalize_image(test_data[i])
        # n_img = ((n_img - data_mean) + 1.) / 2.  No need to do this

        r_img = reconstruct_img(vae, n_img)
        cv2.imshow('reconstructed test image', r_img)
        cv2.waitKey(50)
