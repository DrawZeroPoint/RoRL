#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import cv2
import base64
import torch
import numpy as np
import json
import waitress

from flask import Flask, request, make_response

import rorlkit.torch.pytorch_util as ptu
from rorlkit.util.io import load_local_or_remote_file


app = Flask(__name__, template_folder="templates")

variant = dict(
    imsize=48,
    test_p=0.9,
    # exp1
    data_path='../dataset/panda_arm_rgby.npy',
    flatted_data=False,
    vae_path='../models/vae/panda_arm_rgby_vae.pkl',
    # exp2
    # data_path='/home/dzp/Sawyer.npy',
    # flatted_data=True,
    # vae_path='../models/vae/sawyer_door.pkl'
)

vae_path = variant.get("vae_path")
vae = load_local_or_remote_file(vae_path) if type(vae_path) is str else vae_path


def get_latent(raw_image):
    """Get latent variables (mean vector)"""
    image = cv2.resize(raw_image, (vae.imsize, vae.imsize))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = normalize_image(image)
    # swap order and reshape
    flat_img = torch.from_numpy(image).permute(2, 1, 0).flatten(start_dim=1).numpy()
    latent_distribution_params = vae.encode(ptu.from_numpy(flat_img.reshape(1, -1)).cuda())
    latents = ptu.get_numpy(latent_distribution_params[0])
    return latents


def reconstruct_img(flat_img):
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


def decode_b64_to_image(b64_str: str) -> [bool, np.ndarray]:
    """解码base64字符串为OpenCV图像, 适用于解码三通道彩色图像编码.
    :param b64_str: base64字符串
    :return: ok, cv2_image
    """
    if "," in b64_str:
        b64_str = b64_str.partition(",")[-1]
    else:
        b64_str = b64_str

    try:
        img = base64.b64decode(b64_str)
        return True, cv2.imdecode(np.frombuffer(img, dtype=np.int8), 1)
    except cv2.error:
        return False, None


@app.route('/get_image_latent', methods=['POST'])
def get_image_latent():
    try:
        req_data = json.loads(request.data)
        if 'body' not in req_data:
            return make_response("加载JSON失败", 200)
        else:
            req_body = req_data['body']
    except ValueError:
        return make_response("加载JSON失败", 200)

    header = {}
    response = {'results': []}

    src_img_str = req_body['image']
    ok, src_image = decode_b64_to_image(src_img_str)
    means = get_latent(src_image)
    torch.cuda.empty_cache()

    response['means'] = json.dumps(means.tolist())
    feedback = {'header': header, 'response': response}
    return make_response(json.dumps(feedback), 200)


if __name__ == "__main__":
    waitress.serve(app, host='0.0.0.0', port=6060, threads=6)
