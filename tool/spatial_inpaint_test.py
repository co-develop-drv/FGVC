from __future__ import absolute_import, division, print_function, unicode_literals
import os
import cv2
import numpy as np
import torch


def spatial_inpaint(deepfill, mask, video_comp):
    torch.cuda.empty_cache()
#     print(torch.cuda.memory_stats())
#     print(torch.cuda.memory_summary())
    print("==========================", torch.cuda.memory_allocated())

    # 第一个numpy数组中最大值的索引
    keyFrameInd = np.argmax(np.sum(np.sum(mask, axis=0), axis=0))
    print(keyFrameInd)
    print(video_comp[:, :, :, keyFrameInd].shape, mask[:, :, keyFrameInd].shape)
    with torch.no_grad():
        img_res = deepfill.forward(video_comp[:, :, :, keyFrameInd] * 255., mask[:, :, keyFrameInd]) / 255.
    video_comp[mask[:, :, keyFrameInd], :, keyFrameInd] = img_res[mask[:, :, keyFrameInd], :]
    mask[:, :, keyFrameInd] = False

    return mask, video_comp
