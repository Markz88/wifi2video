######################################
# Evaluation metrics                 #
# ROC & AUPRC & F1-score             #
######################################


import torch.nn as nn
from pytorch_msssim import ssim

import numpy as np


# Evaluation selection
def evaluation(opt, real, sgnl_synth_fake, metric='MSE'):

    if opt['metric'] == 'MSE':
        return mse_video(opt,real,sgnl_synth_fake)
    elif opt['metric'] == 'SSIM':
        return ssim_video(opt,real,sgnl_synth_fake)
    else:
        raise NotImplementedError("Evaluation metric not implemented yet.")

def ssim_video(opt, real, sgnl_synth_fake):
    # Collect metric per video
    similarity_scores = []

    # For each video in the batch, compute SSIM for each frame
    for i in range(opt['videoGAN']['nframes']):
        batch_score = []
        if opt['videoGAN']['img_ch'] == 1:
            real_frame = real[:, :, i, :, :]
            sgnl_fake_frame = sgnl_synth_fake[:, :, i, :, :]

            # Compute the SSIM
            batch_score.append(ssim(real_frame, sgnl_fake_frame, data_range=-1., size_average=False))
        else:
            # Denormalizing [-1 1] -> [0 1] to compute SSIM for each frame
            real_frame = (real[:, :, i, :, :] + 1) / 2
            sgnl_fake_frame = (sgnl_synth_fake[:, :, i, :, :] + 1) / 2

            # Compute the SSIM
            batch_score.append(ssim(real_frame, sgnl_fake_frame, data_range=1., size_average=False))

        # Collect the score
        for score in batch_score:
            similarity_scores.append(np.asarray(score.cpu()))

    return similarity_scores
