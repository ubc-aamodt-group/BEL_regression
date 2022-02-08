import numpy as np
from tqdm import tqdm
from util.logger import setup_logger

from datetime import datetime

import torch.nn as nn

import torch
import numpy as np


def do_evaluation(
        cfg,
        model,
        dataloader,
        device,
        criterion,
        transform,
        name,
        verbose=False
):
    model.eval()
    logger = setup_logger("DRIVINGDATASET", cfg.OUTPUT.DIR,
                          name + '-eval-{0:%Y-%m-%d %H:%M:%S}_log'.format(datetime.now()))
    logger.info("Start evaluating")

    mae = nn.L1Loss()

    loss_records = []
    for iteration, (images, steering_commands, steering_levels) in tqdm(enumerate(dataloader)):
        images = images.to(device)

        predictions = model(images)

        # predictions = predictions.clone().detach().cpu().numpy()
        real_predictions = transform(predictions)
        mae_loss = mae(steering_commands, real_predictions.cpu())
        loss_records.append(mae_loss.item())

        # if verbose:
        #     logger.info("LOSS: \t{}".format(loss))
        #     logger.info("MAE_LOSS: \t{}".format(mae_loss))


    logger.info('LOSS EVALUATION: {}'.format(np.mean(loss_records)))

