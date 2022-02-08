import os
import torch
from model.engine.evaluation import do_evaluation
from datetime import datetime
from util.logger import setup_logger
from util.visdom_plots import VisdomLogger

import torch.nn as nn
import numpy as np
import time

from model import conversion_helper as conversion


mae = nn.L1Loss()

def do_train(
        cfg,
        args,
        model,
        dataloader_train,
        dataloader_evaluation,
        optimizer,
        device,
        criterion,
        transform,
        name
):
    # set mode to training for model (matters for Dropout, BatchNorm, etc.)
    model.train()

    # get the trainer logger and visdom
    visdom = VisdomLogger(cfg.LOG.PLOT.DISPLAY_PORT)
    visdom.register_keys(['loss'])
    logger = setup_logger('balad-mobile.train', False)
    logger.info("Start training")

    output_dir = os.path.join(cfg.LOG.PATH, name + '-train-run_{}'.format(datetime.now().strftime("%Y-%m-%d_%H:%M:%S")))
    os.makedirs(output_dir)

    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10,30], gamma=0.1)

    # start the training loop
    for epoch in range(cfg.SOLVER.EPOCHS):
        print("EPOCH: ", epoch)
        model.train()
        for iteration, (images, steering_commands, steering_levels) in enumerate(dataloader_train):
            images = images.to(device)

            predictions = model(images)
            steering_levels = torch.stack(steering_levels).cuda()

            if args.loss == "mae" or args.loss == "mse":
                outp = transform.convert_continuous(predictions)
                loss = criterion(outp, steering_commands.clone().detach().float().cuda())
            elif args.transform == "mc":
                steering_int_commands = torch.tensor(steering_commands, dtype=int).cuda()
                loss = criterion(predictions, steering_int_commands)
            elif args.loss == "ce":
                outp = transform.convert_discrete(predictions)
                steering_int_commands = torch.tensor(steering_commands, dtype=int).cuda()
                loss = criterion(outp, steering_int_commands)
            elif args.loss == "bce":
                steering_levels = torch.clamp(steering_levels, 0)
                loss = criterion(predictions, steering_levels)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
            # predictions = predictions.clone().detach().cpu().numpy()
            

            if iteration % cfg.LOG.PERIOD == 0:
                print(predictions)

                visdom.update({'loss': [loss.item()]})
                logger.info("LOSS: \t{}".format(loss))
                real_predictions = transform(predictions)
                print(real_predictions)
                mae_loss = mae(steering_commands, real_predictions.cpu())
                logger.info("MAE_LOSS: \t{}".format(mae_loss))

            # if iteration % cfg.LOG.PLOT.ITER_PERIOD == 0:
            #     visdom.do_plotting()

            step = epoch * len(dataloader_train) + iteration
            # if step % cfg.LOG.WEIGHTS_SAVE_PERIOD == 0 and iteration:
            #     torch.save(model.state_dict(),
            #                os.path.join(output_dir, 'weights_{}.pth'.format(str(step))))
            #     do_evaluation(cfg, model, dataloader_evaluation, device, criterion, converter)
        scheduler.step()

    torch.save(model.state_dict(), os.path.join(output_dir, 'weights_final.pth'))
