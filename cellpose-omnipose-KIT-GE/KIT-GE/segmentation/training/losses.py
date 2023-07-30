import torch.nn as nn


def get_loss(loss_function):
    """ Get loss function(s) for the training process.

    :param loss_function: Loss function to use.
        :type loss_function: str
    :return: Loss function / dict of loss functions.
    """

    if loss_function == 'l1':
        border_criterion = nn.L1Loss()
        cell_criterion = nn.L1Loss()
    elif loss_function == 'l2':
        border_criterion = nn.MSELoss()
        cell_criterion = nn.MSELoss()
    elif loss_function == 'smooth_l1':
        border_criterion = nn.SmoothL1Loss()
        cell_criterion = nn.SmoothL1Loss()

    criterion = {'border': border_criterion, 'cell': cell_criterion}

    return criterion
