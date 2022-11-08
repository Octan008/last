import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch

def plot_log_fig(fig, pred, gt, gs, i, props):
    plt.imshow(pred)
    plt.title("predicted")
    # plt.title(props["loss"])
    fig.add_subplot(gs[i, 1])
    plt.imshow(gt)
    plt.title("ground truth")
    # plt.title(props["loss"])
    ax = fig.add_subplot(gs[i, 2])
    ax.axis("off")
    x = 0.0
    keys = list(props.keys())
    for i, key in enumerate(keys):
        y = 0.8 - i*0.1
        plt.text(x, y, props[key])
    # y = 0.8
    # plt.text(x, y, props["title"])
    # y = 0.7
    # plt.text(x, y, props["loss"])
    # y = 0.6
    # plt.text(x, y, "loss_pose_local : " + str(props["loss_pose_local"]))
    # y = 0.5
    # plt.text(x, y, "loss_pose_global : " + str(props["loss_pose_global"]))
    # plt.text()


def log_fig_gen(pred, gt, props):#asnumpy
    display_count= 1
    fig = plt.figure(figsize=(8, 4 * display_count))
    gs = fig.add_gridspec(display_count, 3)

    for i in range(display_count):
		# hooks_dict = log_hooks[i]
        fig.add_subplot(gs[i, 0])
        plot_log_fig(fig, pred, gt, gs, i, props)
    plt.tight_layout()

    return fig

def tensor2imnp(input):
    return (input.detach().cpu().numpy() * 255).astype(np.uint8)



def save_log_im(pred, gt, save_path, props={}):
    fig = log_fig_gen(pred,gt,props)
    fig.savefig(save_path)
    plt.close()


    