import numpy as np
import os
import torch
import collections
from dorsalnet import DorsalNet, DorsalNet_Alt, DorsalNet_DoG


def create_flashing_dots(x,y,r,noise=0.001):

    # x,y are coordinates of the flashing dots
    # r is the radius of the flashing dots
    # returns a 5D array of trials x RGB x frames x height x width

    ntau = 32 # number of frames in the stimulus
    lt = 16 # number of trials
    height = 112
    width = 112
    

    # create a frame with a black background
    empty = np.zeros((height,width))
    # create a frame with a white circle
    dot = empty.copy()
    for i in range(height):
        for j in range(width):
            if (i-x)**2+(j-y)**2<r**2:
                dot[i,j] = 1
    # make a video with ntau frames, first half is empty, second half is the dot
    stim = np.zeros((ntau,height,width))
    stim[ntau//2:,:,:] = dot
    # make the video RGB
    stim = np.repeat(stim[np.newaxis,:,:,:],3,axis=0)
    # make the video into trials
    stim = np.repeat(stim[np.newaxis,:,:,:,:],lt,axis=0)
    # add gaussian noise to the video
    stim += np.random.normal(0,noise,stim.shape)
    return stim



def create_drifting_gratings(ntau=32, ndirections=16, radius=32, lx=16, lt=16):
    # Create stimuli that contain all combos that are needed
    xi, yi = np.meshgrid(np.arange(-55.5, 56.5), np.arange(-55.5, 56.5))
    mask = xi**2 + yi**2 < radius**2
    oi = (np.arange(ndirections) / ndirections * 2 * np.pi).reshape((-1, 1, 1, 1))
    ti = np.arange(ntau)
    ti = ti - ti.mean()

    vals = []
    stims = []

    ri = (np.cos(oi) * xi.reshape((1, 1, xi.shape[0], xi.shape[1])) - np.sin(oi) * yi.reshape((1, 1, xi.shape[0], xi.shape[1])))
    X = mask.reshape((1, 1, xi.shape[0], xi.shape[1])) * np.cos((ri / lx) * 2 * np.pi - ti.reshape((1, -1, 1, 1)) / lt * 2 *np.pi)
    X = np.stack([X, X, X], axis=1) # Go from black and white to RGB
    return X

def get_feature_model(args):
    activations = collections.OrderedDict()

    def hook(name):
        def hook_fn(m, i, o):
            activations[name] = o

        return hook_fn


    if args.features == "airsim_04":
        ckpt_path = (
            "airsim_dorsalnet_batch2_model.ckpt-3174400-2021-02-12 02-03-29.666899.pt"
        )
        path = os.path.join(args.ckpt_root, ckpt_path)
        checkpoint = torch.load(path)

        subnet_dict = extract_subnet_dict(checkpoint)

        model = DorsalNet(False, 32)
        model.load_state_dict(subnet_dict)

        layers = collections.OrderedDict(
            [(f"layer{i:02}", l[-1]) for i, l in enumerate(model.layers)]
        )

        if args.subsample_layers:
            layers = collections.OrderedDict(
                [
                    (f"layer{i:02}", l[-1])
                    for i, l in enumerate(model.layers)
                    if i in (1, 2, 3, 4, 5)
                ]
            )

        metadata = {"sz": 112, "threed": True}
    elif args.features == "airsim_alt":
        ckpt_path = args.ckpt_path
        path = os.path.join(args.ckpt_root, ckpt_path)
        checkpoint = torch.load(path)

        subnet_dict = extract_subnet_dict(checkpoint)

        model = DorsalNet_Alt(False, 32)
        model.load_state_dict(subnet_dict)

        layers = collections.OrderedDict(
            [(f"layer{i:02}", l[-1]) for i, l in enumerate(model.layers)]
        )

        if args.subsample_layers:
            layers = collections.OrderedDict(
                [
                    (f"layer{i:02}", l[-1])
                    for i, l in enumerate(model.layers)
                    if i in (1, 2, 3, 4, 5, 6)
                ]
            )

        metadata = {"sz": 112, "threed": True}
    elif args.features == "airsim_dog":
        ckpt_path = args.ckpt_path
        path = os.path.join(args.ckpt_root, ckpt_path)
        checkpoint = torch.load(path)

        subnet_dict = extract_subnet_dict(checkpoint)

        model = DorsalNet_DoG(False, 32)
        model.load_state_dict(subnet_dict)

        layers = collections.OrderedDict(
            [(f"layer{i:02}", l[-1]) for i, l in enumerate(model.layers)]
        )

        if args.subsample_layers:
            layers = collections.OrderedDict(
                [
                    (f"layer{i:02}", l[-1])
                    for i, l in enumerate(model.layers)
                    if i in (1, 2, 3, 4, 5, 6)
                ]
            )

        metadata = {"sz": 112, "threed": True}
    else:
        raise NotImplementedError("Model not implemented yet")

    for key, layer in layers.items():
        layer.register_forward_hook(hook(key))

    metadata["layers"] = layers

    # Put model in eval mode (for batch_norm, dropout, etc.)
    model.eval()
    return model, activations, metadata



def extract_subnet_dict(d):
    out = {}
    for k, v in d.items():
        if k.startswith("fully_connected"):
            continue
        if k.startswith("subnet.") or k.startswith("module."):
            out[k[7:]] = v
        else:
            out[k] = v

    return out