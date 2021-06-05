import generate as g
import os
import subprocess
import re
from typing import List, Optional

import click
import dnnlib
import numpy as np
from numpy import linalg
import PIL.Image
import torch

import legacy

from opensimplex import OpenSimplex
import generate as g


def generate_interp_images(easing, interpolation, network_pkl, process, random_seed,
    diameter, seeds, space, fps, frames, truncation_psi, noise_mode, outdir,
    movie_name, increment=0.01, start=.1, stop=1.5, projected_w=None, hold=False, hold_len=1 ):

    print('Loading networks from "%s"...' % network_pkl)
    device = torch.device('cuda')
    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device) # type: ignore

    # Synthesize the result of a W projection.
    if (process=='image') and projected_w is not None:
        if seeds is not None:
            print ('Warning: --seeds is ignored when using --projected-w')
        print(f'Generating images from projected W "{projected_w}"')
        ws = np.load(projected_w)['w']
        ws = torch.tensor(ws, device=device) # pylint: disable=not-callable
        assert ws.shape[1:] == (G.num_ws, G.w_dim)
        for idx, w in enumerate(ws):
            img = G.synthesis(w.unsqueeze(0), noise_mode=noise_mode)
            img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
            img = PIL.Image.fromarray(img[0].cpu().numpy(), 'RGB').save(f'{outdir}/proj{idx:03d}.png')
        return

    label=None

    if(process=='image'):
        if seeds is None:
            ctx.fail('--seeds option is required when not using --projected-w')

        # Generate images.
        for seed_idx, seed in enumerate(seeds):
            print('Generating image for seed %d (%d/%d) ...' % (seed, seed_idx, len(seeds)))
            z = torch.from_numpy(np.random.RandomState(seed).randn(1, G.z_dim)).to(device)
            img = G(z, label, truncation_psi=truncation_psi, noise_mode=noise_mode)
            img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
            PIL.Image.fromarray(img[0].cpu().numpy(), 'RGB').save(f'{outdir}/seed{seed:04d}.png')
            print('wrote file to %s' % f'{outdir}/seed{seed:04d}.png')

    elif(process=='interpolation' or process=='interpolation-truncation'):

        vidname = movie_name

        if process=='interpolation-truncation':
            g.interpolate(G,device,projected_w,seeds,random_seed,space,truncation_psi,label,frames,noise_mode,outdir,interpolation,easing,diameter,start,stop)
        else:
            g.interpolate(G,device,projected_w,seeds,random_seed,space,truncation_psi,label,frames,noise_mode,outdir,interpolation,easing,diameter, hold=hold, hold_length=hold_len)

        # convert to video
        #write moves in the root dir, not with frames
        # cmd=f'ffmpeg -y -r {fps} -i {outdir}/frame%04d.png -vcodec libx264 -pix_fmt yuv420p {vidname}'
        cmd=f'ffmpeg -y -r {fps} -i {outdir}/frame%04d.png -vcodec libx264 -pix_fmt yuv420p -vf "scale=w=720:h=720:force_original_aspect_ratio=decrease,pad=1080:1080:x=(ow-iw)/2:y=(oh-ih)/2:color=black" {vidname}'
        print(cmd)
        subprocess.call(cmd, shell=True)
        print(vidname)

    elif(process=='truncation'):
        if seeds is None or (len(seeds)>1):
            ctx.fail('truncation requires a single seed value')

        #vidname
        seed = seeds[0]
        # vidname = f'{process}-seed_{seed}-start_{start}-stop_{stop}-inc_{increment}-{fps}fps'
        vidname = movie_name

        # generate frames
        truncation_traversal(G,device,seeds,label,start,stop,increment,noise_mode,outdir)

        # convert to video
        # cmd=f'ffmpeg -y -r {fps} -i {outdir}/frame%04d.png -vcodec libx264 -pix_fmt yuv420p {vidname}'
        cmd=f'ffmpeg -y -r {fps} -i {outdir}/frame%04d.png -vcodec libx264 -pix_fmt yuv420p -vf "scale=w=720:h=720:force_original_aspect_ratio=decrease,pad=1080:1080:x=(ow-iw)/2:y=(oh-ih)/2:color=black" {vidname}'
        print(cmd)
        subprocess.call(cmd, shell=True)
        print(vidname)