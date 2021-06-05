import numpy as np
import os, datetime

###########################################
## Generate Random seeds here

numberOfSeeds = 1000
seeds = np.random.randint(100000, size=numberOfSeeds)
# seeds = [24880,61744,28319,66648,19560,80782,64862] #57

# seeds = [13013,49780,81503,51921,38366] #453
# seeds = [23893, 11095,93462,64862,96118] #696
# seeds = [63817,23893,8727]
############################################
## Change options below here
loop = 0

if loop:
    # seeds[len(seeds)-1] = seeds[0]  # make the last seed same as the first, so it loops
    # seeds.append(seeds[0])  # make the last seed same as the first, so it loops
    seeds = np.append(seeds, seeds[0])
print("seeds: %s" % seeds)
# seeds = None  # not all options require seeds

process='interpolation'  # 'image', 'interpolation','truncation','interpolation-truncation'
process = "image"
interpolation='linear' # 'linear', 'slerp', 'noiseloop', 'circularloop'

frames = 40  # 'how many frames to produce (with seeds this is frames between each step, with loops this is total length)'

noise_mode='const'  # 'const', 'random', 'none'
diameter = 300  # this is for circularloop interposation
space = 'z'  # the latgent space.  Can be 'z' or 'w'.  Not all options work with 'w'
# seeds = np.random.randint(10000000, size=2)
random_seed = seeds[0]
fps = 24  # fps of the resulting movie file
truncation_psi = 1.0
easing = 'linear' # 'linear', 'easeInOutQuad', 'bounceEaseOut','circularEaseOut','circularEaseOut2'
hold = True
hold_len = 24
##################
# network_pkl = r"T:\projects\other\stylegan2_ada_shin\stylegan2-ada-pytorch_dvschultz\results\00033-demonHeads-mirror-11gb-gpu-resumeffhq512\demonHeads-snapshot-000160.pkl"
network_pkl = r"T:\projects\other\stylegan2_ada_shin\stylegan2-ada-pytorch_dvschultz\results\00026-catheads-mirror-11gb-gpu-resumeffhq512\network-snapshot-000400.pkl"
checkpoint_name = 'catHeads_400'  # this is for the folder name to go with this pkl
###################
outdir = 'out'  # this is where to write output too
###################
## output directories get created.  This will create directories if they don't exist
dt = datetime.datetime.today()
outdir += '/%0d_%0d' % (dt.month, dt.day)  # I have been using the month and day
todaydir = os.path.join(outdir, checkpoint_name)  # I have been noting the pkl that generated
rnum = np.random.randint(1000,size=1) # rnum for movie name, so all movies are unique - no overwrite
outdir = '%s/%s' % (todaydir,rnum[0])
os.makedirs(outdir, exist_ok=True)
movieoutdir = os.path.dirname(outdir)
movie_name = '%s/%s_%s_t%s_d%s_rs%s_%s.mp4' % (movieoutdir,interpolation, space, truncation_psi, diameter, random_seed, rnum[0])
print('movie_name: %s' % movie_name)
print('outdir: %s' % outdir)


########################################################################################
## ^^^^^^^ CHANGE THE VARIABLES ABOVE HERE ^^^^^^^^^^
########################################################################################

import shinGen as sg
sg.generate_interp_images(
    easing,
    interpolation,
    network_pkl,
    process,
    random_seed,
    diameter,
    seeds,
    space,
    fps,
    frames,
    truncation_psi,
    noise_mode,
    outdir,
    movie_name,
    increment = 0.01,
    start = .1,
    stop = 1.5,
    hold = hold,
    hold_len = hold_len
)
