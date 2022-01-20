import numpy as np

np.random.seed(42)

PEOPLE_D2=list(range(40))
PEOPLE_D3=[2,3,4,5,8,9]
# add 1,3,10 for training + 2 more, rest for testing

MAX_PEOPLE_D2=len(PEOPLE_D2)
MAX_PEOPLE_D3=len(PEOPLE_D3)
PEOPLE_D3=[pl+MAX_PEOPLE_D2-1 for pl in PEOPLE_D3] # adjust to be from 40-...
PEOPLE_D2=np.array(PEOPLE_D2)
PEOPLE_D3=np.array(PEOPLE_D3)

ORIGINAL_D3=PEOPLE_D3.copy()
MAX_PEOPLE=MAX_PEOPLE_D2+MAX_PEOPLE_D3

# add so that it can stay constant without randomization of dataset
NEW_PEOPLE=5
NEW_TASKS=4

d2_idxs=np.random.permutation(MAX_PEOPLE_D2)
d3_idxs=np.random.permutation(MAX_PEOPLE_D3)
PEOPLE_D2=PEOPLE_D2[d2_idxs]
PEOPLE_D3=PEOPLE_D3[d3_idxs]
PEOPLE=np.concatenate((PEOPLE_D2, PEOPLE_D3))
PEOPLE_IDXS=np.concatenate((d2_idxs, d3_idxs+len(d2_idxs)))
TEST_PEOPLE_IDXS=PEOPLE_IDXS[-NEW_PEOPLE:]
TRAIN_PEOPLE_IDXS=PEOPLE_IDXS[:-NEW_PEOPLE]
TEST_PEOPLE=PEOPLE[-NEW_PEOPLE:]
TRAIN_PEOPLE=PEOPLE[:-NEW_PEOPLE]

MAX_PEOPLE_TRAIN=MAX_PEOPLE-NEW_PEOPLE
MAX_PEOPLE_TEST=NEW_PEOPLE

TASKS=np.array(list(range(1,41)), dtype=np.uint8)
np.random.shuffle(TASKS)
TEST_TASKS=TASKS[-NEW_TASKS:]
TRAIN_TASKS=TASKS[:-NEW_TASKS]
TASK_DIST=np.array([17,23])
MAX_TASKS=TASK_DIST.sum()

MAX_TASKS_TRAIN=MAX_TASKS-NEW_TASKS

REPS=[1,3,4,6,2,5]
TRAIN_REPS=REPS[:4]
TEST_REPS=REPS[4:]
MAX_TRAIN_REPS=len(TRAIN_REPS)
MAX_TEST_REPS=len(TEST_REPS)
MAX_REPS=len(REPS)

PATH_DIR="/home/breezy/hci/prosthetics/db23/"

BLOCK_SIZE=1

Hz=2000
DOWNSAMPLE=100 # how many frames per second
FACTOR=int(Hz/DOWNSAMPLE)

#RMS_WINDOW=int(np.ceil(150 * Hz / 2048))
# If RMS_WINDOW is equivalent to the DOWNSAMPLE rate, there are no overlapping windows. This would be preferable for online data.
RMS_WINDOW=DOWNSAMPLE
WINDOW_EDGE=(RMS_WINDOW-1)//2

# in ms
PREDICTION_WINDOW=150
PREDICTION_WINDOW_SIZE=PREDICTION_WINDOW*DOWNSAMPLE/1000

TOTAL_WINDOW_SIZE=int(Hz*1)
FINAL_WINDOW_SIZE=TOTAL_WINDOW_SIZE//FACTOR

Hz_glove=25     # much, much lower than that of sEMG
# GLOVE is upsampled to this
GLOVE_FACTOR=int(1/Hz_glove*Hz)
# instantaneous image for emg
GLOVE_WINDOW_SIZE=TOTAL_WINDOW_SIZE//GLOVE_FACTOR

# instantaneous image (always, this won't change)
WINDOW_MS=1
WINDOW_OUTPUT_DIM=FINAL_WINDOW_SIZE # for backward compatability
WINDOW_BLOCK=25

assert WINDOW_OUTPUT_DIM%WINDOW_BLOCK==0
MAX_WINDOW_BLOCKS=( WINDOW_OUTPUT_DIM)//WINDOW_BLOCK

assert FINAL_WINDOW_SIZE%WINDOW_OUTPUT_DIM==0
assert FINAL_WINDOW_SIZE%WINDOW_MS == 0, "Window ms does not fit into total window length"
AMT_WINDOWS=FINAL_WINDOW_SIZE//WINDOW_MS

GLOVE_DIM=22-2    # take out 11th sensor (noisy) and the 6th sensor (nans)
EMG_DIM=12
ACC_DIM=EMG_DIM*3

PREFETCH=2
NUM_WORKERS=0
