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
NEW_PEOPLE=4
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

NEW_TASKS=4
TASKS=list(range(1,41))
np.random.shuffle(TASKS)
TEST_TASKS=TASKS[-NEW_TASKS:]
TRAIN_TASKS=TASKS[:-NEW_TASKS]
TASK_DIST=[17,23]
MAX_TASKS=sum(TASK_DIST)

MAX_TASKS_TRAIN=MAX_TASKS-NEW_TASKS

REPS=[1,3,4,6,2,5]
TRAIN_REPS=REPS[:4]
TEST_REPS=REPS[4:]
MAX_TRAIN_REPS=len(TRAIN_REPS)
MAX_TEST_REPS=len(TEST_REPS)
MAX_REPS=len(REPS)

BLOCK_SIZE=1

Hz=2000
DOWNSAMPLE=100 # how many frames per second
FACTOR=int(Hz/DOWNSAMPLE)

RMS_WINDOW=int(np.ceil(150 * Hz / 2048))
WINDOW_EDGE = (RMS_WINDOW-1)//2

Hz_glove=25     # much, much lower than that of sEMG
MS_GLOVE_SWITCH=(1/Hz_glove)*DOWNSAMPLE # in same sampling rate

TOTAL_WINDOW_SIZE=int(Hz*1.5)
FINAL_WINDOW_SIZE=TOTAL_WINDOW_SIZE//FACTOR

# instantaneous image (always, this won't change)
WINDOW_MS=1

WINDOW_OUTPUT_DIM=FINAL_WINDOW_SIZE # for backward compatability

assert FINAL_WINDOW_SIZE%WINDOW_OUTPUT_DIM==0
assert FINAL_WINDOW_SIZE%WINDOW_MS == 0, "Window ms does not fit into total window length"
AMT_WINDOWS=FINAL_WINDOW_SIZE//WINDOW_MS

EMG_DIM=12

PREFETCH=2
NUM_WORKERS=0
