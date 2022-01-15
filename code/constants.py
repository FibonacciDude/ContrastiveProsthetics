import numpy as np

np.random.seed(42)

PEOPLE_D2=list(range(40))
PEOPLE_D3=[2,3,4,5,8,9]
# add 1,3,10 for training + 2 more, rest for testing

MAX_PEOPLE_D2=len(PEOPLE_D2)
MAX_PEOPLE_D3=len(PEOPLE_D3)
PEOPLE_D3=[pl+MAX_PEOPLE_D2-1 for pl in PEOPLE_D3] # adjust to be from 40-...
ORIGINAL_D3=np.array(PEOPLE_D3).copy()
MAX_PEOPLE=MAX_PEOPLE_D2+MAX_PEOPLE_D3

# add so that it can stay constant without randomization of dataset
NEW_PEOPLE=4
np.random.shuffle(PEOPLE_D2)
np.random.shuffle(PEOPLE_D3)
PEOPLE=np.concatenate((PEOPLE_D2, PEOPLE_D3))
TEST_PEOPLE=PEOPLE[-NEW_PEOPLE:]
TRAIN_PEOPLE=PEOPLE[:-NEW_PEOPLE]

MAX_PEOPLE_TRAIN=MAX_PEOPLE-NEW_PEOPLE

NEW_TASKS=4
TASKS=list(range(40))
np.random.shuffle(TASKS)
TEST_TASKS=TASKS[-NEW_TASKS:]
TRAIN_TASKS=TASKS[:-NEW_TASKS]
TASK_DIST=[17,23]
MAX_TASKS=sum(TASK_DIST)

MAX_TASKS_TRAIN=MAX_TASKS-NEW_TASKS

REPS=[1,3,4,6,2,5]
#REPS=[3,4,5,6,2]
TRAIN_REPS=REPS[:4]
TEST_REPS=REPS[4:]
MAX_TRAIN_REPS=len(TRAIN_REPS)
MAX_TEST_REPS=len(TEST_REPS)
MAX_REPS=len(REPS)
BLOCK_SIZE=1    # 2 might be too large of a batch size
Hz=2000
# downsampling
DOWNSAMPLE=100 # how many frames per second
FACTOR=int(Hz/DOWNSAMPLE)
RMS_WINDOW=int(np.ceil(150 * Hz / 2048))
WINDOW_EDGE = (RMS_WINDOW-1)//2
Hz_glove=25     # much, much lower
MS_GLOVE_SWITCH=(1/Hz_glove)*1000   # in ms

TOTAL_WINDOW_SIZE=int(Hz*1.5)
#WINDOW_MS=20
#WINDOW_STRIDE=10
# instantaneous image
WINDOW_MS=1
WINDOW_STRIDE=1
# to  make it even for the mean in the network
WINDOW_OUTPUT_DIM=TOTAL_WINDOW_SIZE//WINDOW_STRIDE
WINDOW_BLOCK=25
#WINDOW_BLOCK=50
#WINDOW_BLOCK=4
#WINDOW_BLOCK=20
#WINDOW_BLOCK=40
assert WINDOW_OUTPUT_DIM%WINDOW_BLOCK==0
MAX_WINDOW_BLOCKS=WINDOW_OUTPUT_DIM//WINDOW_BLOCK-1 # non-complete stride

assert TOTAL_WINDOW_SIZE%WINDOW_OUTPUT_DIM==0
assert TOTAL_WINDOW_SIZE%WINDOW_MS == 0, "Window ms does not fit into total window length"
AMT_WINDOWS=TOTAL_WINDOW_SIZE//WINDOW_MS
WINDOW_SIZE=int(Hz*(TOTAL_WINDOW_SIZE/1000))
# TODO: see effect
GLOVE_DIM=22-1    # take out 11th sensor (going crazy)
EMG_DIM=12
ACC_DIM=EMG_DIM*3

PREFETCH=2
NUM_WORKERS=0
