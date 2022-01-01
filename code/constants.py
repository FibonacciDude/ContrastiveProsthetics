PEOPLE_D2=list(range(40))
PEOPLE_D3=[2,3,4,5,8,9]
MAX_PEOPLE_D2=len(PEOPLE_D2)
MAX_PEOPLE_D3=len(PEOPLE_D3)
PEOPLE_D3=[pl+MAX_PEOPLE_D2-1 for pl in PEOPLE_D3] # adjust to be from 40-...
MAX_PEOPLE=MAX_PEOPLE_D2+MAX_PEOPLE_D3
REPS=[1,3,4,6,2,5]
TRAIN_REPS=REPS[:4]
TEST_REPS=REPS[4:]
MAX_TRAIN_REPS=len(TRAIN_REPS)
MAX_TEST_REPS=len(TEST_REPS)
MAX_REPS=len(REPS)
TASK_DIST=[17,23]
MAX_TASKS=sum(TASK_DIST)
# In ms.
TOTAL_WINDOW_SIZE=150
WINDOW_MS=15
# skip by 2 to get 1 frame per ms
Hz=2000
assert TOTAL_WINDOW_SIZE % WINDOW_MS == 0, "Window ms does not fit into total window length"
AMT_WINDOWS=TOTAL_WINDOW_SIZE//WINDOW_MS
WINDOW_SIZE=int(Hz*(TOTAL_WINDOW_SIZE/1000))
GLOVE_DIM=22 #-2 # no wrist sensors
EMG_DIM=12
ACC_DIM=EMG_DIM*3
