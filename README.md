# Contrastive Prosthetics

Prosthetic hands usually use classification of sEMG (muscle electric) signals for differentiating different hand gestures in amputees' arms and allow for its control. A machine learning model is used.

To train, validate, and test, we use the ninapro databases 2 and 3.

<img src="https://www.researchgate.net/profile/Henning-Mueller-3/publication/279962755/figure/fig1/AS:614174241599503@1523441958336/The-Ninapro-acquisition-protocol-22.png" width="418" height="212">

The literature, with this same dataset, shows very weak results in EMG signals https://www.nature.com/articles/srep36571: if class imbalance is accounted for, there is a ~20-30% accuracy. This seems almost impossible to improve due to the signal-noise ratio in these signals and the large variability between people and recording sessions. Thus, we instead train a model to allow the user to choose a smaller subset of grasps to classify depending on the context.

We do classification of 41 different grasp types at train-time using contrastive learning between class encoding (one-hot vector) and z-vector on instantaneous sEMG signal. At test-time, the user can choose which classes to classify, the predicted label is the argmin of the inner products between the input and class encodings. This is an example of this procedure in OpenAI's CLIP:

<img src="https://openaiassets.blob.core.windows.net/$web/clip/draft/20210104b/overview-a.svg" width="285" height="402">

The average accuracy per prediction set size on 144 trials (all the possibilities would be too computationally expensive):

<img src="results.png" width="444" height="360">

In further research, the encoding won't be one-hot but glove angle signals (to specify arbitrary hand gestures), to allow for zero-shot generalization procedures. This is done in the name of lightweight adaptivity without backpropagation.

To run experiments:
```
./download_data.sh
./code/go.sh
./code/results.sh
```
or (with customization)

```
./download_data.sh
CUDA_LAUNCH_BLOCKING=1 python code/train.py --final_epochs=8 --crossval_size=150 --batch_size=8 --crossval_load  --test --no_adabn
./code/results.sh
```

For more information about the exigence behind the type of adaptiveness used:
visit https://fibonaccidude.github.io/personal-site/blog

-Science Fair Project 2021-2022
