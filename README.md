# ContrastiveProsthetics

Prosthetic hands usually use classification of sEMG (muscle electric) signals for differentiating different hand gestures in amputees' arms and allow for its control. A machine learning model is used.

To train, validate, and test, we use the ninapro databases 2 and 3:
http://ninaweb.hevs.ch/#:~:text=Ninapro%20is%20a%20publicly%20available,machine%20learning%20based%20control%20systems.

![alt text](https://www.researchgate.net/profile/Henning-Mueller-3/publication/279962755/figure/fig1/AS:614174241599503@1523441958336/The-Ninapro-acquisition-protocol-22.png)

The literature, with this same dataset, shows very weak results in EMG signals https://www.nature.com/articles/srep36571: if class imbalance is accounted for, there is a ~20-30% accuracy. This seems almost impossible to improve due to the signal-noise ratio in these signals and the large variability between people and recording sessions. Thus, we instead train a model to allow the user to choose a smaller subset of grasps to classify depending on the context.

We do classification of 41 different grasp types at train-time using contrastive learning between class encoding (one-hot vector) and z-vector on instantaneous sEMG signal. At test-time, the user can choose which classes to classify, the predicted label is the argmin of the inner products between the input and class encodings.

![alt text](https://openaiassets.blob.core.windows.net/$web/clip/draft/20210104b/overview-a.svg)

Eventually, the encoding won't be one-hot but glove angle signals (to specify arbitrary hand gestures), to allow for zero-shot generalization.

For more information about the exigence behind the type of adaptiveness used:
https://fibonaccidude.github.io/omleda/ideas/2022/03/21/choosing-your-compromises-at-test-time.html
