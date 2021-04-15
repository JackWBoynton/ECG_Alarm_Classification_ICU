# Classification of True or False Arrhythmia ECG Alarms in the ICU

This repository contains supplementary materials from a research project for determining whether an ECG arrhythmia alarm in the ICM is a true alarm or a false alarm. This is done by classifying the ECG segment immediately following an alarm into either that of a true alarm or a false alarm, as accurately as possible and as early as possible. This research resulted in an advancement of the state of the art, mostly resulting from 2015 PhysioNet/CinC Challenge (https://www.physionet.org/content/challenge-2015/1.0.0/).

**Contents in this repository:**

## Paper:
1. CinC 2021 abstract [pdf](abstract.pdf)
2. CinC 2021 full paper [pdf] -- _forthcoming_

## Methods:
1. Deep learning model: ResNet + BiLSTM. Model architecture (figure below) and source code [python (link)](resnet_attention.py)
    ![ResNet + BiLSTM](arch.png)   
2. Prequential evaluation: Growing window version. Source code _forthcoming_
3. Data preparation: WFDB ECG Segment splitting and data preparation. Source code [python (link)](split.py)
4. Data sets: 2015 PhysioNet Challenge data sets [(download)](https://storage.googleapis.com/challenge-2015-1.0.0.physionet.org/reducing-false-arrhythmia-alarms-in-the-icu-the-physionet-computing-in-cardiology-challenge-2015-1.0.0.zip)

## Results:
The result datasets are those used to generate the figures in the paper. (The corresponding figure numbers are specified for each dataset.)
1. Classification time for varying interval: one row for each ECG segment; one column for each batch-interval (4 msec, 0.5 sec, 1 sec, 2 sec) [(download)] (see Figure 3).
2. Model's output probaility over time: one row for each ECG segment; one row for each sample within the sample interval (4 msec) [(download)] (see Figure 4).
3. Classification times (with the 4 msec interval) for all ECG segments, for the polairy approach [(download)] (see Figure 5 left).
4. Classification times (with the 4 msec interval) for all ECG segments, for the threshold approach [(download)] (see Figure 5 right).

