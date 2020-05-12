# trajectory-classification

## Summary ##

This project includes two septerate tasks for trajectory classification. This first uses a standard reinforcement learning model/training setup to identify taxi plate numbers using a single-day's trajectory for a given driver. The second part of this project achieves the same task using a Siamese network for few-shot learning.

## Dataset Description

Each driver's day trajectory consists of a sequence of readings including:

| plate | longitute | latitude | time | status |
|---|---|---|---|---|
|4    |114.10437    |22.573433    |2016-07-02 0:08:45    |1|
|1    |114.179665    |22.558701    |2016-07-02 0:08:52    |1|
|0    |114.120682    |22.543751    |2016-07-02 0:08:51    |0|
|3    |113.93055    |22.545834    |2016-07-02 0:08:55    |0|
|4    |114.102051    |22.571966    |2016-07-02 0:09:01    |1|
|0    |114.12072    |22.543716    |2016-07-02 0:09:01    |0|

* **Plate**: Plate means the taxi's plate. In this project, we change them to keep anonymity. Same plate means same driver, so this is the target label for the classification. 
* **Longitude**: The longitude of the taxi.
* **Latitude**: The latitude of the taxi.
* **Time**: Timestamp of the record.
* **Status**: 1 means taxi is occupied and 0 means a vacant taxi.

### Reinforment Learning Setup

This portion of the project utilized a trajectory dataset with **six months of driver's daily trajectories for 5 drivers**. This was a substantial amount of data for each driver, enough to use standard learning models. 

#### Models 

This portion of the project tested two model setups:
* Fully-Connected, Feed-forward DNN 
* Ensemble Model, containing two LSTM branches and one feed-forwad branch. All branches are concattenated and fed into a feed-forward intepretation model. 


### Few-Shot Learning Setup

This portion of the project utilized a trajectory dataset with **five days of driver's trajectories for 500 drivers**. Becuase of the dataset composition, standard learning models could not be used. Therefore, we implemented a meta-learning / few-shot learning methodology to classify taxi plate numbers. 

#### Models 
* Fully-Connected, Feed-Forward Siamese network with weight sharing between inputs