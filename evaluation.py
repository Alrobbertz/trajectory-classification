import pickle
import numpy as np
# Can be installed with 'pip install haversine'
from haversine import haversine
# Tensorflow
from tensorflow import convert_to_tensor
from tensorflow.keras.preprocessing import sequence

# Custom Imports
from data import *


"""
  Input:
      Traj: a list of list, contains one trajectory for one driver 
      example:[[114.10437, 22.573433, '2016-07-02 00:08:45', 1],
         [114.179665, 22.558701, '2016-07-02 00:08:52', 1]]
  Output:
      Data: any format that can be consumed by your model.
"""
def process_data_ensemble(traj):
    feat_p = []
    feat_np = []

    # To Generate a Heatmap
    heatmap = np.zeros((20, 20))
    frame_bot = 22.3111825
    frame_top = 22.975516499999998
    frame_left = 113.42487000000001
    frame_right = 114.62257799999999
    
    # Add Start Time and End Time to Generated Features
    weekday, start_time = extract_time(traj[0][2])
    _, end_time = extract_time(traj[-1][2])

    # Generated Features
    num_trips = 1
    total_service_time = 1
    total_seek_time = 1
    total_service_dist = 1
    total_seek_dist = 1

    '''
        Samples are input as
        ['longitude', 'latitude', 'time', 'status']
    '''
    # For each Sample in Trajectory
    last_sample=traj[0]
    last_timestamp=start_time
    for sample in traj:
        
        # Extract Time Features
        _, timestamp = extract_time(sample[2])

        # Extract Num Trips
        if sample[-1] == 1 and last_sample[-1] == 0:
            num_trips += 1

        # Extract Total Seek/ Service Time
        # Extract Total Seek/ Service Distance
        if sample[-1] == 1 and last_sample[-1] == 1:
            total_service_time += (timestamp - last_timestamp)
            total_service_dist += haversine((last_sample[0], last_sample[1]), (sample[0], sample[1]))
        elif sample[-1] == 0 and last_sample[-1] == 0:
            total_seek_time += (timestamp - last_timestamp)
            total_seek_dist += haversine((last_sample[0], last_sample[1]), (sample[0], sample[1]))

        # Add the Sample to the Heatmap
        if sample[1] > frame_bot and sample[1] < frame_top and sample[0] > frame_left and sample[0] < frame_right:
            x_reg = int(((frame_right - sample[0]) * 100) % 20)
            y_reg = int(((frame_top - sample[1]) * 100) % 20)
            heatmap[x_reg][y_reg] += 1

        # Split the Trjectory on whether there is a passenger or not
        if sample[-1] == 0: # No Passenger
            feat_np.append([sample[0], sample[1], timestamp])
        else:
            feat_p.append([sample[0], sample[1], timestamp])

        # Reset the Last Sample
        last_sample = sample
        last_timestamp = timestamp
    
    # Generarte Average Features
    avg_service_time = total_service_time / num_trips
    avg_seek_time = total_seek_time / num_trips
    avg_service_dist = total_service_dist / num_trips
    avg_seek_dist = total_seek_dist / num_trips
    avg_service_speed = avg_service_dist / avg_service_time
    avg_seek_speed = avg_seek_dist / avg_seek_time

    # Aggregate Features (10)
    feat_gen = [weekday, start_time, end_time, num_trips, avg_service_time, avg_seek_time, avg_service_dist, avg_seek_dist, avg_service_speed, avg_seek_speed]
    
    # Flatten Heatmap
    flatmap = heatmap.flatten()
    feat_gen.extend(flatmap)

    # # Run Compression on Trajectories
    # print(f'== Pre-Compression {len(feat_p)}, {len(feat_np)} samples')
    # '''
    #     max_dist_error in Km
    #     max_speed_error in Km/s
    # '''
    # feat_p = SPT_compression(feat_p, max_dist_error=0.025, max_speed_error=.0025)
    # feat_np = SPT_compression(feat_np, max_dist_error=0.025, max_speed_error=.0025)
    # print(f'== Compression to {len(feat_p)}, {len(feat_np)} samples')

    # Re-Shape Data
    max_traj_length = 2000
    [feat_p] = sequence.pad_sequences([feat_p], maxlen=max_traj_length, dtype='object')
    [feat_np] = sequence.pad_sequences([feat_np], maxlen=max_traj_length, dtype='object')

    feat_p = np.array(feat_p)
    feat_np = np.array(feat_np)
    feat_gen = np.array(feat_gen)

    # print(f'x_train_p: {feat_p.shape}')
    # print(f'x_train_np: {feat_np.shape}')
    # print(f'x_train_gen: {feat_gen.shape}')

    return [feat_p, feat_np, feat_gen]


"""
    Input:
        Data: the output of process_data function.
            in the format of [x_passengers, x_no_passengers, x_generated]
        Model: your model.
    Output:
        prediction: the predicted label(plate) of the data, an int value.
"""
def run_ensemble(data, model):
    [feat_p, feat_np, feat_gen] = data

    feat_p = convert_to_tensor(feat_p, dtype='float')
    feat_np = convert_to_tensor(feat_np, dtype='float')
    feat_gen = convert_to_tensor(feat_gen, dtype='float')

    result = model.predict(x=[[feat_p], [feat_np], [feat_gen]])
    return np.argmax(result)



"""
  Input:
      Traj: a list of list, contains one trajectory for one driver 
      example:[[114.10437, 22.573433, '2016-07-02 00:08:45', 1],
         [114.179665, 22.558701, '2016-07-02 00:08:52', 1]]
  Output:
      Data: any format that can be consumed by your model.
"""
def process_data_ff(traj):
    # To Generate a Heatmap
    heatmap = np.zeros((20, 20))
    frame_bot = 22.3111825
    frame_top = 22.975516499999998
    frame_left = 113.42487000000001
    frame_right = 114.62257799999999
    
    # Add Start Time and End Time to Generated Features
    weekday, start_time = extract_time(traj[0][2])
    _, end_time = extract_time(traj[-1][2])

    # Generated Features
    num_trips = 1
    total_service_time = 1
    total_seek_time = 1
    total_service_dist = 1
    total_seek_dist = 1

    '''
        Samples are input as
        ['longitude', 'latitude', 'time', 'status']
    '''
    # For each Sample in Trajectory
    last_sample=traj[0]
    last_timestamp=start_time
    for sample in traj:
        
        # Extract Time Features
        _, timestamp = extract_time(sample[2])

        # Extract Num Trips
        if sample[-1] == 1 and last_sample[-1] == 0:
            num_trips += 1

        # Extract Total Seek/ Service Time
        # Extract Total Seek/ Service Distance
        if sample[-1] == 1 and last_sample[-1] == 1:
            total_service_time += (timestamp - last_timestamp)
            total_service_dist += haversine((last_sample[0], last_sample[1]), (sample[0], sample[1]))
        elif sample[-1] == 0 and last_sample[-1] == 0:
            total_seek_time += (timestamp - last_timestamp)
            total_seek_dist += haversine((last_sample[0], last_sample[1]), (sample[0], sample[1]))

        # Add the Sample to the Heatmap
        if sample[1] > frame_bot and sample[1] < frame_top and sample[0] > frame_left and sample[0] < frame_right:
            x_reg = int(((frame_right - sample[0]) * 100) % 20)
            y_reg = int(((frame_top - sample[1]) * 100) % 20)
            heatmap[x_reg][y_reg] += 1

        # Reset the Last Sample
        last_sample = sample
        last_timestamp = timestamp
    
    # Generarte Average Features
    avg_service_time = total_service_time / num_trips
    avg_seek_time = total_seek_time / num_trips
    avg_service_dist = total_service_dist / num_trips
    avg_seek_dist = total_seek_dist / num_trips
    avg_service_speed = avg_service_dist / avg_service_time
    avg_seek_speed = avg_seek_dist / avg_seek_time

    # Aggregate Features (10)
    feat_gen = [weekday, start_time, end_time, num_trips, avg_service_time, avg_seek_time, avg_service_dist, avg_seek_dist, avg_service_speed, avg_seek_speed, total_service_time, total_service_dist, total_seek_time, total_seek_dist]
    
    # Flatten Heatmap
    flatmap = heatmap.flatten()
    feat_gen.extend(flatmap)

    return feat_gen
