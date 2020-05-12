import pickle
import time
import random
from datetime import datetime
import numpy as np
# Can be installed with 'pip install haversine'
from haversine import haversine


def create_pairs(x, digit_indices, num_classes=500):
    '''Positive and negative pair creation.
    Alternates between positive and negative pairs.
    '''
    # Pairs in the format [[input_a,  input_b]]
    pairs = []
    # Labels in the format []
    labels = []
    n = min([len(digit_indices[d]) for d in range(num_classes)]) - 1
    for d in range(num_classes):
        for i in range(n):
            z1, z2 = digit_indices[d][i], digit_indices[d][i + 1]
            pairs += [[x[z1], x[z2]]]
            inc = random.randrange(1, num_classes)
            dn = (d + inc) % num_classes
            z1, z2 = digit_indices[d][i], digit_indices[dn][i]
            pairs += [[x[z1], x[z2]]]
            labels += [1, 0]
    return np.array(pairs), np.array(labels)


def pair_superset(x, y):
    # Generate All Pairings of data points
    pairs = []
    labels = []

    for i in range(len(x)):
        print(f'Pairing {i}')
        for j in range(i, len(x)):
            pairs.append([x[i], x[j]])
            labels.append(1 if y[i] == y[j] else 0)

    pairs = np.array(pairs)
    labels = np.array(labels)

    return (pairs, labels)


def extract_time(time):
    # Extract Time Features
    dt = datetime.strptime(time, '%Y-%m-%d %H:%M:%S')
    weekday = dt.weekday()
    timestamp = dt.hour*3600+dt.minute*60+dt.second
    return weekday, timestamp


def load_raw_training():
    # Load the Raw Data from PKL Files
    training_data = pickle.load(open('.\data\project_4_train.pkl','rb'))

    x_raw = []
    y_raw = []
    # For all the Drivers
    for target_plate, days in training_data.items():
        print(f'Processing  [ {len(days)} ] Trajectories for Driver : {target_plate}')
        for day in days:
            # Each sample is in the form [plate, Longitude, Latitude, Seconds_Since_Midnight, Status, Time(STR)]
            # Get Rid of the Plate Number
            x = [sample[1:] for sample in day]

            x_raw.append(x)
            y_raw.append(target_plate)

        print(f'Sample Feature Vector  : {days[0][0]}')
        print(f'Sample Processed Fearute Vector : {x_raw[0][0]} ')

    x_raw = np.array(x_raw)
    y_raw = np.array(y_raw)

    #pickle.dump((x_raw, y_raw), open('.\data\_raw_data.pkl','wb'))
    print(f'Successfuly Loaded[{len(x_raw)}] Trajectories and {len(y_raw)} Labels ')
    return (x_raw, y_raw)

def load_validation_data():
    # Load Validation Data
    x_val = np.array(pickle.load(open('.\data\project_4_validation_data.pkl','rb')))
    y_val = np.array(pickle.load(open('.\data\project_4_validation_labels.pkl','rb')))

    print(f'Loaded and [ {len(x_val)} ] Trajectories and [ {len(y_val)} ] Labels ')
    print(f'Sample Feature Vector  : {x_val[0][0]}')
    print(f'Sample Label: {y_val[0]} ')
    # pickle.dump((x_val, y_val), open('.\data\_val_data.pkl','wb'))
    print(f'Data Shape {x_val.shape}')
    print(f'Label Shape {y_val.shape}')

    return (x_val, y_val)


"""
  Input:
      Traj: a list of list, contains one trajectory for one driver

    Longitude: The longitude of the taxi.
    Latitude: The latitude of the taxi.
    Second_since_midnight: How many seconds has past since midnight.
    Status: 1 means taxi is occupied and 0 means a vacant taxi.
    Time: Timestamp of the record.

  Output:
      Data: any format that can be consumed by your model.
"""
def process_trajectory(traj):

    # To Generate a Heatmap
    heatmap = np.zeros((20, 20))
    frame_bot = 22.3111825
    frame_top = 22.975516499999998
    frame_left = 113.42487000000001
    frame_right = 114.62257799999999
    outside = 0
    
    # Add Start Time and End Time to Generated Features
    weekday, start_time = extract_time(traj[0][-1])
    _, end_time = extract_time(traj[-1][-1])

    # Generated Features
    num_trips = 1
    total_service_time = 1
    total_seek_time = 1
    total_service_dist = 1
    total_seek_dist = 1

    '''
        Samples are input as
        ['longitude', 'latitude', 'sec_since_midnight', 'status', 'timestamp']
    '''
    # For each Sample in Trajectory
    last_sample=traj[0]
    last_timestamp=start_time
    for sample in traj:
        
        # Extract Time Features
        _, timestamp = extract_time(sample[-1])

        # Extract Num Trips
        if sample[-2] == 1 and last_sample[-2] == 0:
            num_trips += 1

        # Extract Total Seek/ Service Time & Distance
        if sample[-2] == 1 and last_sample[-2] == 1:
            total_service_time += (timestamp - last_timestamp)
            total_service_dist += haversine((last_sample[0], last_sample[1]), (sample[0], sample[1]))
        elif sample[-2] == 0 and last_sample[-2] == 0:
            total_seek_time += (timestamp - last_timestamp)
            total_seek_dist += haversine((last_sample[0], last_sample[1]), (sample[0], sample[1]))

        # Add the Sample to the Heatmap
        if sample[1] > frame_bot and sample[1] < frame_top and sample[0] > frame_left and sample[0] < frame_right:
            x_reg = int(((frame_right - sample[0]) * 100) % 20)
            y_reg = int(((frame_top - sample[1]) * 100) % 20)
            heatmap[x_reg][y_reg] += 1
        else:
            # print(f"Sample Fell Outside Heatmap!!!!!!!!")
            outside += 1

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

    ## AGGREGATRE / GENERATED FEATURES (410)
    # Gen Features (10)
    feat_gen = [weekday, start_time, end_time, num_trips, avg_service_time, avg_seek_time, avg_service_dist, avg_seek_dist, avg_service_speed, avg_seek_speed]
    # Flatten Heatmap (400)
    flatmap = heatmap.flatten()
    feat_gen.extend(flatmap)

    feat_gen = np.array(feat_gen)

    # print(f'x_train_p: {feat_p.shape}')
    # print(f'x_train_np: {feat_np.shape}')
    # print(f'x_train_gen: {feat_gen.shape}')

    return [feat_gen, outside]

def generate_training_features():
    #Load Raw Data
    x_raw, y_raw = pickle.load(open('.\data\_raw_data.pkl','rb'))

    x_train = []
    y_train = []
    total_outside = 0
    violators = []

    # Extract Features For each Trajectory
    for i in range(len(x_raw)):
        print(f'Processing Trajectory: {i}')
        
        # If The Trajectory Has at least one Sample
        if len(x_raw[i]) > 1:
            # Process Trajectory
            [x_feat, outside] = process_trajectory(x_raw[i])
            
            # Append to training
            x_train.append(x_feat)
            y_train.append(y_raw[i])
            total_outside += outside

            if outside > 0:
                violators.append(i)
        else:
            print('Too Short to Process')


    print('Done Processing')
    # Print the Shapes
    x_train = np.array(x_train)
    y_train = np.array(y_train)
    print(f'x_train: {x_train.shape}')
    print(f'Label Shape: {y_train.shape}') 
    print(f'Total Outside : {total_outside}')
    print(f'Num Violators : {len(violators)}')
    print(f'Violators : {violators}') 

    # Save the Data to an object file
    pickle.dump((x_train, y_train), open('.\data\_train_data_simple.pkl','wb'))


# # Time the Preprocessing 
# start_time = time.time()

# generate_training_features()

# # Stop the Preprocessing Timer
# print('PreProcessing took {} seconds'.format(time.time() - start_time))
