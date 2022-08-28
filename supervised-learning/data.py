import pandas as pd
import numpy as np
from haversine import haversine
from datetime import datetime

# Dates
dates = {
    7: 31,
    8: 31,
    9: 30,
    10: 31,
    11: 30,
    12: 26
}

# Split the Data for Drivers
def supertrajectory():
    df=pd.DataFrame()
    # For each Month of Data
    for month, num_days in dates.items():
        # For each day in the month
        for day in range(1, num_days + 1):
            print(f'./data/2016_{month:02d}_{day:02d}')
            # Load the Data
            df = df.append(pd.read_csv(f'./data_5drivers/2016_{month:02d}_{day:02d}.csv'))

    df.to_csv('supertrajectory_.csv')
    return df


# Split the Data for Drivers
def split_raw_data():
    # For each Month of Data
    for month, num_days in dates.items():
        # For each day in the month
        for day in range(1, num_days + 1):
            # Load the Data
            df = pd.read_csv(f'./data_5drivers/2016_{month:02d}_{day:02d}.csv')
            # Separate the Plate Numbers
            for target_plate in range(5):
                filtered = df[df.plate == target_plate]
                filtered[['plate', 'longitude', 'latitude', 'time', 'status']].to_csv(
                    f'./data/2016_{month:02d}_{day:02d}_plate_{target_plate}.csv')


def load_raw_data(num_drivers):
    x_train = []
    y_train = []
    for month, num_days in dates.items():
        for day in range(1, num_days + 1):
            for target_plate in range(num_drivers):
                print(f'./data/2016_{month:02d}_{day:02d}_plate_{target_plate}.csv')
                df = pd.read_csv(f'./data/2016_{month:02d}_{day:02d}_plate_{target_plate}.csv')

                # Make Sure the Time values are sorted
                df = df.sort_values(by=['time'])

                x = np.array(df[['longitude', 'latitude', 'time', 'status']].values)

                x_train.append(x)
                y_train.append(target_plate)

    x_train = np.array(x_train)
    y_train = np.array(y_train)
    return x_train, y_train


def extract_time(time):
    # Extract Time Features
    dt = datetime.strptime(time, '%Y-%m-%d %H:%M:%S')
    weekday = dt.weekday()
    timestamp = dt.hour*3600+dt.minute*60+dt.second
    return weekday, timestamp


def mv_obj_aprox(Ps, Pe, Pi):
    print
    de = Pe[-1] - Ps[-1] if not Pe[-1] == Ps[-1] else 1
    di = Pi[-1] - Ps[-1] if not Pe[-1] == Ps[-1] else 1
    xp = Ps[0] + (di/de) * (Pe[0] -  Ps[0])
    yp = Ps[1] + (di/de) * (Pe[1] -  Ps[1])
    return (xp, yp)


# Format: [long | Lat | status | weeday | time] 
def SPT_compression(s, max_dist_error, max_speed_error):
    # print(f'Compressing S len={len(s)}')
    if(len(s) <= 2):
        return s
    else:
        is_error = False
        e = 1
    while e < len(s) and not is_error:
        i = 1
        while i < e and not is_error:
            (xp, yp) = mv_obj_aprox(s[0], s[e], s[i])
            vi_1 = haversine((s[i][0], s[i][1]), (s[i-1][0], s[i-1][1])) / (  s[i][-1] - s[i-1][-1] if s[i][-1] - s[i-1][-1] > 0 else 1)
            vi   = haversine((s[i+1][0], s[i+1][1]), (s[i][0], s[i][1])) / (s[i+1][-1] - s[i][-1]   if s[i+1][-1] - s[i][-1] > 0 else 1)

            if haversine((s[i][0], s[i][1]), (xp, yp)) > max_dist_error or np.absolute(vi - vi_1) > max_speed_error:
                # print(f'Max Dist Error: {haversine((s[i][0], s[i][1]), (xp, yp)) > max_dist_error}')
                # print(f'Max Speed Error: {(vi - vi_1) > max_speed_error}')
                is_error = True
            else:
                i += 1
        if is_error:
            return [s[0]] + SPT_compression(s[i:], max_dist_error, max_speed_error)
        e += 1
    if not is_error:
        return [s[0], s[-1]]


def avg_time_sync_error(p, a):
    n = np.sum([p[i+1][-1] - p[i][-1]*single_segment_time_sync_error(p[i:i+2], a) for i in range(len(p)-1)])
    d = np.sum([p[i+1][-1] - p[i][-1] for i in range(len(p)-1)])
    return n/d


'''
    p = [[]] of shape (2, 5)
    a = [[]] of shape (n, 5)
'''
def single_segment_time_sync_error(p, a):
    
    # Find Start and End Point of A
    i = 0
    while i < len(a)-2 and a[i][-1] < p[0][-1]:
        i+=1
    aj = a[i]
    aj_1 = a[i+1]

    (x1, y1) = mv_obj_aprox(aj, aj_1, p[0])
    (x2, y2) = mv_obj_aprox(aj, aj_1, p[1])

    dx1=p[0][0] - x1
    dy1=p[0][1] - y1
    dx2=p[1][0] - x2
    dy2=p[1][1] - y2

    c1 = (dx1-dx2)**2 + (dy1-dy2)**2
    if c1 == 0:
        return np.sqrt(dx1**2 + dy1**2)
    c2 = 2*((dx2*p[0][-1] - dx1*p[1][-1])*(dx1 - dx2) + (dy2*p[0][-1] - dy1*p[1][-1])*(dy1 - dy2))
    c3 = (dx2*p[0][-1] - dx1*p[1][-1])**2 + (dy2*p[0][-1] - dx1*p[1][-1])**2
    if c1 > 0:
        if c2**2 - 4*c1*c3 == 0:
            return (1/((p[1][-1] - p[0][-1])**2 if not p[0][-1] == p[1][-1] else 1)) * ((((2*c1*p[1][-1])/(4*c1)) * (np.sqrt(c1*p[1][-1]**2+c2*p[1][-1]+c3))) - (((2*c1*p[0][-1])/(4*c1)) *  (np.sqrt(c1*p[0][-1]**2+c2*p[0][-1]+c3))))
        else:
            return (1/((p[1][-1] - p[0][-1])**2 if not p[0][-1] == p[1][-1] else 1)) * (F(p[1][-1], c1, c2, c3) - F(p[0][-1], c1, c2, c3))
    
   
def F(t, c1, c2, c3):
    t1= (2*c1*t + c2)/(4*c1)
    t2 = np.sqrt(c1 * t**2 + c2 * t + c3)
    t3 = (c2**2-4*c1*c3)/(8*c1*np.sqrt(c1))
    # print(f'Difference: {4*c1*c3-c2**2}')
    # print(f'Terms: 4*c1*c3 = {4*c1*c3} c2**2 = {c2**2}')
    t4 = np.arcsinh((2*c1*t + c2)/(np.sqrt(4*c1*c3-c2**2)))
    return (t1 * t2) - (t3 * t4)



