from io import StringIO
import os
import pickle
from flask import Flask, request
import json
import numpy as np
import pandas as pd
import scipy.spatial.distance
import datetime
import tabulate
from sklearn import preprocessing
import csv
from collections import deque


# spatial threshold or the dispersion itself
max_dispersion = np.deg2rad(1.6)
# temporal threshold or duration
min_duration = 100

def vector_dispersion(vectors):
    distances = scipy.spatial.distance.pdist(vectors, metric='cosine')
    distances.sort()
    cut_off = np.max([distances.shape[0] // 5, 4])
    return np.arccos(1. - distances[-cut_off:].mean())

def gaze_dispersion(eye_data):
    base_data = eye_data

    vectors = []
    for p in base_data:
        vectors.append((p['gazeDirection_x'], p['gazeDirection_y'], p['gazeDirection_z']))
    vectors = np.array(vectors, dtype=np.float32)

    if len(vectors) < 2:
        return float("inf")
    else:
        return vector_dispersion(vectors)  
    

def get_centroid(eye_data):
    '''Calculates the centroid for each point in a df of points.
    Input: Df of points.
    Output: Vector containg the centroid of all points.'''
    x = [p['gazeDirection_x'] for p in eye_data]
    y = [p['gazeDirection_y'] for p in eye_data]
    z = [p['gazeDirection_z'] for p in eye_data]
    return (sum(x) / len(eye_data), sum(y) / len(eye_data), sum(z) / len(eye_data))

def calculate_blink_features(df, timespan): 
    ''' Calculates the blink features for a given df of raw data.
    Input: Dataframe with raw data (incl. invalid points), timespan of data chunk in seconds\
    Output: Dict with the blink features
    '''
    i=0;
    blink_list = []
    blink_duration_list = []
    number_of_blinks = 0
    window_start_time = df["eyeDataTimestamp"][0]
    window_end_time = df["eyeDataTimestamp"][0]
    all_false = 0;
    if (not window_start_time  or not window_end_time):
        return {}
        
    for i, row in df.iterrows():    
        
        cur_number_of_blinks = 0
        if (not row["gazeHasValue"]):
            cur_number_of_blinks+=1
            all_false += 1
            while (i < len(df["gazeHasValue"])) and (not df["gazeHasValue"][i]) and  window_end_time - window_start_time < 1000:
                window_end_time = row["eyeDataTimestamp"]
                i+=1
                all_false +=1
            blink_list.append(cur_number_of_blinks)
            duration = window_end_time - window_start_time
            blink_duration_list.append(duration)
        
        number_of_blinks += cur_number_of_blinks;
        if  window_end_time - window_start_time > 1000:
            window_start_time = window_end_time
    
    blinks_per_second = 0
    if (len(blink_list)  > 0):
        blinks_per_second = number_of_blinks / timespan 
    avg_blink_duration = 0
    min_blink_duration = 0
    max_blink_duration = 0
    if (len(blink_duration_list) > 0):
        avg_blink_duration = sum(blink_duration_list) / len(blink_duration_list)
        min_blink_duration = min(blink_duration_list)
        max_blink_duration = max(blink_duration_list)
        
    # print("all_blinks: ", number_of_blinks, " avg_blink_duration: ", avg_blink_duration, 
    #    " min_blink_duration: ", min_blink_duration, " max_blink_duration: ", max_blink_duration,
    #  " blinks_per_second: ", blinks_per_second, " all false: ", all_false)
    
    return {"number_of_blinks":number_of_blinks, "blinkMean": avg_blink_duration, 
            "blinkMin": min_blink_duration, "blinkMax": max_blink_duration, 
            "blinkRate": blinks_per_second}

def detect_fixations(gaze_data):
    # Convert Pandas data frame to list of Python dictionaries
    gaze_data = gaze_data.T.to_dict().values()

    candidate = deque()
    future_data = deque(gaze_data)
    while future_data:
        # check if candidate contains enough data
        if len(candidate) < 2 or candidate[-1]['eyeDataTimestamp'] - candidate[0]['eyeDataTimestamp'] < min_duration:
            datum = future_data.popleft()
            candidate.append(datum)
            continue

        # Minimal duration reached, check for fixation
        dispersion = gaze_dispersion(candidate)
        if dispersion > max_dispersion:
            # not a fixation, move forward
            candidate.popleft()
            continue

        # Minimal fixation found. Try to extend!
        while future_data:
            datum = future_data[0]
            candidate.append(datum)

            dispersion = gaze_dispersion(candidate)
            if dispersion > max_dispersion:
                # end of fixation found
                candidate.pop()
                break
            else:
                # still a fixation, continue extending
                future_data.popleft()
        centroid = get_centroid(candidate)
        yield {"start": candidate[0]['eyeDataTimestamp'], "end": candidate[-1]['eyeDataTimestamp'],
               "duration": candidate[-1]['eyeDataTimestamp'] - candidate[0]['eyeDataTimestamp'],
               "centroid": centroid, "dispersion": dispersion}
        candidate.clear()


def calculate_fixation_features(df_fixations, timespan):
    '''Calculates the fixation features. 
    Input: Dataframe with fixation, timespan of data chunk in seconds.
    Output: Dict containing the fixation features.'''

    min_fix = df_fixations["duration"].min()
    max_fix = df_fixations["duration"].max()
    mean_fix = df_fixations["duration"].mean()
    var_fix = df_fixations["duration"].var()
    std_fix = df_fixations["duration"].std()
    
    min_dispersion = df_fixations["dispersion"].min()
    max_dispersion = df_fixations["dispersion"].max()
    mean_dispersion = df_fixations["dispersion"].mean()
    var_dispersion = df_fixations["dispersion"].var()
    std_dispersion = df_fixations["dispersion"].std()
    
    fixation_frequency_second = (len(df_fixations["dispersion"]) / timespan)
    
    # print("min: ", min_fix, " max: ", max_fix, " mean: ", mean_fix, " var: ", var_fix, "std: ", std_fix)
    # print("min dispersion: ", min_dispersion, " max: ", max_dispersion, " mean: ", mean_dispersion, 
    #      " var: ", var_dispersion)
    #print("x dispersion: ", dispersion_x, " y dispersion: ", dispersion_y, " z dispersion: ", dispersion_z)
    
    return {"meanFix": mean_fix, "minFix": min_fix, "maxFix": max_fix, "varFix": var_fix, "stdFix": std_fix,
            "meanDis": mean_dispersion, "minDis": min_dispersion, "maxDis": max_dispersion,
            "varDis": var_dispersion, "stdDisp": std_dispersion,
            "freqDisPerSec": fixation_frequency_second}
    

def get_fixation_df(df_valid):
    '''Calls function to calculate Fixations. Converts the list of fixations to a dataframe and numbers the rows as index.
     Input: Dataframe containg valid gaze points.
     Output: Dataframe containing the fixation features.'''
    fixations = list(detect_fixations(df_valid))
    df = pd.DataFrame(fixations)
    df['index'] = range(1, len(df) + 1)
    # df.head()
    return df


def calculate_directions_of_list(points):
    '''Calculates the dominant direction of points.
    Input: Dataframe containing fixation points.
    Output: Dict with dominant direction for x (xDir) and y (yDir).
    '''
    x_values, y_values, z_values = zip(*points['centroid'])
    # Get a list of whether a given value is greater then the previous one in the list
    res_x = [float(val1) < float(val2) for val1, val2 in zip(x_values, x_values[1:])]
    # Sum all that are True
    sum_x = sum(res_x)
    # Divide the sum by the total number of values to get the desired output.
    # dir_x is -1 if there are no fixation (i.e. prevent division by zero)
    dir_x = -1
    if len(res_x) != 0:
        dir_x = sum_x/len(res_x)
        
    res_y = [float(val1) < float(val2) for val1, val2 in zip(y_values, y_values[1:])]
    sum_y = sum(res_y)
    dir_y = -1
    if len(res_y) != 0:
        dir_y = sum_y /len(res_y)
      
    
    return {"xDir": dir_x, "yDir": dir_y}


def calculate_fixation_density(df_all, df_fix):
    '''Calculates the fixation density per area.
    Input: Dataframe with all valid gazepoints, Dataframe with fixations.
    Output: Dict containing the fixation density.'''
    min_x = df_all['gazeDirection_x'].min()
    min_y = df_all['gazeDirection_y'].min()
    max_x = df_all['gazeDirection_x'].max()
    max_y = df_all['gazeDirection_y'].max()
    
    length = abs(max_x-min_x)
    height = abs(max_x-min_x)
    area = length*height
    
    number_of_fixations = len(df_fix)
    
    fix_dens = -1
    if area != 0: 
        fix_dens = number_of_fixations/area
    return {"fixDensPerBB": fix_dens}

def get_features_for_n_seconds(df, timespan, label, participant_id):
    ''' Calculates the features for a raw gaze data in chunks of n seconds.
    Input: Dataframe with raw gaze data, timespan to chunk, label (i.e. activity class).
    Output: List of dictionaries, one dictionary contains a chunk of features.
    '''
    
    list_of_features = []
    i = 0
    while i < len(df)-1: 
        newdf = pd.DataFrame(columns=df.columns)
        start_time = df["eyeDataTimestamp"][i]  
        
        while i < len(df)-1 and df["eyeDataTimestamp"][i] < (start_time+timespan*1000):
            entry = df.iloc[[i]]
            newdf = pd.concat([newdf,entry])
            i+=1
        newdf.reset_index(inplace = True)
        
        if (len(newdf) > timespan*28):
            newdf_valid = only_valid_data(newdf)
            df_fixations = get_fixation_df(newdf_valid)
            
            features = calculate_fixation_features(df_fixations, timespan)
            blinks = calculate_blink_features(newdf,timespan)

            directions = calculate_directions_of_list(df_fixations)            
            density = calculate_fixation_density(newdf_valid, df_fixations)
            
            features.update(blinks)
            features.update(directions)
            features.update(density)
            features["label"] = label
            # print(f"label: {label}")
            features["duration"] = timespan
            features["participant_id"] = participant_id            
            list_of_features.append(features)   
        
    return list_of_features

def save_as_csv(list_of_dict, participant, folder):
    '''Saves a list of dicts as one csv file.
    Input: List of feature dicts, participant id.
    Output: CSV file, saved in same directory as this file.'''
    header = list_of_dict[0].keys()
    rows =  [x.values() for x in list_of_dict]

    keys = list_of_dict[0].keys()
    file_name = os.path.join(folder, f'feature_list_P{participant}.csv')
    # if the file already exists, append the rows
    if os.path.exists(file_name):
        with open(file_name, 'a', newline='') as output_file:
            dict_writer = csv.DictWriter(output_file, keys)
            dict_writer.writerows(list_of_dict)
    # otherwise create a new file
    else:
        with open(file_name, 'w', newline='') as output_file:
            dict_writer = csv.DictWriter(output_file, keys)
            dict_writer.writeheader()
            dict_writer.writerows(list_of_dict)
    # '''        
    file_name_all = os.path.join(folder, f'feature_list_all.csv')
    # if the file already exists, append the rows
    if os.path.exists(file_name_all):
        with open(file_name_all, 'a', newline='') as output_file:
            dict_writer = csv.DictWriter(output_file, keys)
            dict_writer.writerows(list_of_dict)
    # otherwise create a new file
    else:
        with open(file_name_all, 'w', newline='') as output_file:
            dict_writer = csv.DictWriter(output_file, keys)
            dict_writer.writeheader()
            dict_writer.writerows(list_of_dict)
    # '''
    return file_name

def only_valid_data(data):
    '''Returns only valid gaze points. Those have values in gazeDirection_x etc.'''
    return data[(data.gazeHasValue == True) & (data.isCalibrationValid == True)]


app = Flask(__name__)

# read in data from a json dump file and return it as a dict
def read_json_dump(file_name):
    with open(file_name) as json_file:
        data = json.load(json_file)
    return data

def transform_dict_to_csv_row(input_dict):
    # Define the headers for the CSV file
    headers = [
        'eyeDataTimestamp',
        'eyeDataRelativeTimestamp',
        'frameTimestamp',
        'isCalibrationValid',
        'gazeHasValue',
        'gazeOrigin_x',
        'gazeOrigin_y',
        'gazeOrigin_z',
        'gazeDirection_x',
        'gazeDirection_y',
        'gazeDirection_z',
        'gazePointHit',
        'gazePoint_x',
        'gazePoint_y',
        'gazePoint_z'
    ]

    # Initialize a list to store the values in the desired order
    csv_row = []

    # Map the keys in the input_dict to the corresponding headers
    key_mapping = {
        'eyeDataTimestamp': 'EyeDataTimestamp',
        'eyeDataRelativeTimestamp': 'EyeDataRelativeTimestamp',
        'frameTimestamp': 'FrameTimestamp',
        'isCalibrationValid': 'IsCalibrationValid',
        'gazeHasValue': 'GazeHasValue',
        'gazeOrigin_x': 'GazeOrigin',
        'gazeOrigin_y': 'GazeOrigin',
        'gazeOrigin_z': 'GazeOrigin',
        'gazeDirection_x': 'GazeDirection',
        'gazeDirection_y': 'GazeDirection',
        'gazeDirection_z': 'GazeDirection',
        'gazePointHit': 'GazePointHit',
        'gazePoint_x': 'GazePoint',
        'gazePoint_y': 'GazePoint',
        'gazePoint_z': 'GazePoint'
    }

    # Map the values from the input_dict to the corresponding headers and order
    for header in headers:
        if '_' in header:
            # Handle keys with x, y, z
            key, sub_key = header.split('_')
            csv_row.append(input_dict.get(key_mapping[header], {}).get(sub_key, ''))
        else:
            csv_row.append(input_dict.get(key_mapping[header], ''))

    return csv_row




gzd = []

# for file gzd_1.json to gzd_4.json convert it to a df append it to gzd
for i in range(1, 5):
    # write csv header to file
    with open(f'gzd_{i}.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['eyeDataTimestamp', 'eyeDataRelativeTimestamp', 'frameTimestamp', 'isCalibrationValid', 'gazeHasValue', 'gazeOrigin_x', 'gazeOrigin_y', 'gazeOrigin_z', 'gazeDirection_x', 'gazeDirection_y', 'gazeDirection_z', 'gazePointHit', 'gazePoint_x', 'gazePoint_y', 'gazePoint_z'])
    dict_data:dict = read_json_dump(f'gzd_{i}.json')
    for j in range(len(dict_data)):
        data_dict = json.loads(dict_data[j])
        low = transform_dict_to_csv_row(data_dict)
        # append low to csv file
        with open(f'gzd_{i}.csv', 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(low)
    df = pd.DataFrame(dict_data)
    gzd.append(df)

test = gzd[0]
print(test.iloc[0])


@app.route('/gazedata', methods=['POST'])
def receive_gaze_data():
    #json to dict
    json_data = json.loads(request.data)
    print(len(json_data))

    # csv string writer
    csv_string = StringIO()
    # write csv headers
    writer = csv.writer(csv_string)
    writer.writerow(['eyeDataTimestamp', 'eyeDataRelativeTimestamp', 'frameTimestamp', 'isCalibrationValid', 'gazeHasValue', 'gazeOrigin_x', 'gazeOrigin_y', 'gazeOrigin_z', 'gazeDirection_x', 'gazeDirection_y', 'gazeDirection_z', 'gazePointHit', 'gazePoint_x', 'gazePoint_y', 'gazePoint_z'])


    # save the json to a json file + timestamp
    for obs in json_data:
        json_obs = json.loads(obs)
        low = transform_dict_to_csv_row(json_obs)
        # append low to csv file
        writer.writerow(low)

    # csv to df
    df = pd.read_csv(StringIO(csv_string.getvalue()))
    features = get_features_for_n_seconds(df, 2, "testing", "jan")
    # transform list of json to pandas df
    df = pd.DataFrame(features)
    
    feature_set = ['minFix', 'maxFix', 'minDis', 'maxDis', 'blinkMin', 'blinkMax', 'xDir', 'yDir', 'fixDensPerBB']
    features = df[feature_set]
    file_path = os.path.join(os.getcwd(), 'best_model.pkl')
    with open(file_path, 'rb') as file:
    # Use pickle.load to load the dictionary from the file
        loaded_model = pickle.load(file)
        y_pred = loaded_model.predict(features)
    print('I bet you were:', y_pred)
    
    # return 200
    if(len(y_pred) == 0):
        return json.dumps({'activity':'no prediction'}), 200, {'ContentType':'application/json'}
    return json.dumps({'activity':str(y_pred[0][0])}), 200, {'ContentType':'application/json'}



if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
