'''
dashboarding collection using streamlit

'''

import streamlit as st
import pandas as pd
import numpy as np 
import plotly.graph_objects as go
import scipy as sp
from scipy.signal import find_peaks
from scipy import integrate


def read_mot_file(filepath):
    """
    Reads a .mot file and returns the data as a Pandas DataFrame.

    Args:
        filepath (str): Path to the .mot file.

    Returns:
        pd.DataFrame: A DataFrame containing the motion data.
    """
    # Open the file and parse the header to find the start of the data
    with open(filepath, 'r') as file:
        lines = file.readlines()
    
    # Find the line where the column headers start
    header_start_line = 0
    for i, line in enumerate(lines):
        if line.strip().startswith("time") or line.strip().startswith("Time"):
            header_start_line = i
            break
    
    # Read the column headers
    headers = lines[header_start_line].strip().split()
    
    # Read the actual data
    data = []
    for line in lines[header_start_line + 1:]:
        if line.strip():  # Skip empty lines
            data.append([float(value) for value in line.strip().split()])
    
    # Create a DataFrame
    df = pd.DataFrame(data, columns=headers)
    return df

def read_and_adjust_trc(filepath):
    """
    Reads a .trc file, dynamically adjusts rows with mismatched columns, and returns the data as a DataFrame.
    """
    with open(filepath, 'r') as file:
        lines = file.readlines()

    # Extract headers and sub-headers
    main_headers = lines[3].strip().split('\t')
    sub_headers = lines[4].strip().split('\t')
    headers = [main if main else sub for main, sub in zip(main_headers, sub_headers)]

    # Determine expected column count
    expected_columns = len(headers)

    # Read and adjust data rows
    data_start_line = 6
    data = []
    for i, line in enumerate(lines[data_start_line:], start=data_start_line + 1):
        if line.strip():
            row = line.strip().split('\t')
            if len(row) != expected_columns:
                #print(f"Row {i} mismatch: Expected {expected_columns}, got {len(row)}. Adjusting...")
                row = row[:expected_columns] if len(row) > expected_columns else row + [''] * (expected_columns - len(row))
            data.append(row)

    # Convert to DataFrame
    df = pd.DataFrame(data, columns=headers)

    # Convert numeric columns where applicable
    for col in df.columns:
        try:
            df[col] = pd.to_numeric(df[col])
        except ValueError:
            continue

    return df


_='Begin Dashboarding'
st.set_page_config(layout="wide")


st.image('https://www.csipacific.ca/wp-content/uploads/2023/10/logo-Performance-Nation-vertical-lg.png', width = 150)
st.title("Biomechanical Running Analysis")

trial  = st.sidebar.selectbox('Select Trial', ['Running 3.6 m/s', 'Running 5.5 m/s'])

if trial == 'Running 3.6 m/s': 
    filepath_marker = 'Collection Feb 21/Session_1/OpenCapData_d779ad48-7221-41ca-a68e-6728b177a6fb/MarkerData/run_3_6_1.trc'
    filepath_kin = 'Collection Feb 21/Session_1/OpenCapData_d779ad48-7221-41ca-a68e-6728b177a6fb/OpenSimData/Kinematics/run_3_6_1.mot'
    filepath_FP = 'Collection Feb 21/Session_1/Session_1_forces_2025_02_21_164149.csv'

    treadmill_vel = 3.6
    plate = 1
    

elif trial == 'Running 5.5 m/s': 
    filepath_marker = 'Collection Feb 21/Session_1/OpenCapData_d779ad48-7221-41ca-a68e-6728b177a6fb/MarkerData/run_5_5.trc'
    filepath_kin = 'Collection Feb 21/Session_1/OpenCapData_d779ad48-7221-41ca-a68e-6728b177a6fb/OpenSimData/Kinematics/run_5_5.mot'
    filepath_FP = 'Collection Feb 21/Session_1/Session_1_forces_2025_02_21_164402.csv'

    treadmill_vel = 3.6
    plate = 1
    

df_marker = read_and_adjust_trc(filepath_marker)
df_kin = read_mot_file(filepath_kin)
treadmill_force = pd.read_csv(filepath_FP, skiprows=4)

def detect_strides(force):
        
    force_peaks,_ = find_peaks(force, height = 700, distance = 90)
 

    stride_start = []
    stride_end = []
    for peak_index in force_peaks:
        zc_indices = np.where(np.diff(np.sign(treadmill_force[f'{plate}:FZ']-2)) != 0)[0]
        left_candidates = zc_indices[zc_indices < peak_index]
        left_zc = left_candidates[-1] if len(left_candidates) > 0 else None

        right_candidates = zc_indices[zc_indices > peak_index]
        right_zc = right_candidates[0] if len(right_candidates) > 0 else None

        stride_start.append(left_zc)
        stride_end.append(right_zc)
    return stride_start, stride_end, force_peaks

def lowpass(signal, highcut, frequency):
    '''
    Apply a low-pass filter using Butterworth design

    Inputs;
    signal = array-like input
    high-cut = desired cutoff frequency
    frequency = data sample rate in Hz

    Returns;
    filtered signal
    '''
    order = 2
    nyq = 0.5 * frequency
    highcut = highcut / nyq #normalize cutoff frequency
    b, a = sp.signal.butter(order, [highcut], 'lowpass', analog=False)
    y = sp.signal.filtfilt(b, a, signal, axis=0)
    return y

treadmill_force[f'{plate}:FZ'] = lowpass(treadmill_force[f'{plate}:FZ'],50,1000)
treadmill_force[f'{plate}:FY'] = lowpass(treadmill_force[f'{plate}:FY'],50,1000)
treadmill_force[f'{plate}:FX'] = lowpass(treadmill_force[f'{plate}:FX'],50,1000)

stride_onset = np.where(treadmill_force[f'{plate}:FZ']<0)[0][0]
treadmill_force = treadmill_force.iloc[stride_onset:,:].reset_index(drop=True)

stride_start, stride_end, force_peaks = detect_strides(treadmill_force[f'{plate}:FZ'])

stride_start = list(filter(lambda x: x is not None, stride_start))
stride_end = list(filter(lambda x: x is not None, stride_end))


if len(stride_end)<len(stride_start):
    stride_start = stride_start[0:len(stride_end)]
    force_peaks = force_peaks[0:len(stride_end)]

#stride_times = np.array(treadmill_force['Time (s)'][stride_end])-np.array(treadmill_force['Time (s)'][stride_start])
stride_times = np.diff(treadmill_force['Time (s)'][stride_start])
stride_times = np.append(stride_times,[0])
stride_lengths = treadmill_vel*stride_times

stride_metrics = pd.DataFrame()
stride_metrics['length'] = stride_lengths
stride_metrics['stride_time'] = stride_times
stride_metrics['force_peak'] = np.array(treadmill_force[f'{plate}:FZ'][force_peaks])


_='''
finding the 'middle' of the force plate to differenciate left and right COP values
'''

treadmill_force[f'{plate}:COPX'] = treadmill_force[f'{plate}:COPX'] + 0.26

cop_means = []
impulses = []

for i in range(0, len(stride_start)): 
    mean = np.mean(treadmill_force[f'{plate}:COPX'][stride_start[i]:stride_end[i]])
    impulse = integrate.cumulative_trapezoid(treadmill_force[f'{plate}:FZ'][stride_start[i]:stride_end[i]], (treadmill_force['Time (s)'][stride_start[i]:stride_end[i]] - treadmill_force['Time (s)'][stride_start[i]]), initial = 0)[-1]
    cop_means.append(mean)
    impulses.append(impulse)

side_list = []
for val in cop_means:
    if val >0 : side_list.append('R')
    elif val<0 : side_list.append('L')
    else: side_list.append('UK')

if side_list[0] == 'L':
    stride_metrics['foot_side'] = ['L' if i % 2 == 0 else 'R' for i in stride_metrics.index]
elif side_list[0] == 'R':
    stride_metrics['foot_side'] = ['R' if i % 2 == 0 else 'L' for i in stride_metrics.index]


stride_metrics['impulse'] = impulses


fig = go.Figure()

fig.add_trace(go.Scatter(
    y = treadmill_force[f'{plate}:FZ'],
    x = treadmill_force['Time (s)'], 
    name = 'Vertical Force'

))
fig.add_trace(go.Scatter(
    y = treadmill_force[f'{plate}:FX'],
    x = treadmill_force['Time (s)'], 
    name = 'Horizontal Force'

))
fig.add_trace(go.Scatter(
    y = treadmill_force[f'{plate}:FY'],
    x = treadmill_force['Time (s)'], 
    name = 'Lateral Force'

))

fig.add_trace(go.Scatter(
    y = treadmill_force[f'{plate}:FZ'][force_peaks],
    x = treadmill_force['Time (s)'][force_peaks], 
    mode = 'markers', 
    line=dict(color='green'),
    name = 'Force Peak'

))

fig.add_trace(go.Scatter(
    y = treadmill_force[f'{plate}:FZ'][stride_start],
    x = treadmill_force['Time (s)'][stride_start], 
    mode = 'markers', 
    line=dict(color='black'),
    name = 'Contact'

))

fig.add_trace(go.Scatter(
    y = treadmill_force[f'{plate}:FZ'][stride_end],
    x = treadmill_force['Time (s)'][stride_end], 
    mode = 'markers', 
    line=dict(color='orange'),
    name = 'Lift-off'

))
fig.update_layout(xaxis_title = '<b>Time</b> (s)')
fig.update_layout(yaxis_title = '<b>Force</b> (N)')
fig.update_layout(title = '<b>Force Time Trace</b>')

st.plotly_chart(fig)

# Add the bar trace
stride_fig = go.Figure()
stride_fig.add_trace(go.Bar(
    y=stride_metrics['length'][stride_metrics['foot_side'] == 'L'],
    name='Left Stride Length',
    marker_color='blue'  # You can change the color here
))
stride_fig.add_trace(go.Bar(
    y=stride_metrics['length'][stride_metrics['foot_side'] == 'R'],
    name='Right Stride Length',
    marker_color='red'  # You can change the color here
))
stride_fig.update_layout(xaxis_title = '<b>Stride Number</b>')
stride_fig.update_layout(yaxis_title = '<b>Stride Length</b> (m)')
stride_fig.update_layout(title = '<b>Stride Length Plot</b>')

col1,col2,col3, col4 = st.columns([2,2,2,2])


SL_left = round(np.mean(stride_metrics['length'][stride_metrics['foot_side'] == 'L'][2:-2]),3)
SL_right = round(np.mean(stride_metrics['length'][stride_metrics['foot_side'] == 'R'][2:-2]),3)

ST_left = round(np.mean(stride_metrics['stride_time'][stride_metrics['foot_side'] == 'L'][2:-2]),3)
ST_right = round(np.mean(stride_metrics['stride_time'][stride_metrics['foot_side'] == 'R'][2:-2]),3)

PF_left = round(np.mean(stride_metrics['force_peak'][stride_metrics['foot_side'] == 'L'][2:-2]),1)
PF_right = round(np.mean(stride_metrics['force_peak'][stride_metrics['foot_side'] == 'R'][2:-2]),1)

imp_left = round(np.mean(stride_metrics['impulse'][stride_metrics['foot_side'] == 'L'][2:-2]),2)
imp_right = round(np.mean(stride_metrics['impulse'][stride_metrics['foot_side'] == 'R'][2:-2]),2)

with col1:
    st.header('Stride Time')
    st.metric('Average Stride Time Left', ST_left)
    st.metric('Average Stride Time Right', ST_right)
    st.metric('Asymetry',round(ST_left/ST_right*100, 2))

with col2:
    st.header('Stride Length')
    st.metric('Average Stride Length Left', SL_left)
    st.metric('Average Stride Length Right', SL_right)
    st.metric('Asymetry',round(SL_left/SL_right*100, 2))

with col3:
    st.header('Peak Force')
    st.metric('Average Force Peak Left', PF_left)
    st.metric('Average Force Peak Right', PF_right)
    st.metric('Asymetry',round(PF_left/PF_right*100, 2))

with col4:
    st.header('Peak Force')
    st.metric('Average Impulse Left', imp_left)
    st.metric('Average Impulse Right', imp_right)
    st.metric('Asymetry',round(imp_left/imp_right*100,2))


#st.plotly_chart(stride_fig)


_='''
Stride Segmentation using Markers
'''

right_SO,_ = find_peaks(df_marker['RBigToe']*-1, distance=50)
left_SO,_ = find_peaks(df_marker['LBigToe']*-1, distance =50)

st.header('Marker Data')

markers = st.multiselect('Select Markers to Plot', df_marker.columns)

marker_fig = go.Figure()
for marker in markers:
    marker_fig.add_trace(go.Scatter(
        y = df_marker[f'{marker}'],
        x = df_marker['Time'], 
        name = f'{marker}'

    ))
left_info = st.sidebar.checkbox('Show Left Stride')
right_info = st.sidebar.checkbox('Show Right Stride')

if right_info == True:
    for r_stride in right_SO:
        marker_fig.add_vline(df_marker['Time'][r_stride], line_color = 'red', name = 'Right Stride')

if left_info == True:
    for l_stride in left_SO:
        marker_fig.add_vline(df_marker['Time'][l_stride], line_color = 'green', name = 'Left Stride')

st.plotly_chart(marker_fig)


_='''
Model Kinematics Info

'''

st.header('Model Kinematics')

joints = st.multiselect('Select Kinematics to Plot', df_kin.columns)

joint_fig = go.Figure()
for joint in joints:
    joint_fig.add_trace(go.Scatter(
        y = df_kin[f'{joint}'],
        x = df_kin['time'], 
        name = f'{joint}'

    ))

if right_info == True:
    for r_stride in right_SO:
        joint_fig.add_vline(df_kin['time'][r_stride], line_color = 'red', name = 'Right Stride')

if left_info == True:
    for l_stride in left_SO:
        joint_fig.add_vline(df_kin['time'][l_stride], line_color = 'green', name = 'Left Stride')


st.plotly_chart(joint_fig)

if len(joints) ==2: 
    st.metric('Kinematic Asymmetry', round(df_kin[f'{joints[0]}'].mean()/df_kin[f'{joints[1]}'].mean()*100, 2))
