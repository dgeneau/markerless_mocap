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
from streamlit_plotly_events import plotly_events


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

def summarize_options(options):

    summarized_list = []
    seen_bases = set()

    for opt in options:
        # Only process options that end with '_l' or '_r'
        if opt.endswith('_l') or opt.endswith('_r'):
            base_opt = opt[:-2]  # remove last two characters
            if base_opt not in seen_bases:
                seen_bases.add(base_opt)
                summarized_list.append(base_opt)

    return summarized_list


_='Begin Dashboarding'
st.set_page_config(layout="wide")


st.image('https://www.csipacific.ca/wp-content/uploads/2023/10/logo-Performance-Nation-vertical-lg.png', width = 150)
st.title("Biomechanical Running Analysis")

session_date = st.sidebar.selectbox('Select Date', ['Feb 21', 'Feb 28', 'March 7'])

if session_date == 'Feb 21':
    trial  = st.sidebar.selectbox('Select Trial', ['Running 3.6 m/s', 'Running 5.5 m/s'])

    if trial == 'Running 3.6 m/s': 
        filepath_marker = f'Collection Feb 21/Session_1/OpenCapData_d779ad48-7221-41ca-a68e-6728b177a6fb/MarkerData/run_3_6_1.trc'
        filepath_kin = 'Collection Feb 21/Session_1/OpenCapData_d779ad48-7221-41ca-a68e-6728b177a6fb/OpenSimData/Kinematics/run_3_6_1.mot'
        filepath_FP = 'Collection Feb 21/Session_1/Session_1_forces_2025_02_21_164149.csv'
        video = 'Collection Feb 21/Session_1/OpenCapData_d779ad48-7221-41ca-a68e-6728b177a6fb/Videos/Cam0/InputMedia/run_3_6_1/run_3_6_1_sync.mp4'
        start = 0
        treadmill_vel = 3.6
        plate = 1
        

    elif trial == 'Running 5.5 m/s': 
        filepath_marker = 'Collection Feb 21/Session_1/OpenCapData_d779ad48-7221-41ca-a68e-6728b177a6fb/MarkerData/run_5_5.trc'
        filepath_kin = 'Collection Feb 21/Session_1/OpenCapData_d779ad48-7221-41ca-a68e-6728b177a6fb/OpenSimData/Kinematics/run_5_5.mot'
        filepath_FP = 'Collection Feb 21/Session_1/Session_1_forces_2025_02_21_164402.csv'
        video = 'Collection Feb 21/Session_1/OpenCapData_d779ad48-7221-41ca-a68e-6728b177a6fb/Videos/Cam0/InputMedia/run_5_5/run_5_5_sync.mp4'
        start = 120*3
        treadmill_vel = 5.5
        plate = 1
        
elif session_date == 'Feb 28': 
    trial  = st.sidebar.selectbox('Select Trial', ['Running 3.6 m/s', 'Running 5.0 m/s','Running 5.5 m/s'])
    session_id = 'OpenCapData_65d2792a-163c-4f5c-818f-c9374e99940b'

    if trial == 'Running 3.6 m/s': 
        filepath_marker = f'Collection {session_date}/Session_1/{session_id}/MarkerData/run_3_6_2.trc'
        filepath_kin = f'Collection {session_date}/Session_1/{session_id}/OpenSimData/Kinematics/run_3_6_2.mot'
        filepath_FP = f'Collection {session_date}/Session_1/Session_1_forces_2025_02_28_174729.csv'
        video = f'Collection {session_date}/Session_1/{session_id}/Videos/Cam0/InputMedia/run_3_6_2/run_3_6_2_sync.mp4'
        start = 0
        treadmill_vel = 3.6
        plate = 1

    elif trial == 'Running 5.0 m/s': 
        filepath_marker = f'Collection {session_date}/Session_1/{session_id}/MarkerData/run_5_0.trc'
        filepath_kin = f'Collection {session_date}/Session_1/{session_id}/OpenSimData/Kinematics/run_5_0.mot'
        filepath_FP = f'Collection {session_date}/Session_1/Session_1_forces_2025_02_28_174246.csv'
        video = f'Collection Feb 28/Session_1/{session_id}/Videos/Cam0/InputMedia/run_5_5/run_5_5_sync.mp4'
        start = 0
        treadmill_vel = 5.0
        plate = 1


    elif trial == 'Running 5.5 m/s': 
        filepath_marker = f'Collection {session_date}/Session_1/{session_id}/MarkerData/run_5_5.trc'
        filepath_kin = f'Collection {session_date}/Session_1/{session_id}/OpenSimData/Kinematics/run_5_5.mot'
        filepath_FP = f'Collection {session_date}/Session_1/Session_1_forces_2025_02_28_174841.csv'
        video = f'Collection {session_date}/Session_1/{session_id}/Videos/Cam0/InputMedia/run_5_5/run_5_5_sync.mp4'
        start = 0
        treadmill_vel = 5.5
        plate = 1

elif session_date == 'March 7': 
    trial  = st.sidebar.selectbox('Select Trial', ['Running 3.6 m/s','Running 3.6 m/s, 3 Que', 'Running 3.6 m/s, Toes', 'Running 3.6 m/s, Ring Pits','Running 5.5 m/s', 'Running 5.5 m/s, 3 Que', 'Running 5.5 m/s, Ring Pits'])
    session_id = 'OpenCapData_8c309990-af14-4186-8d6f-29e5ad3fa95e'

    if trial == 'Running 3.6 m/s': 
        filepath_marker = f'Collection {session_date}/Session_3/{session_id}/MarkerData/run_3_6_1.trc'
        filepath_kin = f'Collection {session_date}/Session_3/{session_id}/OpenSimData/Kinematics/run_3_6_1.mot'
        filepath_FP = f'Collection {session_date}/Session_3/Session_3_forces_2025_03_07_174950.csv'
        video = f'Collection {session_date}/Session_3/{session_id}/Videos/Cam0/InputMedia/run_3_6_1/run_3_6_1_sync.mp4'
        start = 0
        treadmill_vel = 3.6
        plate = 1
    
    elif trial == 'Running 3.6 m/s, 3 Que': 
        filepath_marker = f'Collection {session_date}/Session_3/{session_id}/MarkerData/run_3_6_3Que.trc'
        filepath_kin = f'Collection {session_date}/Session_3/{session_id}/OpenSimData/Kinematics/run_3_6_3Que.mot'
        filepath_FP = f'Collection {session_date}/Session_3/Session_3_forces_2025_03_07_180208.csv'
        video = f'Collection {session_date}/Session_3/{session_id}/Videos/Cam0/InputMedia/run_3_6_3Que/run_3_6_3Que_sync.mp4'
        start = 0
        treadmill_vel = 3.6
        plate = 1

    elif trial == 'Running 3.6 m/s, Toes': 
        filepath_marker = f'Collection {session_date}/Session_3/{session_id}/MarkerData/run_3_6bendtoes_2.trc'
        filepath_kin = f'Collection {session_date}/Session_3/{session_id}/OpenSimData/Kinematics/run_3_6bendtoes_2.mot'
        filepath_FP = f'Collection {session_date}/Session_3/Session_3_forces_2025_03_07_180029.csv'
        video = f'Collection {session_date}/Session_3/{session_id}/Videos/Cam0/InputMedia/run_3_6bendtoes_2/run_3_6bendtoes_2_sync.mp4'
        start = 0
        treadmill_vel = 3.6
        plate = 1

    elif trial == 'Running 3.6 m/s, Ring Pits': 
        filepath_marker = f'Collection {session_date}/Session_3/{session_id}/MarkerData/run_3_6ringpits.trc'
        filepath_kin = f'Collection {session_date}/Session_3/{session_id}/OpenSimData/Kinematics/run_3_6ringpits.mot'
        filepath_FP = f'Collection {session_date}/Session_3/Session_3_forces_2025_03_07_175502.csv'
        video = f'Collection {session_date}/Session_3/{session_id}/Videos/Cam0/InputMedia/run_3_6ringpits/run_3_6ringpits_sync.mp4'
        start = 0
        treadmill_vel = 3.6
        plate = 1

    elif trial == 'Running 5.5 m/s': 
        filepath_marker = f'Collection {session_date}/Session_3/{session_id}/MarkerData/run_5_5.trc'
        filepath_kin = f'Collection {session_date}/Session_3/{session_id}/OpenSimData/Kinematics/run_5_5.mot'
        filepath_FP = f'Collection {session_date}/Session_3/Session_3_forces_2025_03_07_175042.csv'
        video = f'Collection {session_date}/Session_3/{session_id}/Videos/Cam0/InputMedia/run_5_5/run_5_5_sync.mp4'
        start = 0
        treadmill_vel = 5.5
        plate = 1
    
    elif trial == 'Running 5.5 m/s, 3 Que': 
        filepath_marker = f'Collection {session_date}/Session_3/{session_id}/MarkerData/run_5_5_3que.trc'
        filepath_kin = f'Collection {session_date}/Session_3/{session_id}/OpenSimData/Kinematics/run_5_5_3que.mot'
        filepath_FP = f'Collection {session_date}/Session_3/Session_3_forces_2025_03_07_180252.csv'
        video = f'Collection {session_date}/Session_3/{session_id}/Videos/Cam0/InputMedia/run_5_5_3que/run_5_5_3que_sync.mp4'
        start = 0
        treadmill_vel = 5.5
        plate = 1

    elif trial == 'Running 5.5 m/s, Ring Pits': 
        filepath_marker = f'Collection {session_date}/Session_3/{session_id}/MarkerData/run_5_5ringpits.trc'
        filepath_kin = f'Collection {session_date}/Session_3/{session_id}/OpenSimData/Kinematics/run_5_5ringpits.mot'
        filepath_FP = f'Collection {session_date}/Session_3/Session_3_forces_2025_03_07_175618.csv'
        video = f'Collection {session_date}/Session_3/{session_id}/Videos/Cam0/InputMedia/run_5_5ringpits/run_5_5ringpits_sync.mp4'
        start = 0
        treadmill_vel = 5.5
        plate = 1

df_marker = read_and_adjust_trc(filepath_marker)
df_marker = df_marker[start:].reset_index(drop=True)
df_kin = read_mot_file(filepath_kin)
df_kin = df_kin[start:].reset_index(drop=True)
treadmill_force = pd.read_csv(filepath_FP, skiprows=4)

def detect_strides(force):
        
    force_peaks,_ = find_peaks(force, height = 755, distance = 90)
 

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

_='''

Working on Brushing the data using matplot lib

- Can this be done in other plotting packages? Do I care? Questions, questions

'''

df_downsampled = treadmill_force.iloc[::10].reset_index(drop=True)
crop_fig = go.Figure()
crop_fig.add_scatter(
    x=df_downsampled["Time (s)"],
    y=df_downsampled["1:FZ"],
    mode = 'markers+lines',
    marker=dict(size=1), 
    name="Vertical Force Data"
)

# 2) Set default drag mode to "select" (instead of zoom)
crop_fig.update_layout(dragmode="select")
crop_fig.update_layout(xaxis_title = '<b>Time</b> (s)')
crop_fig.update_layout(yaxis_title = '<b>Force</b> (N)')
crop_fig.update_layout(title = '<b>Data Selector</b>')


# 3) Render the figure in Streamlit and capture user selection
#    - Box/Lasso selection is turned on by default in the Plotly figure toolbar.
selected_points = plotly_events(
   crop_fig,
    click_event=False,
    select_event=True,
    hover_event=False,
    override_height=300  # Adjust if you need a different figure height
)

if selected_points:
    # Extract the x-values of the selected points
    selected_x_vals = [pt["x"] for pt in selected_points]
    crop_start = np.where(treadmill_force["Time (s)"] == selected_x_vals[0])[0][0]
    crop_end = np.where(treadmill_force["Time (s)"] == selected_x_vals[-1])[0][0]
    
    #df_selected = treadmill_force[treadmill_force["Time (s)"].isin(selected_x_vals)]
    df_selected = treadmill_force[crop_start:crop_end]
    treadmill_force = df_selected.reset_index(drop=True)
    
else:
    treadmill_force = treadmill_force


#####################################################################################################



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
    if val > 0 : side_list.append('R')
    elif val < 0 : side_list.append('L')
    else: side_list.append('UK')



if side_list[0] == 'L':
    stride_metrics['foot_side'] = ['L' if i % 2 == 0 else 'R' for i in stride_metrics.index]
elif side_list[0] == 'R':
    stride_metrics['foot_side'] = ['R' if i % 2 == 0 else 'L' for i in stride_metrics.index]


stride_metrics['impulse'] = impulses
stride_metrics['start'] = stride_start
stride_metrics['end'] = stride_end


fig = go.Figure()

fig.add_trace(go.Scatter(
    y = treadmill_force[f'{plate}:FZ'],
    x = treadmill_force['Time (s)'], 
    name = 'Vertical Force'

))
fig.add_trace(go.Scatter(
    y = treadmill_force[f'{plate}:FX'],
    x = treadmill_force['Time (s)'], 
    name = 'Lateral Force'

))
fig.add_trace(go.Scatter(
    y = treadmill_force[f'{plate}:FY'],
    x = treadmill_force['Time (s)'], 
    name = 'Horizontal Force'

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

_='''
Average Force Trace for each step
'''
force_trace = go.Figure()



all_force_r = []
all_force_l = []

stride_starts_r = stride_metrics['start'][stride_metrics['foot_side'] == 'R'].reset_index(drop=True)
stride_ends_r = stride_metrics['end'][stride_metrics['foot_side'] == 'R'].reset_index(drop=True)

stride_starts_l = stride_metrics['start'][stride_metrics['foot_side'] == 'L'].reset_index(drop=True)
stride_ends_l = stride_metrics['end'][stride_metrics['foot_side'] == 'L'].reset_index(drop=True)


# ----- RIGHT SIDE -----
for i in range(0,len(stride_starts_r)):
    # Extract the stride for this segment
    stride_f_r = treadmill_force[f'{plate}:FZ'][stride_starts_r[i]:stride_ends_r[i]].reset_index(drop=True)
    
    # Plot individual stride
    force_trace.add_trace(go.Scatter(
        y=stride_f_r,
        line=dict(color='red'),
        opacity = .05,
    ))
    
    # Save this stride's numeric values for later averaging
    all_force_r.append(stride_f_r.values)

# Once we have all right-side strides, we can pad them with NaNs and compute an average
if all_force_r:
    # Determine the maximum stride length
    max_len_r = max(len(arr) for arr in all_force_r)

    # Pad each stride to max_len_r
    padded_force_r = []
    for arr in all_force_r:
        # Calculate how many NaNs we need
        pad_len = max_len_r - len(arr)
        # Pad at the end with NaN
        padded_arr = np.pad(arr, (0, pad_len), mode='constant', constant_values=np.nan)
        padded_force_r.append(padded_arr)

    # Convert to an array of shape (num_strides, max_stride_len)
    padded_force_r = np.array(padded_force_r)  

    # Compute average ignoring NaNs
    avg_stride_r = np.nanmean(padded_force_r, axis=0)

    # Plot the average stride
    force_trace.add_trace(go.Scatter(
        y=avg_stride_r,
        line=dict(color='red', width=5),  # thicker line
    ))

# ----- LEFT SIDE -----

all_force_l = []

for i in range(0,len(stride_starts_l)):
    # Extract the stride
    stride_f_l = treadmill_force[f'{plate}:FZ'][stride_starts_l[i]:stride_ends_l[i]].reset_index(drop=True)
      
    # Plot individual stride
    force_trace.add_trace(go.Scatter(
        y=stride_f_l,
        line=dict(color='green'),
        opacity = .05,
    ))
    
    # Save stride in list
    all_force_l .append(stride_f_l.values)

# Pad with NaNs and average, if we have left strides
if all_force_l:
    # Determine the maximum stride length
    max_len_l = max(len(arr) for arr in all_force_l)

    # Pad each stride to max_len_l
    padded_force_l = []
    for arr in all_force_l:
        pad_len = max_len_l - len(arr)
        padded_arr = np.pad(arr, (0, pad_len), mode='constant', constant_values=np.nan)
        padded_force_l.append(padded_arr)

    # Convert to array
    padded_force_l = np.array(padded_force_l)

    # Compute average ignoring NaNs
    avg_force_l = np.nanmean(padded_force_l, axis=0)

    # Plot the average stride
    force_trace.add_trace(go.Scatter(
        y=avg_force_l,
        line=dict(color='green', width=5),
    ))

force_trace.update_layout(xaxis_title = '<b>Sample Number</b>')
force_trace.update_layout(yaxis_title = '<b>Force</b> (N)')
force_trace.update_layout(title = '<b>Foot Contact Force Trace</b>')
force_trace.update_traces(showlegend=False) 

temp_force, avg_force = st.columns([5,2])
with temp_force:
    st.plotly_chart(fig)
with avg_force:
    st.plotly_chart(force_trace)

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
    st.header('Stride Impulse')
    st.metric('Average Impulse Left', imp_left)
    st.metric('Average Impulse Right', imp_right)
    st.metric('Asymetry',round(imp_left/imp_right*100,2))


#st.plotly_chart(stride_fig)


_='''
Stride Segmentation using Markers

'''

kin_start, kin_end = st.slider("Select a range of values", 0, len(df_kin), (0, len(df_kin)))
df_kin = df_kin[kin_start:kin_end].reset_index(drop=True)
df_marker = df_marker[kin_start:kin_end].reset_index(drop=True)

right_SO,_ = find_peaks(df_marker['RBigToe']*-1, distance=50)
left_SO,_ = find_peaks(df_marker['LBigToe']*-1, distance =50)

_=''''
st.header('Marker Data')

markers = st.multiselect('Select Markers to Plot', df_marker.columns)
'''
marker_fig = go.Figure()
_='''
for marker in markers:
    marker_fig.add_trace(go.Scatter(
        y = df_marker[f'{marker}'],
        x = df_marker['Time'], 
        name = f'{marker}'

    ))
'''
left_info = st.sidebar.checkbox('Show Left Stride')
right_info = st.sidebar.checkbox('Show Right Stride')

if right_info == True:
    for r_stride in right_SO:
        marker_fig.add_vline(df_marker['Time'][r_stride], line_color = 'red', name = 'Right Stride')

if left_info == True:
    for l_stride in left_SO:
        marker_fig.add_vline(df_marker['Time'][l_stride], line_color = 'green', name = 'Left Stride')

#st.plotly_chart(marker_fig)


_='''
Model Kinematics Info

'''

st.header('Model Kinematics')


joint_list = summarize_options(list(df_kin.columns))

joints = st.multiselect('Select Kinematics to Plot', joint_list)


joint_fig = go.Figure()
for joint in joints:
    joint_fig.add_trace(go.Scatter(
        y = df_kin[f'{joint}_l'],
        x = df_kin['time'], 
        name = f'{joint}'

    ))
    joint_fig.add_trace(go.Scatter(
        y = df_kin[f'{joint}_r'],
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



avg_fig = go.Figure()
avg_fig = go.Figure()
show_left = st.checkbox('Average Left Kinematics')
show_right = st.checkbox('Average Right Kinematics')

for kin in joints:
    # ----- RIGHT SIDE -----
    if show_right:
        all_strides_r = []

        for i in range(len(right_SO) - 1):
            # Extract the stride for this segment
            stride_kin_r = df_kin[f'{kin}_r'][right_SO[i]:right_SO[i+1]].reset_index(drop=True)
            
            # Plot individual stride
            avg_fig.add_trace(go.Scatter(
                y=stride_kin_r,
                line=dict(color='red'),
                opacity = .05,
                name=f'{kin}_r_stride{i}'   # optional
            ))
            
            # Save this stride's numeric values for later averaging
            all_strides_r.append(stride_kin_r.values)

        # Once we have all right-side strides, we can pad them with NaNs and compute an average
        if all_strides_r:
            # Determine the maximum stride length
            max_len_r = max(len(arr) for arr in all_strides_r)

            # Pad each stride to max_len_r
            padded_r = []
            for arr in all_strides_r:
                # Calculate how many NaNs we need
                pad_len = max_len_r - len(arr)
                # Pad at the end with NaN
                padded_arr = np.pad(arr, (0, pad_len), mode='constant', constant_values=np.nan)
                padded_r.append(padded_arr)

            # Convert to an array of shape (num_strides, max_stride_len)
            padded_r = np.array(padded_r)  

            # Compute average ignoring NaNs
            avg_stride_r = np.nanmean(padded_r, axis=0)

            # Plot the average stride
            avg_fig.add_trace(go.Scatter(
                y=avg_stride_r,
                line=dict(color='red', width=5),  # thicker line
                name=f'{kin}_r_average'
            ))

    # ----- LEFT SIDE -----
    if show_left:
        all_strides_l = []

        for i in range(len(left_SO) - 1):
            # Extract the stride
            stride_kin_l = df_kin[f'{kin}_l'][left_SO[i]:left_SO[i+1]].reset_index(drop=True)
            
            # Plot individual stride
            avg_fig.add_trace(go.Scatter(
                y=stride_kin_l,
                line=dict(color='green'),
                opacity = .05,
                name=f'{kin}_l_stride{i}'
            ))
            
            # Save stride in list
            all_strides_l.append(stride_kin_l.values)

        # Pad with NaNs and average, if we have left strides
        if all_strides_l:
            # Determine the maximum stride length
            max_len_l = max(len(arr) for arr in all_strides_l)

            # Pad each stride to max_len_l
            padded_l = []
            for arr in all_strides_l:
                pad_len = max_len_l - len(arr)
                padded_arr = np.pad(arr, (0, pad_len), mode='constant', constant_values=np.nan)
                padded_l.append(padded_arr)

            # Convert to array
            padded_l = np.array(padded_l)

            # Compute average ignoring NaNs
            avg_stride_l = np.nanmean(padded_l, axis=0)

            # Plot the average stride
            avg_fig.add_trace(go.Scatter(
                y=avg_stride_l,
                line=dict(color='green', width=5),
                name=f'{kin}_l_average'
            ))

avg_fig.update_layout(xaxis_title = '<b>Sample Number</b>')
avg_fig.update_layout(yaxis_title = '<b>Speed</b>')
avg_fig.update_layout(title = '<b>Stride Kinematics Comparison</b>')
avg_fig.update_traces(showlegend=False) 

plot, video_box = st.columns([7,1.2])
with plot: 
    st.plotly_chart(avg_fig)


video_file = open(video, "rb")
video_bytes = video_file.read()


with video_box:
    st.video(video_bytes)


avg_l, std_l, avg_r, std_r = st.columns(4)
if show_left and show_right:
    with avg_l:
        st.metric('Average Left', round(np.mean(avg_stride_l), 2))
    with std_l:
        st.metric('Standard Dev Left', round(np.std(avg_stride_l), 2))
    with avg_r:
        st.metric('Average Right', round(np.mean(avg_stride_r), 2))
    with std_r:
        st.metric('Standard Dev Right', round(np.std(avg_stride_r), 2))

