import sys
import os
import numpy as np


visibility_constant = 1.0

# landmark name lookup table
keypoints = {
    0: 'nose', 
    1: 'left_eye_inner', 2: 'left_eye', 3: 'left_eye_outer', 
    4: 'right_eye_inner', 5: 'right_eye', 6: 'right_eye_outer', 
    7: 'left_ear', 8: 'right_ear', 
    9: 'mouth_left', 10: 'mouth_right', 
    11: 'left_shoulder', 12: 'right_shoulder', 
    13: 'left_elbow', 14: 'right_elbow', 
    15: 'left_wrist', 16: 'right_wrist', 
    17: 'left_pinky', 18: 'right_pinky', 
    19: 'left_index', 20: 'right_index', 
    21: 'left_thumb', 22: 'right_thumb', 
    23: 'left_hip', 24: 'right_hip', 
    25: 'left_knee', 26: 'right_knee', 
    27: 'left_ankle', 28: 'right_ankle', 
    29: 'left_heel', 30: 'right_heel', 
    31: 'left_foot_index', 32: 'right_foot_index'
}

number_of_keypoints = len(keypoints)

csv_header = 'frame,' + ''.join([f'{v}_x,{v}_y,{v}_visibility,' for v in keypoints.values()])[:-1]

def do_path_check(input_path: str, output_path: str) -> bool:
    '''
    checks validity of io paths
    returns true if everything is fine
    '''
    if not os.path.exists(input_path):
        print('ERROR: invalid input path')
        return False

    if os.path.exists(output_path):
        while True:
            print(f'{output_path} already exists, overwrite? (y/n)')

            keyinput = input()
            if keyinput.lower() == 'y':
                return True
            elif keyinput.lower() == 'n':
                return False


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print('usage:\npython posedata_to_csv.py input_path output_path')
        sys.exit()

    input_path = sys.argv[1]
    output_path = sys.argv[2]

    if do_path_check(input_path, output_path) == False:
        print('aborting')
        sys.exit()

    with open(output_path, 'w', encoding='utf8') as f:
        data = np.genfromtxt(input_path, dtype=int, skip_header=1, delimiter=' ')

        starting_frame_id = data[0, 0]

        frames = np.zeros((data[:, 2:].shape[0], data[:, 2:].shape[1] + 1), dtype=float)
        frames[:, :-1] = data[:, 2:].astype(float)
        frames[:, 2] += visibility_constant
        frames = np.reshape(frames, (len(data) // number_of_keypoints, 3 * number_of_keypoints))
        frames = frames.astype(str)
        
        f.write(csv_header + '\n')
        for id_offset, frame in enumerate(frames):
            frame_id = id_offset + starting_frame_id
            row = f'{frame_id:08d},' + ','.join(frame) + '\n'
            f.write(row)