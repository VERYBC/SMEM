import cv2
import os
import torch
import math
import time 
import numpy as np
import pandas as pd
from tqdm import tqdm

from matching_model.LightGlue.lightglue import LightGlue, SuperPoint_
from matching_model.LightGlue.lightglue.utils import load_image, rbd


def haversine(loga, lata, logb, latb):
    # log   lat
    EARTH_RADIUS = 6378.137
    PI = math.pi

    lat_a = lata * PI / 180
    lat_b = latb * PI / 180
    a = lat_a - lat_b
    b = loga * PI / 180 - logb * PI / 180
    dis = 2 * math.asin(math.sqrt(math.pow(math.sin(a / 2), 2) + math.cos(lat_a) * math.cos(lat_b) * math.pow(math.sin(b / 2), 2)))

    distance = EARTH_RADIUS * dis * 1000
    return distance

# setting
DATA_LIST = [18] # X_1: Xian-18 | X_2: Xian-19 | X_3: Weinan-1 | X_4: Weinan-2
data_name  = 'Xian' # Xian | Weinan
model = 'superpoint+lightglue'
data_root = 'data/XIAN_Visloc/' + data_name + '/drone'

drone_resolution = '_392' 
read_homography_type = '_romav2' # ''|'_romav2' 
ckpt_type = '_xian' # '_uav'|'_xian' Training weight type
gsd = 0.247 

if_location     = True
read_checkpoint = True # Load the trained weights of lightglue

if data_name == 'Xian':
    orginal_resolution = [0 for _ in range(17)]+ [1959, 2083] # Original resolution of satellite image 
    satellite_resolution = '' 
    mode = ''
elif data_name == 'Weinan':
    orginal_resolution = [1959,2083] 
    satellite_resolution = '' 
    mode = ''    

save_path = 'result/' + f'location_{DATA_LIST[0]:02}_' + model + '.csv'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# SuperPoint settings
max_num_keypoints = 1024  # -1 keep all keypoints
keypoint_threshold = 0.01  # Remove keypoints with low confidence. Set to -1 to keep all keypoints.
nms_radius = 4  # Non-maxima suppression: keypoints with similar responses in a small neighborhood are removed.

config = {
    'superpoint': {
        'nms_radius': nms_radius,
        'keypoint_threshold': keypoint_threshold,
        'max_num_keypoints': max_num_keypoints,
        "detection_threshold": 0.0005
    },

}

if if_location:

    if 'superpoint' in model:
        extractor = SuperPoint_(**config['superpoint']).eval().to(device)
        if read_checkpoint:
                ckpt_path = './checkpoints/checkpoint_best' + ckpt_type + '.tar'
                print(f'Read checkpoint from: {ckpt_path}')
                ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
                state_dict = ckpt["model"]
                clean_state = {}
                for k, v in state_dict.items():
                    k = k.replace("module.", "").replace("ema.", "")
                    clean_state[k] = v

                extractor_state = {}
                matcher_state = {}

                for k, v in clean_state.items():
                    if k.startswith("extractor."):
                        extractor_state[k.replace("extractor.", "")] = v
                    elif k.startswith("matcher."):
                        matcher_state[k.replace("matcher.", "")] = v

                extractor.load_state_dict(extractor_state, strict=True)

        if 'lightglue' in model:
            matcher = LightGlue(features='superpoint', filter_threshold=0.0005).eval().to(device) 
          
            if read_checkpoint:
                matcher.load_state_dict(matcher_state, strict=False)

    # Read image path and matching results
    query_folder = data_root + f'/{DATA_LIST[0]:02}/drone' + drone_resolution
    query_paths = [os.path.join(query_folder, f) 
               for f in sorted(os.listdir(query_folder))
               if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'))]

    satellite_folder = data_root + f'/{DATA_LIST[0]:02}/satellite' + mode + satellite_resolution
    satellite_paths = [os.path.join(satellite_folder, f) for f in os.listdir(satellite_folder)
                   if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'))]

    pos_pair_path = data_root + f'/{DATA_LIST[0]:02}/satellite'+ mode + satellite_resolution + f'/pairs.csv'
    pos_info_path = data_root + f'/{DATA_LIST[0]:02}/satellite'+ mode + satellite_resolution + f'/{DATA_LIST[0]:02}_coordinates.csv'
    pos_pairs_df = pd.read_csv(pos_pair_path, header=None)
    pos_coordinates_df = pd.read_csv(pos_info_path, header=None)


    results = []
    mkpts_nums = []
    times = []
    None_count = 0
    Keypoints_insufficient = 0

    print('Model: ' + model)
    with torch.no_grad():
        start = 0
        pbar = tqdm(query_paths[start:], desc="localization：")
        for i, query_path in enumerate(pbar, start=start):

            satellite_index_name = pos_pairs_df.iloc[i, 1]
            satellite_pos = pos_coordinates_df[pos_coordinates_df.iloc[:, 0] == satellite_index_name].values[0,1:] 

            satellite_path = data_root + f'/{DATA_LIST[0]:02}/satellite'+ mode + satellite_resolution + '/'+ satellite_index_name

            image_size = (392,392)
            query_image = load_image(query_path, image_size).to(device)
            satellite_image = load_image(satellite_path, image_size).to(device)

            imageA = query_image
            imageB = satellite_image

            H_A, W_A = imageA.shape[-2:]
            H_B, W_B = imageB.shape[-2:]

            if i ==  start:
                print(f'Drone image resolution：({H_A}, {W_A})')
                print(f'Satellite image resolution：  ({H_B}, {W_B})')

            start = time.perf_counter()
            
            if 'superpoint' in model:
                feats0 = extractor.extract(query_image)
                feats1 = extractor.extract(satellite_image)
                kpts0 = feats0['keypoints'][0].cpu().numpy()
                kpts1 = feats1['keypoints'][0].cpu().numpy()
                # Lightglue
                if 'lightglue' in model:
                    matches01 = matcher({'image0': feats0, 'image1': feats1})
                    feats0_, feats1_, matches01 = [rbd(x) for x in [feats0, feats1, matches01]]  # remove batch dimension
                    matches = matches01['matches']  # indices with shape (K,2)
                    mkpts0 = feats0_['keypoints'][matches[..., 0]].cpu().numpy()  # coordinates in image #0, shape (K,2)
                    mkpts1 = feats1_['keypoints'][matches[..., 1]].cpu().numpy() # coordinates in image #1, shape (K,2)

                    scores = matches01['scores'].cpu().numpy()

                    valid_indices0 = matches[..., 0].cpu().numpy()
                    valid_indices1 = matches[..., 1].cpu().numpy()

                    pred_pairs = set(zip(valid_indices0.tolist(), valid_indices1.tolist()))

            if len(mkpts0) >= 4:
                H, mask = cv2.findHomography(mkpts0, mkpts1, cv2.RANSAC, 5.0)

                if H is not None:

                    GSD = gsd * orginal_resolution[DATA_LIST[0]-1] /  H_B
                    
                    center_query = np.array([[W_A / 2, H_A / 2]], dtype=np.float32).reshape(-1, 1, 2)
                    center_satellite_pixel = cv2.perspectiveTransform(center_query, H)

                    x = center_satellite_pixel[0, 0, 0]
                    y = center_satellite_pixel[0, 0, 1]
                    
                    lat0, lon0 = satellite_pos

                    dx_px = center_satellite_pixel[0, 0, 0] - W_B/2
                    dy_px = center_satellite_pixel[0, 0, 1] - H_B/2

                    dx_m = dx_px * GSD
                    dy_m = dy_px * GSD

                    lat = lat0 - dy_m / 110540
                    lon = lon0 + dx_m / (111320 * np.cos(np.deg2rad(lat0)))
                    
                    current_location = [lat, lon]

                    results.append([os.path.basename(query_path), current_location[0], current_location[1]])
                    mkpts_nums.append(len(mkpts0))

                    last_location = current_location
                else:
                    results.append([os.path.basename(query_path), np.nan, np.nan])
                    None_count += 1
            else:
                results.append([os.path.basename(query_path), np.nan, np.nan])
                Keypoints_insufficient += 1

            end = time.perf_counter()
            times.append(end - start)

            pbar.update(1)
            pbar.set_postfix({'H_None': None_count, 'Keypoints_insufficient': Keypoints_insufficient})

    os.makedirs('result', exist_ok=True)
    df = pd.DataFrame(results, columns=['Image', 'Longitude', 'Latitude'])
    df.to_csv(save_path, index=False)
    print('Save the results to '+ save_path)

# localization results
file_dir = os.path.join(data_root, f'{DATA_LIST[0]:02}')
drone_info_path = os.path.join(file_dir, f'{DATA_LIST[0]:02}.csv')
drone_coordinates_df = pd.read_csv(drone_info_path)
drone_pos = drone_coordinates_df.iloc[:, 2:4].values # lat,lon

location_coordinates_df = pd.read_csv(save_path)
location_pos = location_coordinates_df.iloc[:, 1:3].values # lat, lon

errors = []
for (lat_gt, lon_gt), (lat_pred, lon_pred) in zip(drone_pos, location_pos):
    if np.isnan(lon_pred) or np.isnan(lat_pred):
        errors.append(np.nan)
    else:
        dist = haversine(lon_gt, lat_gt, lon_pred, lat_pred)
        errors.append(dist)

errors = np.array(errors)
mean_error = np.nanmean(errors)
median_error = np.nanmedian(errors)
valid_count = np.sum(~np.isnan(errors))

print(f"Number of successful frames: {valid_count}/{len(errors)} ({(valid_count/len(errors)) * 100:.2f}%)")
print(f"Average localization error : {mean_error:.2f} m")
print(f"Median localization error  : {median_error:.2f} m")
print(f"Matched point pairs        : {np.array(mkpts_nums).mean():.2f}")

avg_time = sum(times) / len(times)
print(f"average time               : {avg_time * 1000:.2f} ms")

