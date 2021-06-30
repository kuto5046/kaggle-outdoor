import os
from datetime import datetime
import multiprocessing
from typing import Collection
from sklearn.preprocessing import LabelEncoder
import wandb
from sklearn.model_selection import StratifiedKFold
from wandb.lightgbm import wandb_callback
from sklearn.metrics import mean_squared_error
from sklearn.multioutput import MultiOutputRegressor
import lightgbm as lgbm
from tqdm import tqdm
from scipy.spatial.distance import cdist
import simdkalman
import numpy as np
import pandas as pd
import pickle
import math
from math import *
from pathlib import Path
from scipy.signal import butter, lfilter
import warnings
warnings.simplefilter('ignore')

###########
# Config_
###########


class Config():
    root_dir = Path('../../input/')
    input_dir = root_dir/'google-smartphone-decimeter-challenge'
    gt_dir = root_dir/''
    seed = 1996
    max_epochs = 100
    n_splits = 5
    use_folds = [0, 1, 2, 3, 4]
    debug = False
    exp_message = "stop_reg"
    notes = "速度を回帰で求める"

def init_logger(log_file='logger.log'):
    from logging import getLogger, INFO, FileHandler,  Formatter,  StreamHandler
    logger = getLogger(__name__)
    logger.setLevel(INFO)
    handler1 = StreamHandler()
    handler1.setFormatter(Formatter("%(message)s"))
    handler2 = FileHandler(filename=log_file)
    handler2.setFormatter(Formatter("%(message)s"))
    logger.addHandler(handler1)
    logger.addHandler(handler2)
    return logger


# config global var
c = Config()
ROOT_DIR = c.root_dir
INPUT_DIR = c.input_dir
SEED = c.seed
N_SPLITS = c.n_splits
USE_FOLDS = c.use_folds
DEBUG = c.debug
EXP_MESSAGE = c.exp_message
NOTES = c.notes
EXP_NAME = str(Path().resolve()).split('/')[-1]
TARGET = 'target_Degree'

today = datetime.now().strftime('%Y-%m-%d')
logger = init_logger(log_file=f'./{today}.log')
logger.info('Start Logging...')

###########
# Utils
###########


def seed_everything(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)

def to_pickle(filename, obj):
    with open(filename, mode='wb') as f:
        pickle.dump(obj, f)

def from_pickle(filename):
    with open(filename, mode='rb') as f:
        obj = pickle.load(f)
    return obj

def load_imu_data(input_dir, phase):
    acc_df = from_pickle(input_dir/'imu_dataset_v0/acc.pkl')
    mag_df = from_pickle(input_dir/'imu_dataset_v0/mag.pkl')
    if phase == 'train':
        gt_df = pd.read_csv(input_dir/'all_ground_truth.csv')
        return acc_df, mag_df, gt_df
    else:
        return acc_df, mag_df

def visualize():
    pass

def get_ground_truth(args):
    (collectionName, phoneName), df = args
    path = INPUT_DIR/f"train/{collectionName}/{phoneName}/ground_truth.csv"
    target_df = pd.read_csv(path)
    output_df = pd.DataFrame()
    # merge and target by 'millisSinceGpsEpoch'
    for epoch, epoch_df in df.groupby('millisSinceGpsEpoch'):
        idx = (target_df['millisSinceGpsEpoch'] - epoch).abs().argmin()
        epoch_diff = epoch - target_df.loc[idx, 'millisSinceGpsEpoch']
        # epoch_df['epoch_diff'] = epoch_diff
        epoch_df['target_latDeg'] = target_df.loc[idx, 'latDeg']
        epoch_df['target_lngDeg'] = target_df.loc[idx, 'lngDeg']
        # epoch_df['speedMps'] = target_df.loc[idx, 'speedMps']
        output_df = pd.concat([output_df, epoch_df]).reset_index(drop=True)
    return output_df

def get_gt_degree(args):
    (collectionName, phoneName), df = args
    path = INPUT_DIR/f"train/{collectionName}/{phoneName}/ground_truth.csv"
    target_df = pd.read_csv(path)
    
    # merge and target by 'millisSinceGpsEpoch'
    output_df = pd.DataFrame()
    for epoch, epoch_df in df.groupby('millisSinceGpsEpoch'):
        idx = (target_df['millisSinceGpsEpoch'] - epoch).abs().argmin()
        epoch_diff = epoch - target_df.loc[idx, 'millisSinceGpsEpoch']
        # epoch_df['epoch_diff'] = epoch_diff
        epoch_df['target_latDeg'] = target_df.loc[idx, 'latDeg']
        epoch_df['target_lngDeg'] = target_df.loc[idx, 'lngDeg']
        output_df = pd.concat([output_df, epoch_df]).reset_index(drop=True)
    
    _, target_degree = calc_haversine(output_df["target_latDeg"], output_df["target_lngDeg"],output_df["target_latDeg"].shift(1), output_df["target_lngDeg"].shift(1), angle=True)
    output_df[TARGET] = target_degree
    # output_df = output_df.interpolate(limit_direction='both')
    return output_df

def get_gt_degree2(args):
    (collectionName, phoneName), df = args
    path = INPUT_DIR/f"train/{collectionName}/{phoneName}/ground_truth.csv"
    target_df = pd.read_csv(path)

    # merge and target by 'millisSinceGpsEpoch'
    output_df = pd.DataFrame()
    for epoch, epoch_df in df.groupby('millisSinceGpsEpoch'):
        idx = (target_df['millisSinceGpsEpoch'] - epoch).abs().argmin()
        epoch_diff = epoch - target_df.loc[idx, 'millisSinceGpsEpoch']
        # epoch_df['epoch_diff'] = epoch_diff
        epoch_df['target_latDeg'] = target_df.loc[idx, 'latDeg']
        epoch_df['target_lngDeg'] = target_df.loc[idx, 'lngDeg']

        epoch_df[TARGET] = 0
        gt_lat_prev = 0
        gt_lng_prev = 0
        for i in range(1, len(epoch_df)):
            if i > 1:
                res = vincenty_inverse(gt_lat_prev, gt_lng_prev, df["target_latDeg"].loc[i],df["target_lngDeg"].loc[i])
                if res:
                    epoch_df[TARGET].loc[i] = res
                else:
                    if i > 0:
                        epoch_df[TARGET].loc[i] = epoch_df[TARGET].loc[i-1]
                    else:
                        epoch_df[TARGET].loc[i] = 0

        output_df = pd.concat([output_df, epoch_df]).reset_index(drop=True)
    return output_df


def check_score(input_df: pd.DataFrame, is_return=False) -> pd.DataFrame:
    if "phone" not in input_df.columns:
        input_df["phone"] = input_df["collectionName"] + \
            "_" + input_df["phoneName"]

    if "target_latDeg" not in input_df.columns:
        processes = multiprocessing.cpu_count()
        with multiprocessing.Pool(processes=processes) as pool:
            gr = input_df.groupby(['collectionName', 'phoneName'])
            dfs = pool.imap_unordered(get_ground_truth, gr)
            dfs = tqdm(dfs, total=len(gr))
            dfs = list(dfs)
        input_df = pd.concat(dfs).sort_values(
            ['collectionName', 'phoneName', 'millisSinceGpsEpoch']).reset_index(drop=True)

    output_df = input_df.copy()

    output_df['error'] = input_df.apply(
        lambda r: calc_haversine(
            r.latDeg, r.lngDeg, r.target_latDeg, r.target_lngDeg
        ),
        axis=1
    )

    meter_score = output_df['error'].mean()
    print(f'mean error: {meter_score}')

    scores = []
    p_50_scores = []
    p_95_scores = []
    mean_scores = []
    phones = []
    score_df = pd.DataFrame()
    for phone in output_df['phone'].unique():
        _index = output_df['phone'] == phone
        p_50 = np.percentile(output_df.loc[_index, 'error'], 50)
        p_95 = np.percentile(output_df.loc[_index, 'error'], 95)
        # print(f"{phone} | 50:{p_50:.5g}| 95:{p_95:.5g}")
        p_50_scores.append(p_50)
        p_95_scores.append(p_95)
        mean_scores.append(np.mean([p_50, p_95]))
        phones.append(phone)

        scores.append(p_50)
        scores.append(p_95)

    score_df["phone"] = phones
    score_df["p_50_score"] = p_50_scores
    score_df["p_95_score"] = p_95_scores
    score_df["mean_score"] = mean_scores

    comp_score = sum(scores) / len(scores)
    print(f"competition metric:{comp_score}")
    if is_return:
        return output_df, score_df


#############
# for IMU preprocess
#############
def gnss_log_to_dataframes(path):
    # logger.info(f'Loading {path}')
    gnss_section_names = {'Raw', 'UncalAccel', 'UncalGyro',
                          'UncalMag', 'Fix', 'Status', 'OrientationDeg'}
    with open(path) as f_open:
        datalines = f_open.readlines()

    datas = {k: [] for k in gnss_section_names}
    gnss_map = {k: [] for k in gnss_section_names}
    for dataline in datalines:
        is_header = dataline.startswith('#')
        dataline = dataline.strip('#').strip().split(',')
        # skip over notes, version numbers, etc
        if is_header and dataline[0] in gnss_section_names:
            try:
                gnss_map[dataline[0]] = dataline[1:]
            except:
                pass
        elif not is_header:
            try:
                datas[dataline[0]].append(dataline[1:])
            except:
                pass
    results = dict()
    for k, v in datas.items():
        results[k] = pd.DataFrame(v, columns=gnss_map[k])
    # pandas doesn't properly infer types from these lists by default
    for k, df in results.items():
        for col in df.columns:
            if col == 'CodeType':
                continue
            try:
                results[k][col] = pd.to_numeric(results[k][col])
            except:
                pass
    return results

# lowpass filter
def butter_lowpass(cutoff, fs, order):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff=2.5, fs=50.0, order=3):
    b, a = butter_lowpass(cutoff, fs, order)
    y = lfilter(b, a, data)
    return y

# Offset correction
# refarence https://github.com/J-ROCKET-BOY/SS-Fitting
def SS_fit(data):

    x = data[:, [0]]
    y = data[:, [1]]
    z = data[:, [2]]

    data_len = len(x)

    x2 = np.power(x, 2)
    y2 = np.power(y, 2)
    z2 = np.power(z, 2)

    r1 = -x*(x2+y2+z2)
    r2 = -y*(x2+y2+z2)
    r3 = -z*(x2+y2+z2)
    r4 = -(x2+y2+z2)

    left = np.array([[np.sum(x2), np.sum(x*y), np.sum(x*z), np.sum(x)],
                     [np.sum(x*y), np.sum(y2), np.sum(y*z), np.sum(y)],
                     [np.sum(x*z), np.sum(y*z), np.sum(z2), np.sum(z)],
                     [np.sum(x), np.sum(y), np.sum(z), data_len]])

    right = np.array([np.sum(r1),
                      np.sum(r2),
                      np.sum(r3),
                      np.sum(r4)])

    # si = np.dot(np.linalg.inv(left), right)
    si = np.dot(np.linalg.pinv(left), right)

    x0 = (-1/2) * si[0]
    y0 = (-1/2) * si[1]
    z0 = (-1/2) * si[2]

    return np.array([x0, y0, z0])

# Vincenty's formulae
# refarence https://qiita.com/r-fuji/items/99ca549b963cedc106ab
# lat1, lon1, lat2, lon2 -> distance, degree
def vincenty_inverse(lat1, lon1, lat2, lon2):
    # Not advanced
    if isclose(lat1, lat2) and isclose(lon1, lon2):
        return False

    # WGS84
    a = 6378137.0
    ƒ = 1 / 298.257223563
    b = (1 - ƒ) * a

    lat_1 = atan((1 - ƒ) * tan(radians(lat1)))
    lat_2 = atan((1 - ƒ) * tan(radians(lat2)))

    lon_diff = radians(lon2) - radians(lon1)
    λ = lon_diff

    for i in range(1000):
        sinλ = sin(λ)
        cosλ = cos(λ)
        sinσ = sqrt((cos(lat_2) * sinλ) ** 2 + (cos(lat_1) *
                    sin(lat_2) - sin(lat_1) * cos(lat_2) * cosλ) ** 2)
        cosσ = sin(lat_1) * sin(lat_2) + cos(lat_1) * cos(lat_2) * cosλ
        σ = atan2(sinσ, cosσ)
        sinα = cos(lat_1) * cos(lat_2) * sinλ / sinσ
        cos2α = 1 - sinα ** 2
        cos2σm = cosσ - 2 * sin(lat_1) * sin(lat_2) / cos2α
        C = ƒ / 16 * cos2α * (4 + ƒ * (4 - 3 * cos2α))
        λʹ = λ
        λ = lon_diff + (1 - C) * ƒ * sinα * (σ + C * sinσ *
                                             (cos2σm + C * cosσ * (-1 + 2 * cos2σm ** 2)))

        if abs(λ - λʹ) <= 1e-12:
            break
    else:
        return None

    α = atan2(cos(lat_2) * sinλ, cos(lat_1) *
              sin(lat_2) - sin(lat_1) * cos(lat_2) * cosλ)

    if α < 0:
        α = α + pi * 2

    return degrees(α)

def calc_haversine(lat1, lon1, lat2, lon2, angle=False):
    """Calculates the great circle distance between two points
    on the earth. Inputs are array-like and specified in decimal degrees.
    """
    RADIUS = 6_367_000
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + \
        np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    dist = 2 * RADIUS * np.arcsin(a**0.5)

    if angle:
        angle = np.degrees(np.arctan2(dlat, dlon))  # radian -> degree
        angle = (angle + 360) % 360
        angle = angle.interpolate(limit_direction='both')
        return dist, angle
    else:
        return dist

#  lat1, lon1, degree, distancd -> lat2, lon2
def vincenty_direct(row):
    # https://github.com/pktrigg/pyall/blob/master/geodetic.py
    # https://gist.github.com/jtornero/9f3ddabc6a89f8292bb2
    # -------------------------------------------------------------------------------
    # Vincenty's Direct formulae							|
    # Given: latitude and longitude of a point (latitude1, longitude1) and 			|
    # the geodetic azimuth (alpha1Tp2) 						|
    # and ellipsoidal distance in metres (s) to a second point,			|
    # 										|
    # Calculate: the latitude and longitude of the second point (latitude2, longitude2) 	|
    # and the reverse azimuth (alpha21).						|
    # 										|
    # -------------------------------------------------------------------------------
    """
    Returns the lat and long of projected point and reverse azimuth
    given a reference point and a distance and azimuth to project.
    lats, longs and azimuths are passed in decimal degrees
    Returns ( latitude2,  longitude2,  alpha2To1 ) as a tuple 
    """

    latitude1, longitude1, alpha1To2, s = row['latDeg'], row['lngDeg'], row['Degree'], row['distance']+1.0e-9
    f = 1.0 / 298.257223563		# WGS84
    a = 6378137.0 			# metres

    piD4 = math.atan(1.0)
    two_pi = piD4 * 8.0

    latitude1 = latitude1 * piD4 / 45.0
    longitude1 = longitude1 * piD4 / 45.0
    alpha1To2 = alpha1To2 * piD4 / 45.0
    if (alpha1To2 < 0.0):
        alpha1To2 = alpha1To2 + two_pi
    if (alpha1To2 > two_pi):
        alpha1To2 = alpha1To2 - two_pi

    b = a * (1.0 - f)

    TanU1 = (1-f) * math.tan(latitude1)
    U1 = math.atan(TanU1)
    sigma1 = math.atan2(TanU1, math.cos(alpha1To2))
    Sinalpha = math.cos(U1) * math.sin(alpha1To2)
    cosalpha_sq = 1.0 - Sinalpha * Sinalpha

    u2 = cosalpha_sq * (a * a - b * b) / (b * b)
    A = 1.0 + (u2 / 16384) * (4096 + u2 * (-768 + u2 *
                                           (320 - 175 * u2)))
    B = (u2 / 1024) * (256 + u2 * (-128 + u2 * (74 - 47 * u2)))

    # Starting with the approximation
    sigma = (s / (b * A))

    last_sigma = 2.0 * sigma + 2.0  # something impossible

    # Iterate the following three equations
    #  until there is no significant change in sigma

    # two_sigma_m , delta_sigma
    while (abs((last_sigma - sigma) / sigma) > 1.0e-9):
        two_sigma_m = 2 * sigma1 + sigma

        delta_sigma = B * math.sin(sigma) * (math.cos(two_sigma_m)
                                             + (B/4) * (math.cos(sigma) *
                                                        (-1 + 2 * math.pow(math.cos(two_sigma_m), 2) -
                                                         (B/6) * math.cos(two_sigma_m) *
                                                         (-3 + 4 * math.pow(math.sin(sigma), 2)) *
                                                         (-3 + 4 * math.pow(math.cos(two_sigma_m), 2))))) \

        last_sigma = sigma
        sigma = (s / (b * A)) + delta_sigma

    latitude2 = math.atan2((math.sin(U1) * math.cos(sigma) + math.cos(U1) * math.sin(sigma) * math.cos(alpha1To2)),
                           ((1-f) * math.sqrt(math.pow(Sinalpha, 2) +
                                              pow(math.sin(U1) * math.sin(sigma) - math.cos(U1) * math.cos(sigma) * math.cos(alpha1To2), 2))))

    lembda = math.atan2((math.sin(sigma) * math.sin(alpha1To2)), (math.cos(U1) * math.cos(sigma) -
                                                                  math.sin(U1) * math.sin(sigma) * math.cos(alpha1To2)))

    C = (f/16) * cosalpha_sq * (4 + f * (4 - 3 * cosalpha_sq))

    omega = lembda - (1-C) * f * Sinalpha *  \
        (sigma + C * math.sin(sigma) * (math.cos(two_sigma_m) +
                                        C * math.cos(sigma) * (-1 + 2 * math.pow(math.cos(two_sigma_m), 2))))

    longitude2 = longitude1 + omega

    alpha21 = math.atan2(Sinalpha, (-math.sin(U1) * math.sin(sigma) +
                                    math.cos(U1) * math.cos(sigma) * math.cos(alpha1To2)))

    alpha21 = alpha21 + two_pi / 2.0
    if (alpha21 < 0.0):
        alpha21 = alpha21 + two_pi
    if (alpha21 > two_pi):
        alpha21 = alpha21 - two_pi

    latitude2 = latitude2 * 45.0 / piD4
    longitude2 = longitude2 * 45.0 / piD4
    alpha21 = alpha21 * 45.0 / piD4
    return latitude2, longitude2


def calc3(row):
    deg = - degrees(atan2(-1*row['calc2'], row['calc1']))
    if deg < 0:
        deg += 360
    return deg

# https://www.kaggle.com/museas/estimating-the-direction-with-a-magnetic-sensor
def calc_degree_loop(args):
    (collection_name, phone_name, phase), base_df = args
    imu_path = INPUT_DIR / f"{phase}/{collection_name}/{phone_name}/{phone_name}_GnssLog.txt"
    imu = gnss_log_to_dataframes(imu_path)
    acc_df = imu['UncalAccel']
    mag_df = imu['UncalMag']

    # TODO 単純に変換できるわけではない
    acc_df["millisSinceGpsEpoch"] = acc_df["utcTimeMillis"] - 315964800000
    mag_df["millisSinceGpsEpoch"] = mag_df["utcTimeMillis"] - 315964800000
    # acce filtering and smooting
    acc_df["global_x"] = acc_df["UncalAccelZMps2"]
    acc_df["global_y"] = acc_df["UncalAccelXMps2"]
    acc_df["global_z"] = acc_df["UncalAccelYMps2"]

    acc_df["x_f"] = butter_lowpass_filter(acc_df["global_x"])
    acc_df["y_f"] = butter_lowpass_filter(acc_df["global_y"])
    acc_df["z_f"] = butter_lowpass_filter(acc_df["global_z"])

    smooth_range = 1000
    acc_df["x_f"] = acc_df["x_f"].rolling(
        smooth_range, center=True, min_periods=1).mean()
    acc_df["y_f"] = acc_df["y_f"].rolling(
        smooth_range, center=True, min_periods=1).mean()
    acc_df["z_f"] = acc_df["z_f"].rolling(
        smooth_range, center=True, min_periods=1).mean()

    # magn filtering and smooting , offset correction
    mag_df["global_mx"] = mag_df["UncalMagZMicroT"]
    mag_df["global_my"] = mag_df["UncalMagYMicroT"]
    mag_df["global_mz"] = mag_df["UncalMagXMicroT"]
    mag_df["global_mx"] = mag_df["global_mx"].rolling(
        smooth_range,  min_periods=1).mean()
    mag_df["global_my"] = mag_df["global_mz"].rolling(
        smooth_range,  min_periods=1).mean()
    mag_df["global_mz"] = mag_df["global_my"].rolling(
        smooth_range,  min_periods=1).mean()

    offset = SS_fit(
        np.array(mag_df[["global_mx", "global_my", "global_mz"]]))
    mag_df["global_mx"] = (mag_df["global_mx"] - offset[0])*-1
    mag_df["global_my"] = mag_df["global_my"] - offset[1]
    mag_df["global_mz"] = mag_df["global_mz"] - offset[2]

    # merge the value of the one with the closest time
    # TODO +10で良い？
    acc_df["sSinceGpsEpoch"] = acc_df["millisSinceGpsEpoch"]//1000 + 10
    mag_df["sSinceGpsEpoch"] = mag_df["millisSinceGpsEpoch"]//1000 + 10
    base_df["sSinceGpsEpoch"] = base_df["millisSinceGpsEpoch"]//1000

    # TODO mergeの仕方
    acc_df = pd.merge_asof(
            acc_df.sort_values("sSinceGpsEpoch"), 
            mag_df[["global_mx", "global_my", "global_mz", "sSinceGpsEpoch"]].sort_values("sSinceGpsEpoch"), on='sSinceGpsEpoch', direction='nearest')
    
    output_df = pd.merge_asof(
            base_df.sort_values("sSinceGpsEpoch"), 
            acc_df[["sSinceGpsEpoch", "x_f", "y_f", "z_f", "global_mx", "global_my", "global_mz"]].sort_values("sSinceGpsEpoch"), 
            on='sSinceGpsEpoch', direction='nearest')

    # TODO as a sensor value when stopped
    start_mean_range = 10
    x_start_mean = output_df[:start_mean_range]["x_f"].mean()
    y_start_mean = output_df[:start_mean_range]["y_f"].mean()
    z_start_mean = output_df[:start_mean_range]["z_f"].mean()

    # roll and picth, device tilt
    r = atan(y_start_mean/z_start_mean)
    p = atan(x_start_mean/(y_start_mean**2 + z_start_mean**2)**0.5)

    # calculation　degrees
    output_df["calc1"] = output_df["global_mx"] * \
        cos(p) + output_df["global_my"]*sin(r) * \
        sin(p) + output_df["global_mz"]*sin(p)*cos(r)
    output_df["calc2"] = output_df["global_mz"] * \
        sin(r) - output_df["global_my"]*cos(r)
    output_df["Degree"] = output_df.apply(calc3, axis=1)
    return output_df
 
def calc_degree(input_df, phase):
    """
    collection='2020-08-03-US-MTV-1'
    phone = 'Mi8'
    一部データではimuデータがないので同じcollectionの別の端末のデータで置換する
    """
    logger.info("calc degree...")
    input_df["phase"] = phase
    gr = input_df.groupby(['collectionName', 'phoneName', 'phase'])
    processes = multiprocessing.cpu_count()
    with multiprocessing.Pool(processes=processes) as pool:
        dfs = pool.imap_unordered(calc_degree_loop, gr)
        dfs = tqdm(dfs, total=len(gr))
        dfs = list(dfs)
    output_df = pd.concat(dfs).sort_values(['collectionName', 'phoneName', 'millisSinceGpsEpoch']).reset_index(drop=True)   
    output_df = output_df.drop(["phase"], axis=1)
    return output_df

def drop_nan_imu_data(input_df):
    """
    欠損しているcollection
    [train]
    2020-08-03-US-MTV-1
    2020-08-06-US-MTV-2
    [test]
    2020-08-03-US-MTV-2
    2020-08-13-US-MTV-1
    """
    # collection内のすべての端末で欠損ありならそのcollectionの方角は算出しない
    df_list = []
    for collection_name, df in input_df.groupby('collectionName'):
        if df['Degree'].notna().sum() == len(df):
            df_list.append(df)
        elif df['Degree'].notna().sum() == 0:
            logger.info(f'All data is nan in {collection_name}')
        else:
            # 欠損は同じcollectionのもので補間する?
            # と思ったけどなかった。
            logger.info(f'There are nan in {collection_name}')
    output_df = pd.concat(df_list).reset_index(drop=True)
    return output_df


##########
# preprocessing
##########
def apply_pp(input_df):
    output_df = input_df.copy()
    # output_df = linear_interpolation(output_df)
    # output_df = apply_kf_smoothing(output_df)
    # output_df = apply_mean(output_df)
    # output_df = get_removedevice(output_df, device='SamsungS20Ulta')
    assert input_df.shape == output_df.shape
    return output_df

def apply_kf_smoothing(df):
    logger.info('[START] Kalman Smoothing')
    # define kf model
    T = 1.0
    state_transition = np.array([[1, 0, T, 0, 0.5 * T ** 2, 0], [0, 1, 0, T, 0, 0.5 * T ** 2], [0, 0, 1, 0, T, 0],
                                [0, 0, 0, 1, 0, T], [0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 1]])
    process_noise = np.diag(
        [1e-5, 1e-5, 5e-6, 5e-6, 1e-6, 1e-6]) + np.ones((6, 6)) * 1e-9
    observation_model = np.array([[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0]])
    observation_noise = np.diag([5e-5, 5e-5]) + np.ones((2, 2)) * 1e-9

    kf = simdkalman.KalmanFilter(
        state_transition=state_transition,
        process_noise=process_noise,
        observation_model=observation_model,
        observation_noise=observation_noise)

    unique_paths = df[['collectionName', 'phoneName']
                      ].drop_duplicates().to_numpy()
    for collection, phone in tqdm(unique_paths):
        cond = np.logical_and(df['collectionName'] ==
                              collection, df['phoneName'] == phone)
        data = df[cond][['latDeg', 'lngDeg']].to_numpy()
        data = data.reshape(1, len(data), 2)
        smoothed = kf.smooth(data)
        df.loc[cond, 'latDeg'] = smoothed.states.mean[0, :, 0]
        df.loc[cond, 'lngDeg'] = smoothed.states.mean[0, :, 1]
    return df

def linear_interpolation(input_df, speed_thr=45):
    logger.info('[START] linear interpolation')
    dfs = pd.DataFrame()
    use_col = input_df.columns
    for (collectionName, phoneName), df in input_df.groupby(['collectionName', 'phoneName']):

        df['delta'] = calc_haversine(
            df['latDeg'], df['lngDeg'], df['latDeg'].shift(1), df['lngDeg'].shift(1))
        df['time_delta'] = df['millisSinceGpsEpoch'] - \
            df['millisSinceGpsEpoch'].shift(1)
        df['delta'].fillna(0, inplace=True)
        df['time_delta'].fillna(0, inplace=True)
        df['speed'] = df['delta'] / (df['time_delta']/1000)  # m/s
        df['speed'].fillna(0, inplace=True)

        # 一度欠損値にする
        df.loc[speed_thr < df['speed'], ['latDeg', 'lngDeg']] = np.nan
        df['dummy_datetime'] = pd.to_datetime(df['millisSinceGpsEpoch'])
        df = df.set_index('dummy_datetime')

        # 時間に合わせて線形補間
        df = df.interpolate(method='time').reset_index(drop=True)
        dfs = pd.concat([dfs, df]).reset_index(drop=True)
    return dfs[use_col]

def make_lerp_data(df):
    '''
    Generate interpolated lat,lng values for different phone times in the same collection.
    '''
    org_columns = df.columns

    # Generate a combination of time x collection x phone and combine it with the original data (generate records to be interpolated)
    time_list = df[['collectionName', 'millisSinceGpsEpoch']].drop_duplicates()
    phone_list = df[['collectionName', 'phoneName']].drop_duplicates()
    tmp = time_list.merge(phone_list, on='collectionName', how='outer')

    lerp_df = tmp.merge(
        df, on=['collectionName', 'millisSinceGpsEpoch', 'phoneName'], how='left')

    lerp_df['phone'] = lerp_df['collectionName'] + '_' + lerp_df['phoneName']
    lerp_df = lerp_df.sort_values(['phone', 'millisSinceGpsEpoch'])

    # linear interpolation
    lerp_df['latDeg_prev'] = lerp_df['latDeg'].shift(1)
    lerp_df['latDeg_next'] = lerp_df['latDeg'].shift(-1)
    lerp_df['lngDeg_prev'] = lerp_df['lngDeg'].shift(1)
    lerp_df['lngDeg_next'] = lerp_df['lngDeg'].shift(-1)
    lerp_df['phone_prev'] = lerp_df['phone'].shift(1)
    lerp_df['phone_next'] = lerp_df['phone'].shift(-1)
    lerp_df['time_prev'] = lerp_df['millisSinceGpsEpoch'].shift(1)
    lerp_df['time_next'] = lerp_df['millisSinceGpsEpoch'].shift(-1)

    # Leave only records to be interpolated(missing coords data)
    lerp_df = lerp_df[(lerp_df['latDeg'].isnull()) & (lerp_df['phone'] == lerp_df['phone_prev']) & (
        lerp_df['phone'] == lerp_df['phone_next'])].copy()
    # calc lerp
    lerp_df['latDeg'] = lerp_df['latDeg_prev'] + ((lerp_df['latDeg_next'] - lerp_df['latDeg_prev']) * (
        (lerp_df['millisSinceGpsEpoch'] - lerp_df['time_prev']) / (lerp_df['time_next'] - lerp_df['time_prev'])))
    lerp_df['lngDeg'] = lerp_df['lngDeg_prev'] + ((lerp_df['lngDeg_next'] - lerp_df['lngDeg_prev']) * (
        (lerp_df['millisSinceGpsEpoch'] - lerp_df['time_prev']) / (lerp_df['time_next'] - lerp_df['time_prev'])))

    # Leave only the data that has a complete set of previous and next data.
    lerp_df = lerp_df[~lerp_df['latDeg'].isnull()]

    return lerp_df[org_columns]

def calc_mean_pred(df, lerp_df):
    '''
    Make a prediction based on the average of the predictions of phones in the same collection.
    '''
    add_lerp = pd.concat([df, lerp_df])
    mean_pred_result = add_lerp.groupby(['collectionName', 'millisSinceGpsEpoch'])[
        ['latDeg', 'lngDeg']].mean().reset_index()
    mean_pred_df = df.copy()
    mean_pred_df = mean_pred_df.drop(["latDeg", "lngDeg"], axis=1)
    mean_pred_df = mean_pred_df.merge(mean_pred_result[['collectionName', 'millisSinceGpsEpoch', 'latDeg', 'lngDeg']], on=[
                                      'collectionName', 'millisSinceGpsEpoch'], how='left')
    return mean_pred_df

def apply_mean(df):
    logger.info('[START] phone-mean')
    lerp = make_lerp_data(df)
    mean_df = calc_mean_pred(df, lerp)
    return mean_df

def get_removedevice(input_df: pd.DataFrame, device: str) -> pd.DataFrame:
    logger.info('[START] remove device')
    input_df['index'] = input_df.index
    input_df = input_df.sort_values('millisSinceGpsEpoch')
    input_df.index = input_df['millisSinceGpsEpoch'].values

    output_df = pd.DataFrame()
    for _, subdf in input_df.groupby('collectionName'):

        phones = subdf['phoneName'].unique()

        if (len(phones) == 1) or (not device in phones):
            output_df = pd.concat([output_df, subdf])
            continue

        origin_df = subdf.copy()

        _index = subdf['phoneName'] == device
        subdf.loc[_index, 'latDeg'] = np.nan
        subdf.loc[_index, 'lngDeg'] = np.nan
        subdf = subdf.interpolate(method='index', limit_area='inside')

        _index = subdf['latDeg'].isnull()
        subdf.loc[_index, 'latDeg'] = origin_df.loc[_index, 'latDeg'].values
        subdf.loc[_index, 'lngDeg'] = origin_df.loc[_index, 'lngDeg'].values

        output_df = pd.concat([output_df, subdf])

    output_df.index = output_df['index'].values
    output_df = output_df.sort_index()

    del output_df['index']

    return output_df


###########
# ML
###########
def preprocess(input_df, le1, le2):
    output_df = input_df.copy()
    # LE
    output_df['collectionName'] = le1.transform(input_df['collectionName'])
    output_df['phoneName'] = le2.transform(input_df['phoneName'])
    return output_df

"""
def create_features(input_df, phase="train"):
    output_df = pd.DataFrame()
    for (collection_name, phone_name), df in input_df.groupby(["collectionName", "phoneName"]):
        df = df.sort_values("millisSinceGpsEpoch").reset_index(drop=True)
        if phase == "train":
            df = df[["collectionName", "phoneName",
                     "millisSinceGpsEpoch", "latDeg", "lngDeg", "speedMps"]]
        else:
            df = df[["collectionName", "phoneName",
                     "millisSinceGpsEpoch", "latDeg", "lngDeg"]]

        for i in [1, 3, 5, 7, 10, 15, 20]:
            df[f"pre{i}_dist"] = calc_haversine(
                df["latDeg"], df["lngDeg"], df["latDeg"].shift(i), df["lngDeg"].shift(i)).abs()
            df[f"post{i}_dist"] = calc_haversine(
                df["latDeg"], df["lngDeg"], df["latDeg"].shift(i*-1), df["lngDeg"].shift(i*-1)).abs()
            df[f"pre{i}_delta_t"] = (
                df["millisSinceGpsEpoch"] - df["millisSinceGpsEpoch"].shift(i)).abs() / 1000
            df[f"post{i}_delta_t"] = (
                df["millisSinceGpsEpoch"] - df["millisSinceGpsEpoch"].shift(i*-1)).abs() / 1000
            df[f"pre{i}_speedMps"] = df[f"pre{i}_dist"] / df[f"pre{i}_delta_t"]
            df[f"post{i}_speedMps"] = df[f"post{i}_dist"] / \
                df[f"post{i}_delta_t"]
            df[f"mean{i}_speedMps"] = (
                df[f"pre{i}_speedMps"] + df[f"post{i}_speedMps"]) / 2

        df["elapsed_time"] = (df["millisSinceGpsEpoch"] -
                              df["millisSinceGpsEpoch"].min()) / 1000
        df["arrived_time"] = (
            df["millisSinceGpsEpoch"].max() - df["millisSinceGpsEpoch"]) / 1000
        output_df = pd.concat([output_df, df]).reset_index(drop=True)
    assert len(input_df) == len(output_df)
    return output_df
"""

def create_features(input_df, phase):
    output_df = pd.DataFrame()
    for (collection_name, phone_name), df in input_df.groupby(["collectionName", "phoneName"]):
        df = df.sort_values("millisSinceGpsEpoch").reset_index(drop=True)
        # df = df[["collectionName", "phoneName", "millisSinceGpsEpoch", "latDeg", "lngDeg", "Degree"]]

        for i in [1, 3, 5, 7, 10, 15, 20]:
        
            df[f"pre{i}_dist"] = calc_haversine(df["latDeg"], df["lngDeg"], df["latDeg"].shift(i), df["lngDeg"].shift(i)).abs()
            df[f"post{i}_dist"] = calc_haversine(df["latDeg"], df["lngDeg"], df["latDeg"].shift(i*-1), df["lngDeg"].shift(i*-1)).abs()
            df[f"pre{i}_Degree"] = df["Degree"].shift(i)
            df[f"post{i}_Degree"] = df["Degree"].shift(i*-1)
            df[f"pre{i}_delta_t"] = (df["millisSinceGpsEpoch"] - df["millisSinceGpsEpoch"].shift(i*1)).abs() / 1000
            df[f"post{i}_delta_t"] = (df["millisSinceGpsEpoch"] - df["millisSinceGpsEpoch"].shift(i*-1)).abs() / 1000
            df[f"pre{i}_speedMps"] = df[f"pre{i}_dist"] / df[f"pre{i}_delta_t"]
            df[f"post{i}_speedMps"] = df[f"post{i}_dist"] / df[f"post{i}_delta_t"]
            df[f"mean{i}_speedMps"] = (df[f"pre{i}_speedMps"] + df[f"post{i}_speedMps"]) / 2

        df["elapsed_time"] = (df["millisSinceGpsEpoch"] - df["millisSinceGpsEpoch"].min()) / 1000
        df["arrived_time"] = (df["millisSinceGpsEpoch"].max() - df["millisSinceGpsEpoch"]) / 1000
        output_df = pd.concat([output_df, df]).reset_index(drop=True)
    assert len(input_df) == len(output_df)
    return output_df

def fit_lgbm(X,
             y,
             test,
             params: dict = None,
             verbose: int = 50):
    """lightGBM を CrossValidation の枠組みで学習を行なう function"""
    # パラメータがないときは、空の dict で置き換える
    if params is None:
        params = {}

    oofs = []  # 全てのoofをdfで格納する
    preds = []  # 全ての予測値をdfで格納する
    val_scores = []

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    train_fold = [(trn_idx, val_idx)
                  for trn_idx, val_idx in skf.split(X, X['collectionName'])]
    for fold in range(5):
        # 指定したfoldのみループを回す
        if fold not in USE_FOLDS:
            continue

        print('=' * 20)
        print(f'Fold {fold}')
        print('=' * 20)

        # training data の target と同じだけのゼロ配列を用意
        oof_pred = np.zeros(y.shape[0], dtype=np.float)

        # train/valid data
        trn_idx_for_train, val_idx_for_train = train_fold[fold]
        x_train = X.loc[trn_idx_for_train, :].reset_index(drop=True)
        x_valid = X.loc[val_idx_for_train, :].reset_index(drop=True)
        y_train = y.loc[trn_idx_for_train].reset_index(drop=True)
        y_valid = y.loc[val_idx_for_train].reset_index(drop=True)

        # clf = MultiOutputRegressor(lgbm.LGBMRegressor(**params))
        clf = lgbm.LGBMRegressor(**params)

        # loggers
        RUN_NAME = EXP_NAME + "_" + EXP_MESSAGE
        wandb.init(project='outdoor', entity='kuto5046', group=RUN_NAME)
        wandb.run.name = RUN_NAME + f'-fold-{fold}'
        wandb_config = wandb.config
        wandb_config['model_name'] = "lightGBM"
        wandb_config['comment'] = NOTES
        # wandb.watch(clf)

        clf.fit(x_train, y_train,
                eval_set=[(x_valid, y_valid)],
                early_stopping_rounds=100,
                verbose=verbose,
                callbacks=[wandb_callback()])

        pred_i = clf.predict(x_valid)
        x_valid["pred"] = pred_i
        x_valid["speedMps"] = y_valid.to_numpy()
        oofs.append(x_valid)

        score = mean_squared_error(y_valid, pred_i) ** .5
        val_scores.append(score)
        print(f'Fold {fold} RMSE: {score:.4f}')

        wandb.finish()

        pred = clf.predict(test)
        preds.append(pred)
    oof_df = pd.concat(oofs).reset_index(drop=True)
    all_score = mean_squared_error(oof_df["speedMps"], oof_df["oof"]) ** .5
    print('-' * 50)
    print('FINISHED | Whole RMSE: {:.4f}'.format(all_score))
    features = x_train.columns.values
    return oof_df, preds


def main():

    train_df = pd.read_csv(INPUT_DIR / 'baseline_locations_train.csv')
    test_df = pd.read_csv(INPUT_DIR / 'baseline_locations_test.csv')
    sub_df = pd.read_csv(INPUT_DIR / 'sample_submission.csv')

    # baselineに対する後処理で座標をtargetに近づける
    train_pp_df = apply_pp(train_df)
    test_pp_df = apply_pp(test_df)
    check_score(train_pp_df, is_return=False)

    # baselineでdegreeを計算する
    # trainにはtargetを付与
    train_deg_df = calc_degree(train_pp_df, phase='train')
    test_deg_df = calc_degree(test_pp_df, phase='test')  

    # imuデータが存在していないcollectionをdrop
    train_deg_df = drop_nan_imu_data(train_deg_df)
    test_deg_df = drop_nan_imu_data(test_deg_df)

    # get gt-degree
    logger.info('[START] get gt_degree')
    processes = multiprocessing.cpu_count()
    with multiprocessing.Pool(processes=processes) as pool:
        gr = train_deg_df.groupby(['collectionName','phoneName'])
        dfs = pool.imap_unordered(get_gt_degree2, gr)
        dfs = tqdm(dfs, total=len(gr))
        dfs = list(dfs)
    train_deg_df = pd.concat(dfs).sort_values(['collectionName', 'phoneName', 'millisSinceGpsEpoch']).reset_index(drop=True)
    assert TARGET in train_deg_df.columns
    logger.info("before learning RMSE:{}".format(mean_squared_error(train_deg_df[TARGET], train_deg_df["Degree"], squared=False)))

    # 特徴量作成
    logger.info('[START] create features')
    train_feat_df = create_features(train_deg_df, phase="train")
    test_feat_df = create_features(test_deg_df, phase="test")

    # preprocessing
    logger.info('[START] preprocessing')
    whole_df = pd.concat([train_feat_df, test_feat_df]).reset_index(drop=True)
    le_co = LabelEncoder()
    le_co.fit(whole_df['collectionName'])
    le_ph = LabelEncoder()
    le_ph.fit(whole_df['phoneName'])
    train = preprocess(train_feat_df, le1=le_co, le2=le_ph)
    test = preprocess(test_feat_df, le1=le_co, le2=le_ph)

    
    y = train[TARGET]
    X = train.drop([TARGET, "target_latDeg", "target_lngDeg", "phone"], axis=1)
    test = test.drop(["phone"], axis=1)
    assert X.shape[1] == test.shape[1]

    params = {
        # 目的関数. これの意味で最小となるようなパラメータを探します.
        'objective': 'rmse',
        'metric': 'rmse',

        # 学習率. 小さいほどなめらかな決定境界が作られて性能向上に繋がる場合が多いです、
        # がそれだけ木を作るため学習に時間がかかります
        'learning_rate': .05,

        # L2 Reguralization
        'reg_lambda': 1.,
        # こちらは L1
        'reg_alpha': .1,

        # 木の深さ. 深い木を許容するほどより複雑な交互作用を考慮するようになります
        'max_depth': 5,

        # 木の最大数. early_stopping という枠組みで木の数は制御されるようにしていますのでとても大きい値を指定しておきます.
        'n_estimators': 100000,

        # 木を作る際に考慮する特徴量の割合. 1以下を指定すると特徴をランダムに欠落させます。小さくすることで, まんべんなく特徴を使うという効果があります.
        'colsample_bytree': .5,

        # 最小分割でのデータ数. 小さいとより細かい粒度の分割方法を許容します.
        'min_child_samples': 10,

        # bagging の頻度と割合
        'subsample_freq': 3,
        'subsample': .9,

        # 特徴重要度計算のロジック(後述)
        'importance_type': 'gain',
        'random_state': 71,
    }

    oof_df, preds = fit_lgbm(X, y, test, params=params, verbose=2000)

    # plt.scatter(oof_df["speedMps"], oof_df["oof"], alpha=0.1)
    test["pred"] = np.mean(preds, axis=0)
    oof_df["collectionName"] = le_co.inverse_transform(oof_df["collectionName"])
    test["collectionName"] = le_co.inverse_transform(test["collectionName"])

    oof_df["phoneName"] = le_ph.inverse_transform(oof_df["phoneName"])
    test["phoneName"] = le_ph.inverse_transform(test["phoneName"])

    train_df = pd.read_csv(ROOT_DIR / "baseline_locations_train.csv")
    train_df = train_df.merge(oof_df[["collectionName", "phoneName", "millisSinceGpsEpoch", "pred"]], on=[
                              "collectionName", "phoneName", "millisSinceGpsEpoch"])

    test_df = pd.read_csv(ROOT_DIR / "baseline_locations_test.csv")
    test_df = test_df.merge(test[["collectionName", "phoneName", "millisSinceGpsEpoch", "pred"]], on=[
                            "collectionName", "phoneName", "millisSinceGpsEpoch"])

    train_df.to_csv(
        ROOT_DIR / "baseline_locations_train_with_direction.csv", index=False)
    test_df.to_csv(
        ROOT_DIR / "baseline_locations_test_with_direction.csv", index=False)


if __name__ == '__main__':
    main()
