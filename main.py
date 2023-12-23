import time, threading, can
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from scipy.spatial.distance import euclidean
from utils.system_communication import Radar
from utils.system_communication import AdasCANCommunication, ClusterCANCommunication

np.set_printoptions(precision=2)

forward_radar = Radar(serial_port='COM8', serial_baudrate=921600)
vehicle_can_ch = can.interface.Bus(bustype='pcan', channel='PCAN_USBBUS1', bitrate=500000)

adas_can_parser = AdasCANCommunication(dbc_filename='resource/ADAS_can_protocol.dbc')
clu_can_parser = ClusterCANCommunication(dbc_filename='resource/Evits_EV_CAN_DBC_CLU_ADAS.dbc')

radar_filter = DBSCAN(eps=0.05, min_samples=3, n_jobs=4)

forward_distance_prv = 0.0
forward_distance = 0.0
radar_points = []
clu_id_list = []
ttc_thr = 1.0


def radar_com():
    global forward_distance, forward_distance_prv, radar_points, clu_id_list

    while True:
        forward_radar.get_point_clouds()

        if forward_radar.read_n_point_clouds():
            radar_points = forward_radar.read_point_clouds()

            radar_filter.fit(radar_points[:, 0].reshape(-1, 1), radar_points[:, 1].reshape(-1, 1))
            clu_id_list = radar_filter.fit_predict(radar_points[:, 0].reshape(-1, 1), radar_points[:, ])
            n_radar_clu = np.max(clu_id_list)
            radar_cluster = []
            distance = []

            if n_radar_clu > 0:
                for clu_id in range(n_radar_clu + 1):
                    radar_cluster_row = []

                    for i, valid in enumerate(clu_id_list == clu_id):
                        if valid:
                            radar_cluster_row.append(radar_points[i, :])

                    radar_cluster.append(radar_cluster_row)

                for radar_set in radar_cluster:
                    x_pos = []
                    y_pos = []

                    for i, point in enumerate(radar_set):
                        if -1 <= point[0] <= 1:
                            x_pos.append(point[0])
                            y_pos.append(point[1])

                    if len(y_pos):
                        distance.append(np.mean(y_pos))

                if len(distance):
                    forward_distance = (np.min(distance)*0.5)+(forward_distance_prv*0.5)
                    forward_distance_prv = np.min(distance)


def vehicle_can_com():
    for can_msg in vehicle_can_ch:
        adas_can_parser.get_vehicle_status(packet=can_msg)


radar_parsing_task = threading.Thread(target=radar_com)
radar_parsing_task.daemon = True
radar_parsing_task.start()

vehicle_can_parsing_task = threading.Thread(target=vehicle_can_com)
vehicle_can_parsing_task.daemon = True
vehicle_can_parsing_task.start()

while True:
    time.sleep(0.02)

    if forward_radar.read_n_point_clouds():
        print(forward_distance, adas_can_parser.read_vehicle_speed())
