import time, threading, can
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from utils.system_communication import Radar
from utils.system_communication import AdasCANCommunication, ClusterCANCommunication


np.set_printoptions(precision=2)

vehicle_can_ch = can.interface.Bus(bustype='pcan', channel='PCAN_USBBUS1', bitrate=500000)

adas_can_parser = AdasCANCommunication(dbc_filename='resource/ADAS_can_protocol.dbc')
clu_can_parser = ClusterCANCommunication(dbc_filename='resource/Evits_EV_CAN_DBC_CLU_FCWS.dbc')

forward_radar = Radar(serial_port='COM3', serial_baudrate=406800)
radar_filter = DBSCAN(eps=0.005, min_samples=3, n_jobs=4)

forward_distance = 0.0
vehicle_speed = 0.0
fcw_state = 0
ttc_val = 0.0
msg_list = []


def vehicle_com():
    global vehicle_speed

    for packet in vehicle_can_ch:
        vehicle_speed = adas_can_parser.get_vehicle_status(packet=packet)


def radar_com():
    while True:
        forward_radar.get_point_clouds()


radar_parsing_task = threading.Thread(target=radar_com)
radar_parsing_task.daemon = True
radar_parsing_task.start()

vehicle_can_parsing_task = threading.Thread(target=vehicle_com)
vehicle_can_parsing_task.daemon = True
vehicle_can_parsing_task.start()


while True:
    if forward_radar.read_n_point_clouds():
        radar_points = forward_radar.read_point_clouds()

        radar_filter.fit(radar_points[:, 0].reshape(-1, 1), radar_points[:, 1].reshape(-1, 1))
        clu_id_list = radar_filter.fit_predict(radar_points[:, 0].reshape(-1, 1), radar_points[:, ])
        n_radar_clu = np.max(clu_id_list)+1
        radar_cluster = []
        distance = []

        if n_radar_clu > 0:
            for clu_id in range(n_radar_clu+1):
                radar_cluster_row = []

                for i, valid in enumerate(clu_id_list == clu_id):
                    if valid:
                        radar_cluster_row.append(radar_points[i, :])

                radar_cluster.append(radar_cluster_row)

            for radar_set in radar_cluster:
                x_pos = []
                y_pos = []

                for i, point in enumerate(radar_set):
                    if -0.5 <= point[0] <= 0.5:
                        x_pos.append(point[0])
                        y_pos.append(point[1])

                if len(y_pos):
                    distance.append(np.mean(y_pos))

            if len(distance):
                forward_distance = np.min(distance)

        ttc_val = forward_distance / (vehicle_speed*1000/3600)

        if ttc_val > 10:
            ttc_val = 10
            fcw_state = 0

        if ttc_val < 0:
            ttc_val = 0

        if 0 < ttc_val <= 2:
            fcw_state = 1

        #plt.clf()
        #plt.scatter(x=radar_points[:, 0], y=radar_points[:, 1], c=clu_id_list)
        #plt.xlim(-3, 3)
        #plt.ylim(0, 20)
        #plt.pause(0.01)

    msg_list.append(adas_can_parser.create_fcw_can_msg(fcw_state=fcw_state+1, object_type=2, ttc=ttc_val))
    msg_list.append(clu_can_parser.create_fcw_can_msg(fcw_state=fcw_state))

    for msg in msg_list:
        vehicle_can_ch.send(msg)
        vehicle_can_ch.flush_tx_buffer()

    msg_list.clear()
