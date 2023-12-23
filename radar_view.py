import time, threading, warnings
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from scipy.spatial.distance import euclidean
from utils.system_communication import Radar

np.set_printoptions(precision=2)

forward_radar = Radar(serial_port='COM8', serial_baudrate=921600)
radar_filter = DBSCAN(eps=0.05, min_samples=3, n_jobs=4)


def radar_com():
    while True:
        forward_radar.get_point_clouds()


radar_parsing_task = threading.Thread(target=radar_com)
radar_parsing_task.daemon = True
radar_parsing_task.start()


forward_distance = 0.0

while True:
    if forward_radar.read_n_point_clouds():
        radar_points = forward_radar.read_point_clouds()

        radar_filter.fit(radar_points[:, 0].reshape(-1, 1), radar_points[:, 1].reshape(-1, 1))
        clu_id_list = radar_filter.fit_predict(radar_points[:, 0].reshape(-1, 1), radar_points[:, ])
        n_radar_clu = np.max(clu_id_list)
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
                    if -1 <= point[0] <= 1:
                        x_pos.append(point[0])
                        y_pos.append(point[1])

                if len(y_pos):
                    distance.append(np.mean(y_pos))

            if len(distance):
                forward_distance = np.min(distance)

        plt.clf()
        plt.scatter(x=radar_points[:, 0], y=radar_points[:, 1], c=clu_id_list)
        plt.xlim(-3, 3)
        plt.ylim(0, 10)
        plt.pause(0.05)

    print(forward_distance)
    time.sleep(0.02)