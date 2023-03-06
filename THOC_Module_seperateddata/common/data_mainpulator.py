import random
from pathlib import Path

import numpy as np
import torch


def generate_sub_route_dist(batch_size, num_nodes, max_visit_nodes):
    K = random.randint(num_nodes // 4, num_nodes // 2)
    # number of to visit nodes
    distance = []

    while K > 0:
        to_include_num = random.randint(1, min(K, max_visit_nodes - 2))
        # print(f"K:{K}, to_incl: {to_include_num}")
        dist = torch.randint(50 * 2, 350 * 2, (batch_size, 1)).type(torch.float32)
        distance.append(dist)
        K = K - to_include_num

    result = torch.concat(distance, dim=-1)
    return result


def create_distance_lst(batch_size, T, num_nodes, max_visit_nodes, t_in_day=2):
    dist_lst = []
    day_change_val = []
    temp_sum = 0

    for t in range(T):
        dist_t = generate_sub_route_dist(batch_size, num_nodes, max_visit_nodes)
        n_sub_route = dist_t.shape[1]
        temp_sum += n_sub_route

        for i in range(n_sub_route):
            d = np.expand_dims(dist_t[:, i], 1)
            dist_lst.append(d)

        if t % 2 == 1:
            day_change_val.append(temp_sum)

    return dist_lst, day_change_val


if __name__ == '__main__':
    batch_size = 500
    T = 30 * 2
    num_nodes = 50
    max_visit_nodes = 4
    num_vehicles = 17
    threshold = 0.9

    folder_name = f"../../data/synthetic_{T}_{num_vehicles}/"
    file_name = f"distance.data"
    path = Path(folder_name)

    if not path.exists():
        path.mkdir(parents=True)

    dist_lst, day_change_val = create_distance_lst(batch_size, T, num_nodes, max_visit_nodes)
    together = {"dist_lst": dist_lst, "day_change_val": day_change_val}

    torch.save(together, folder_name+file_name)