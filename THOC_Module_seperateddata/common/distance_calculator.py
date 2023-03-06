import torch


class DistUtil(object):
    def __init__(self, input_type):
        self.input_type = input_type
        self.depot_node_xy = None
        self.graph = None

    def set_data(self, data):
        if self.input_type == "graph":
            self.graph = data

        elif self.input_type == "2d":
            self.depot_node_xy = data

        else:
            raise NotImplementedError

    def get_dist(self, seq_nodes):
        if self.input_type == 'graph':
            batch_size = seq_nodes.size(0)

            _from = seq_nodes
            _to = seq_nodes.roll(-1)

            batch_idx = torch.arange(batch_size).unsqueeze(-1).unsqueeze(-1)
            segments = self.graph[batch_idx, _from, _to]
            travel_distances = segments.sum(dim=-1).to(torch.float32)
            return travel_distances

        else:
            num_subroutes = seq_nodes.size(1)

            gathering_index = seq_nodes[:, :, :, None].expand(-1, -1, -1, 2)
            # shape: (batch, pomo, selected_list_length, 2)
            all_xy = self.depot_node_xy[:, None, :, :].expand(-1, num_subroutes, -1, -1)
            # shape: (batch, pomo, problem+1, 2)

            ordered_seq = all_xy.gather(dim=2, index=gathering_index)
            # shape: (batch, selected_list_length, 2)

            rolled_seq = ordered_seq.roll(dims=2, shifts=-1)
            segment_lengths = ((ordered_seq - rolled_seq) ** 2).sum(3).sqrt()
            # shape: (batch, selected_list_length)

            travel_distances = segment_lengths.sum(2)
            return travel_distances
