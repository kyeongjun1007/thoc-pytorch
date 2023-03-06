def get_param_list(dataset):
    n_hidden = [32, 64, 84]
    n_centroids = [[6,6,6], [12, 6, 1], [12, 6, 4], [18, 6, 1], [18, 12, 4], [18, 12, 6], [32, 12, 6]]
    max_loss_maintain = [3, 5]
    tau = [100, 1, 0.1, 0.5]
    batch_size = [32, 64, 128]
    skip_length = [[1, 2, 4], [1, 4, 8], [1, 4, 12], [1, 4, 16]]
    lambda_orth = [0.01, 0.1, 1, 10, 100] # 이게 문제 되는거 아님?

    lambda_tss = [0.01, 0.1, 1, 10, 100]

    if dataset in ['2d_gesture.csv', 'powerdemand.csv', 'kddcup99.csv']:
        lr = [0.01]
    else:
        lr = [0.001]

    param_list = [max_loss_maintain, n_hidden, n_centroids, tau, batch_size, skip_length,
                  lambda_orth, lambda_tss, lr]

    return param_list
