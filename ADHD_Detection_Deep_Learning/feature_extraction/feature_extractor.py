import numpy as np

def extract_features(activation_map, window_size=100):
    def check_nan(feature, feature_name):
        if np.isnan(feature):
            print(f"Error: {feature_name} contains NaN")
            print(f"Activation map: {activation_map}")
            raise ValueError(f"{feature_name} contains NaN")

    # 1. Global statistics features
    mean_activation = np.mean(activation_map)
    check_nan(mean_activation, 'mean_activation')

    max_activation = np.max(activation_map)
    check_nan(max_activation, 'max_activation')

    std_activation = np.std(activation_map)
    check_nan(std_activation, 'std_activation')

    # 2. Frequency domain features (using FFT)
    fft_activation = np.fft.fft(activation_map)
    fft_magnitude = np.abs(fft_activation)
    fft_power = np.square(fft_magnitude)

    fft_mean = np.mean(fft_magnitude)
    check_nan(fft_mean, 'fft_mean')

    fft_std = np.std(fft_magnitude)
    check_nan(fft_std, 'fft_std')

    fft_max = np.max(fft_magnitude)
    check_nan(fft_max, 'fft_max')

    fft_power_mean = np.mean(fft_power)
    check_nan(fft_power_mean, 'fft_power_mean')

    fft_power_max = np.max(fft_power)
    check_nan(fft_power_max, 'fft_power_max')

    # 3. Local statistics features (sliding window)
    num_windows = len(activation_map) // window_size
    local_means = []
    local_stds = []
    local_maxs = []

    for i in range(0, num_windows * window_size, window_size):
        local_window = activation_map[i:i + window_size]

        local_mean = np.mean(local_window)
        check_nan(local_mean, f'local_mean at window {i}')
        local_means.append(local_mean)

        local_std = np.std(local_window)
        check_nan(local_std, f'local_std at window {i}')
        local_stds.append(local_std)

        local_max = np.max(local_window)
        check_nan(local_max, f'local_max at window {i}')
        local_maxs.append(local_max)

    local_means = np.mean(local_means)
    check_nan(local_means, 'local_means')

    local_stds = np.mean(local_stds)
    check_nan(local_stds, 'local_stds')

    local_maxs = np.mean(local_maxs)
    check_nan(local_maxs, 'local_maxs')

    # 4. Concatenate all features
    features = np.hstack([mean_activation, max_activation, std_activation,
                          fft_mean, fft_std, fft_max, fft_power_mean, fft_power_max,
                          local_means, local_stds, local_maxs])

    return features
