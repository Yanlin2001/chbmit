import pyedflib
import numpy as np
from tqdm import tqdm
import mne
from scipy.signal import welch,stft
from scipy.stats import skew, kurtosis
from scipy.spatial.distance import euclidean
import pywt
from scipy.signal import welch

# @NOTE: 6个时域特征
def extract_basic_features(signal):
    signal = (signal - np.mean(signal)) / np.std(signal)
    mean = np.mean(signal)   #计算平均值
    std = np.std(signal)  #计算标准差
    sample_entropy = np.log(np.std(np.diff(signal)))  #计算样本熵
    fuzzy_entropy = -np.log(euclidean(signal[:-1], signal[1:]) / len(signal)) #计算模糊熵
    skewness = skew(signal)  #计算偏度
    kurt = kurtosis(signal)  #计算峰度
    return [mean, std, sample_entropy, fuzzy_entropy, skewness, kurt]

def extract_advanced_features(data, fs, window_length_sec=4):
    """
    使用短时傅里叶变换（STFT）从 EEG 数据中提取高级特征。

    :param data: EEG 信号数据。
    :param fs: 采样频率。
    :param window_length_sec: STFT的每个窗口长度（秒）。
    :return: 从 STFT 提取的特征。
    """

    # 执行 STFT
    f, t, Zxx = stft(data, fs, nperseg=window_length_sec*fs)
    
    # 从 STFT 提取特征
    # 在此，我们可以从 STFT 中提取各种特征。
    # 为简化起见，这里计算每个频率带的平均功率。
    # 可以通过计算 STFT 幅度的平方的平均值来完成。
    power = np.mean(np.abs(Zxx)**2, axis=1)  # 每个频率下的平均功率

    return power

# @NOTE: 从小波分解后的子带信号中提取特征
# @IDEA: 包括标准差（STD）、功率谱密度（PSD）、频带能量和模糊熵（FuzzyEn）
# @DATA: signal.shape = (1024,) = (256 * 4,)
def extract_wavelet_features(signal, sfreq):
    """
    对信号进行离散小波变换 (DWT)，并提取每个子带的特征。

    :param signal: EEG 信号。
    :param sfreq: 采样频率。
    :return: 小波分解后的特征。
    """
    # wavelet = 'db4'  # 使用
    # coeffs = pywt.wavedec(signal, wavelet, level=5)

    # @DATA: {依次折半} cD1.shape = (512,) cD2.shape = (256,) cD3.shape = (128,) cD4.shape = (64,) cD5.shape = (32,) cA5.shape = (16,)
    # D1-D5 是细节系数，A5 是近似系数
    (A1,D1) = pywt.dwt(signal, 'haar')
    (A2,D2) = pywt.dwt(A1, 'haar')
    (A3,D3) = pywt.dwt(A2, 'haar')
    (A4,D4) = pywt.dwt(A3, 'haar')
    (A5,D5) = pywt.dwt(A4, 'haar')
    cA5 , cD5, cD4, cD3, cD2, cD1 = A5,D5,D4,D3,D2,D1
    # cA5, cD5, cD4, cD3, cD2, cD1 = coeffs

    # 初始化一个空列表来存储子带特征
    wavelet_features = []

    # 对每个子带提取特征
    for subband in [cD1, cD2, cD3, cD4, cD5, cA5]:
        # 计算标准差
        std = np.std(subband)

        ''' # 计算功率谱密度 (PSD)
        # @DATA: freqs.shape = (129,) psd.shape = (129,) 129 = 256 / 2 + 1
        # @BUG: UserWarning: nperseg = 256 is greater than input length  = 192 , using nperseg = 192 
        # @BUG: sfreq = 256是调用函数的时候传入的，但是这里的subband长度是小波变换后的长度，不是原始信号的长度
        new_sfreq = subband.shape[0]
        freqs, psd = welch(subband, subband.shape[0])
        mean_psd = np.mean(psd)'''

        # 计算带能量
        band_energy = np.sum(np.square(subband))

        # 将特征添加到列表中
        wavelet_features.extend([std, band_energy])
        # @DATA: wavelet_features.shape = (2 * 6,) = (12,)
        # wavelet_features.extend([std, mean_psd, band_energy])

    return wavelet_features

def preprocess_and_extract_features_mne_with_timestamps(file_name):
    """
    使用 mne 库预处理 EEG 数据，并提取基础和高级特征。
    在每个特征数组的开头加入对应的时间戳。
    """
    # 将日志级别设置为 WARNING（只打印警告和错误）
    mne.set_log_level('CRITICAL')
    '''
    可选的日志级别
    'CRITICAL': 仅显示关键错误
    'ERROR': 仅显示错误
    'WARNING': 显示警告和错误
    'INFO': 显示常规信息（默认）
    'DEBUG': 显示调试信息
    '''
    # 加载数据
    raw = mne.io.read_raw_edf(file_name, preload=True)

    # 应用带通滤波器
    raw.filter(1., 50., fir_design='firwin')

    # 选择 EEG 通道
    raw.pick_types(meg=False, eeg=True, eog=False)

    # 定义短时间窗口的参数
    window_length = 4  # 窗口长度（秒）
    sfreq = raw.info['sfreq']  # 采样频率
    window_samples = int(window_length * sfreq)

    # 初始化一个空列表来存储特征和时间戳
    features_with_timestamps = []

    # 获取总窗口数以便在进度条中使用
    total_windows = len(raw.times) // window_samples

    # 使用tqdm包装range对象以显示进度条
    for start in tqdm(range(0, len(raw.times), window_samples), total=total_windows, desc="Processing windows"):
        end = start + window_samples
        if end > len(raw.times):
            break

        # 提取并预处理这个窗口中的数据
        window_data, times = raw[:, start:end]
        window_data = np.squeeze(window_data)

        # 获取窗口的开始时间戳
        timestamp = raw.times[start]
        # 定义需要提取特征的通道索引
        channel_indexes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]
        # 为每个通道的每个窗口提取基础和高级特征
        combined_channels_features = None

        for idx, channel_data in enumerate(window_data):
            if idx in channel_indexes:
                basic_features = extract_basic_features(channel_data)
                # advanced_features = extract_advanced_features(channel_data, sfreq)
                wavelet_features = extract_wavelet_features(channel_data, sfreq)
                combined_channels_features = np.concatenate([combined_channels_features, wavelet_features]) if combined_channels_features is not None else np.concatenate([wavelet_features])

        # 将特征与时间戳结合
        combined_features = np.concatenate([[timestamp], combined_channels_features])
        features_with_timestamps.append(combined_features)

    '''    # 用于存储拼接后的数组
        updated_features = []

        # 遍历 features_with_timestamps
        for i in range(len(features_with_timestamps)):
            current_array = features_with_timestamps[i]
            
            # 如果当前索引大于等于 15，拼接序号比自身小 15 的数组
            if i >= 15:
                # 取出序号比当前小 15 的数组
                to_concat = features_with_timestamps[i - 15]
                # 拼接数组
                # 数据作差
                # diff = current_array[1:] - to_concat[1:]
                current_array = np.concatenate((current_array, to_concat[1:]))
            
            # 只将进行拼接操作的数组添加到新的列表
            if i >= 15:  # 只保留进行拼接操作的数组
                updated_features.append(current_array)

        # 用拼接后的数组更新原始列表
        features_with_timestamps = updated_features'''

    return np.array(features_with_timestamps)
