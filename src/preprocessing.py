import numpy as np
from spafe.features.spfeats import extract_feats
import matplotlib.pyplot as plt
import numpy as np
from scipy.fftpack import rfft
import scipy.stats
from scipy.signal import stft as scipy_stft

import librosa
from tqdm import tqdm
import pandas as pd
import sklearn
import sklearn.preprocessing
from pathlib import Path

from spafe.features.mfcc import mfcc, imfcc
from spafe.features.bfcc import bfcc
from spafe.features.lfcc import lfcc
from spafe.features.lpc import lpc, lpcc
from spafe.features.msrcc import msrcc
from spafe.features.ngcc import ngcc
from spafe.features.psrcc import psrcc
from spafe.features.rplp import plp, rplp
from spafe.features.gfcc import gfcc


#?뚯꽦 ?곗씠???뱀쭠?ㅼ쓣 媛곴컖 遺꾨쪟???볦? 寃?
SP_FEATS_NAMES = [
    'duration','spectrum', 'mean_frequency', 'peak_frequency', 'frequencies_std', 'amplitudes_cum_sum', 'mode_frequency', 'median_frequency', 'frequencies_q25', 'frequencies_q75',
    'iqr', 'freqs_skewness', 'freqs_kurtosis', 'spectral_entropy', 'spectral_flatness', 'spectral_centroid', 'spectral_bandwidth', 'spectral_spread', 'spectral_rolloff', 'energy',
    'rms', 'zcr', 'spectral_mean', 'spectral_rms', 'spectral_std', 'meanfun', 'minfun', 'maxfun', 'meandom', 'mindom', 'maxdom', 'dfrange', 'modindex'
]

SP = ['spectral_centroid', 'spectral_skewness', 'spectral_kurtosis', 'spectral_entropy', 'spectral_spread', 'spectral_flatness', 'spectral_rolloff',
      'spectral_flux', 'spectral_mean', 'spectral_rms', 'spectral_std', 'spectral_variance']


#constants.py ?댁슜
MEDIA_INFO_FEATURES = ['bit_rate']
SPECTRUM_FEATURES = ['mfcc', 'bfcc', 'lfcc', 'lpc', 'lpcc', 'msrcc', 'ngcc', 'psrcc', 'plp', 'rplp', 'gfcc'] #?뚯꽦 ?ㅽ럺?몃읆 ?뱀쭠 異붿텧
DROP_FEATURES = ["label","duration", "size", "spectral_bandwidth"]
SPECTRAL_COMPLEX_VALUES = ['spectral_flatness', 'spectral_centroid', 'spectral_spread'] #蹂듭냼??媛믪쓣 媛吏???뚯꽦 ?뱀쭠
TWO_DEMENSION_FEATURES = ['spectrum', 'amplitudes_cum_sum', 'energy']

SPECTRUM_FEATURES_FUNCTIONS = [mfcc, bfcc, lfcc, lpc, lpcc, msrcc, ngcc, psrcc, plp, rplp, gfcc] #?뚯꽦 ?ㅽ럺?몃읆 ?뱀쭠 異붿텧???⑥닔



import numpy as np
from scipy.fftpack import rfft
import scipy.stats
from scipy.signal import stft as scipy_stft
import scipy.signal
import librosa

# STFT ?⑥닔 ?뺤쓽
def stft(sig, fs, nperseg=256, noverlap=None):
    f, t, Zxx = scipy_stft(sig, fs, nperseg=nperseg, noverlap=noverlap)
    return f, t, Zxx

# RFFT ?⑥닔 ?뺤쓽
def rfft(sig, n=None):
    return np.fft.rfft(sig, n=n)

# Fundamental Frequencies Extractor ?대옒???뺤쓽
class FundamentalFrequenciesExtractor:
    def __init__(self, fs):
        self.fs = fs

    def compute_fund_freqs(self, sig):
        fourrier_transform = np.fft.fft(sig)
        magnitude_spectrum = np.abs(fourrier_transform)
        frequencies = np.fft.fftfreq(len(magnitude_spectrum), 1/self.fs)
        fundamental_freq = frequencies[np.argmax(magnitude_spectrum)]
        return fundamental_freq

# Dominant Frequencies ?⑥닔 ?뺤쓽
def get_dominant_frequencies(sig, fs, lower_cutoff=50, upper_cutoff=3000):
    nyquist = fs / 2
    low = lower_cutoff / nyquist
    high = upper_cutoff / nyquist
    b, a = scipy.signal.butter(1, [low, high], btype='band')
    filtered_sig = scipy.signal.lfilter(b, a, sig)
    dominant_freqs = np.fft.fftfreq(len(filtered_sig), 1/fs)
    return dominant_freqs[np.argmax(np.abs(np.fft.fft(filtered_sig)))]

def compute_fund_freqs(sig, fs):
    # fundamental frequencies calculations
    fund_freqs_extractor = FundamentalFrequenciesExtractor(fs)
    fundamental_freq = fund_freqs_extractor.compute_fund_freqs(sig)
    return np.array([fundamental_freq])


def extract_frequency_feats(sig, fs, nfft=512):
    feats = {}

    fourrier_transform = rfft(sig, nfft)
    magnitude_spectrum = (1/nfft) * np.abs(fourrier_transform)
    power_spectrum = (1/nfft)**2 * magnitude_spectrum**2

    frequencies = np.fft.fftfreq(nfft, 1 / fs)
    positive_freqs = frequencies[:nfft//2]
    magnitude_spectrum = magnitude_spectrum[:len(positive_freqs)]
    power_spectrum = power_spectrum[:len(positive_freqs)]

    spectrum = power_spectrum
    amplitudes = power_spectrum
    amp_cumsum = np.cumsum(amplitudes)

    feats["duration"] = len(sig) / float(fs)
    feats["spectrum"] = spectrum   #2李⑥썝

    feats["mean_frequency"] = positive_freqs.sum()
    feats["peak_frequency"] = positive_freqs[np.argmax(amplitudes)]
    feats["frequencies_std"] = positive_freqs.std()
    feats["amplitudes_cum_sum"] = amp_cumsum   #2李⑥썝
    feats["mode_frequency"] = positive_freqs[amplitudes.argmax()]
    feats["median_frequency"] = np.median(positive_freqs)
    feats["frequencies_q25"] = positive_freqs[np.searchsorted(amp_cumsum, 0.25 * amp_cumsum[-1])]
    feats["frequencies_q75"] = positive_freqs[np.searchsorted(amp_cumsum, 0.75 * amp_cumsum[-1])]
    feats["iqr"] = feats["frequencies_q75"] - feats["frequencies_q25"]

    feats["freqs_skewness"] = scipy.stats.skew(positive_freqs)
    feats["freqs_kurtosis"] = scipy.stats.kurtosis(positive_freqs)

    feats["energy"] = magnitude_spectrum     #2李⑥썝

    feats["rms"] = np.sqrt(np.mean(sig**2))

    feats["zcr"] = ((sig[:-1] * sig[1:]) < 0).sum() / len(sig)

    fund_freqs = compute_fund_freqs(sig, fs)
    feats["meanfun"] = fund_freqs.mean()
    feats["minfun"] = fund_freqs.min()
    feats["maxfun"] = fund_freqs.max()

    dom_freqs = get_dominant_frequencies(sig, fs)
    feats["meandom"] = dom_freqs
    feats["mindom"] = dom_freqs
    feats["maxdom"] = dom_freqs
    feats["dfrange"] = feats["maxdom"] - feats["mindom"]
    feats["modindex"] = 0  # Placeholder, replace with appropriate calculation if available

    return feats



# ?ㅻ뵒???뚯씪?먯꽌 ?뚯꽦?뱀쭠 異붿텧
def extract_sp_feats(file_path: str, dtype: str = "float64") -> dict:
    y, sr = librosa.load(file_path, sr=44100)  # librosa濡??ㅻ뵒???뚯씪 濡쒕뱶

    sp_feats = extract_feats(sig=y, fs=sr)  # extract濡??뚯꽦?뚯씪???뱀쭠??異붿텧?댁꽌 ?뺤뀛?덈━濡?諛섑솚
    for sp_feat_name in sp_feats:
        sp_feat_value = sp_feats[sp_feat_name]
        if isinstance(sp_feat_value, (tuple, np.ndarray, list)):
            sp_feat_value = np.array(sp_feat_value)
            sp_feats[sp_feat_name] = sp_feat_value.mean() if sp_feat_value.size > 0 else 0
        elif sp_feat_name in SPECTRAL_COMPLEX_VALUES:
            sp_feats[sp_feat_name] = np.array(sp_feat_value).real.mean()

    sp_fre_feats = extract_frequency_feats(sig=y, fs=sr)
    for sp_feat_name, sp_feat_value in sp_fre_feats.items():
        sp_feat_value = np.array(sp_feat_value)
        if sp_feat_name not in sp_feats:
            sp_feats[sp_feat_name] = sp_feat_value.mean() if sp_feat_value.size > 0 else 0
        else:
            if sp_feat_name in SPECTRAL_COMPLEX_VALUES:
                sp_feats[sp_feat_name] = sp_feat_value.real.mean()
            else:
                sp_feats[sp_feat_name] = sp_feat_value.mean() if sp_feat_value.size > 0 else 0

    return sp_feats


def extract_spectrum_data(sample: str) -> dict:
    y, sr = librosa.load(sample, sr=44100)

    spectrum_dict = {}
    spectrum_dict["signal"] = y.mean()

    for idx in range(len(SPECTRUM_FEATURES)):
        try:
            feature = SPECTRUM_FEATURES_FUNCTIONS[idx](sig=y, fs=sr)

            if idx == 3:
                lpc_1 = np.array(feature[0])
                lpc_2 = np.array(feature[1])
                feature = (lpc_1.mean() + lpc_2.mean()) / 2
            else:
                if isinstance(feature, (np.ndarray, list)):
                    feature = np.array(feature)
                    if feature.ndim == 2:
                        feature = feature.mean()
                    else:
                        feature = feature.mean()
                else:
                    feature = feature

            # NaN ?먮뒗 臾댄븳? 媛?泥섎━
            if np.isnan(feature) or np.isinf(feature):
                feature = 0.0

            spectrum_dict[SPECTRUM_FEATURES[idx]] = feature

        except np.linalg.LinAlgError:
            spectrum_dict[SPECTRUM_FEATURES[idx]] = 0.0  # 湲곕낯媛?

    return spectrum_dict

def filter_features(features: dict) -> list[float]:
    filtered_features = {}
    for f in features:
        if f not in DROP_FEATURES:
            value = features[f]
            # NaN ?먮뒗 臾댄븳? 媛?泥섎━
            if np.isnan(value) or np.isinf(value):
                value = 0.0
            filtered_features[f] = value

    return filtered_features


def get_all_features_from_sample(file_path: str) -> list[float]:
    sp_feats = extract_sp_feats(file_path)
    spectrum_data = extract_spectrum_data(file_path)
    features = sp_feats | spectrum_data
    return filter_features(features)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"


def resolve_audio_path(file_path):
    path = Path(str(file_path).strip())
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def main(input_csv=DATA_DIR / "features" / "test_sr_44100.csv", output_csv=OUTPUT_DIR / "test_sr_44100_features.csv"):
    df = pd.read_csv(input_csv)
    OUTPUT_DIR.mkdir(exist_ok=True)

    if output_csv.exists():
        features_df = pd.read_csv(output_csv)
    else:
        features_df = pd.DataFrame()

    processed_ids = set(features_df["id"].values) if "id" in features_df.columns else set()

    for _, row in tqdm(df.iterrows(), total=df.shape[0]):
        sample_id = row.get("id")
        if sample_id in processed_ids:
            continue

        try:
            audio_path = resolve_audio_path(row["path"])
            y, _ = librosa.load(audio_path, sr=16000)

            if np.any(np.isnan(y)) or np.any(np.isinf(y)):
                feature = {f"feature_{i + 1}": 0.0 for i in range(47)}
            else:
                feature = get_all_features_from_sample(str(audio_path))

            new_row = {"id": sample_id}
            new_row.update(feature)
        except Exception:
            new_row = {"id": sample_id}
            new_row.update({f"feature_{i + 1}": 0.0 for i in range(47)})

        new_df = pd.DataFrame([new_row])
        header = not output_csv.exists()
        new_df.to_csv(output_csv, mode="a", header=header, index=False)


if __name__ == "__main__":
    main()
