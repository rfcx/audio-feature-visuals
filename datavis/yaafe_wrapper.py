import numpy as np
import yaafelib
from scipy.stats import median_absolute_deviation


class YaafeWrapper(object):
    def __init__(self, fs: int, config: dict):
        yaafe_config = {}
        for feature_name, feature_params in config.items():
            if feature_params['use']:
                specs = feature_name + ' ' + str(feature_params['params']).replace("'", '').replace(",", "").replace(": ", "=")[1:-1]
                yaafe_config[feature_name] = specs

        if yaafe_config:
            feature_plan = yaafelib.FeaturePlan(sample_rate=fs, normalize=True)
            for feature_name, setting in yaafe_config.items():
                feature_plan.addFeature(feature_name + ': ' + setting)
            data_flow = feature_plan.getDataFlow()
            self.engine = yaafelib.Engine()
            self.engine.load(data_flow)
        else:
            self.engine = None

    def compute_features(self, audio_data: np.ndarray) -> dict:
        features = self.engine.processAudio(audio_data.reshape(1, -1).astype('float64'))
        return features

    def compute_feature_stats(self, audio_data: np.ndarray) -> dict:
        features = self.engine.processAudio(audio_data.reshape(1, -1).astype('float64'))

        flat_dict = {}
        for name, values in features.items():
            if values.shape[1] == 1:
                flat_dict[f'{name}'] = values.mean()
            else:
                temporal_mean = values.mean(axis=0)
                q25, q50, q75 = np.quantile(temporal_mean, [0.25, 0.50, 0.75])
                flat_dict[name + '_q25'] = q25
                flat_dict[name + '_q50'] = q50
                flat_dict[name + '_q75'] = q75
                flat_dict[name + '_MAD'] = median_absolute_deviation(temporal_mean)
                flat_dict[name + '_IQR'] = q75 - q25

        return flat_dict
