import numpy as np
import pandas as pd
import yaafelib


YAAFE_FEATURES = {
    'Chroma': 'Chroma',
    'LPC': 'Linear Predictor Coefficients',
    'LSF': 'Line Spectral Frequency',
    'MFCC': 'Mel-frequency cepstral coefficients',
    'OBSI': 'Octave band signal intensity with triangular filter',
    'PerceptualSharpness': 'Perceptual sharpness',
    'PerceptualSpread': 'Perceptual spread',
    'SpectralCrestFactorPerBand': 'Spectral crest factors',
    'SpectralIrregularity': 'Spectral irregularity',
    'SpectralDecrease': 'Spectral decrease',
    'SpectralFlatness': 'Spectral flatness',
    'SpectralFlux': 'Spectral flux',
    'SpectralRolloff': 'Spectral rolloff',
    'SpectralVariation': 'Spectral variation',
    'SpectralSlope': 'Spectral slope',
    'ZCR': 'Zero-crossing rate'
}

class YaafeWrapper3(object):

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

    def get_features(self, audio_data: np.ndarray) -> dict:
        features = self.engine.processAudio(audio_data.reshape(1, -1).astype('float64'))
        return features


class YaafeWrapper2(object):

    def __init__(self, fs: int, config: dict):

        yaafe_config = {}

        chroma = config['Features']['Chroma']
        LPC = config['Features']['LPC']
        LSF = config['Features']['LSF']
        MFCC = config['Features']['MFCC']
        OBSIR = config['Features']['OBSIR']
        perceptual_sharpness = config['Features']['PerceptualSharpness']
        perceptual_spread = config['Features']['PerceptualSpread']
        spectral_crest = config['Features']['SpectralCrestFactorPerBand']
        spectral_irregularity = config['Features']['SpectralIrregularity']
        spectral_decrease = config['Features']['SpectralDecrease']

        spectral_flatness = config['Features']['SpectralFlatness']
        spectral_flux = config['Features']['SpectralFlux']
        spectral_rolloff = config['Features']['SpectralRolloff']
        spectral_variation = config['Features']['SpectralVariation']
        spectral_slope = config['Features']['SpectralSlope']
        ZCR = config['Features']['ZCR']

        if chroma['use']:
            chroma = chroma["params"]
            yaafe_config['Chroma'] = f'Chroma2 ' \
                                     f'CQTAlign={chroma["CQTAlign"]} ' \
                                     f'CQTBinsPerOctave={chroma["CQTBinsPerOctave"]} ' \
                                     f'CQTMinFreq={chroma["CQTMinFreq"]} ' \
                                     f'CQTNbOctaves={chroma["CQTNbOctaves"]}  ' \
                                     f'CZBinsPerSemitone={chroma["CZBinsPerSemitone"]} ' \
                                     f'CZNbCQTBinsAggregatedToPCPBin={chroma["CZNbCQTBinsAggregatedToPCPBin"]} ' \
                                     f'CZTuning={chroma["CZTuning"]}  ' \
                                     f'stepSize={chroma["step_size"]}'

        if LPC['use']:
            LPC = LPC["params"]
            yaafe_config['LPC'] = f'LPC LPCNbCoeffs={LPC["LPCNbCoeffs"]} blockSize={LPC["block_size"]} stepSize={LPC["step_size"]}'

        if LSF['use']:
            LSF = LSF["params"]
            yaafe_config['LSF'] = f'LSF blockSize={LSF["block_size"]} stepSize={LSF["step_size"]}'

        if MFCC['use']:
            MFCC = MFCC["params"]
            yaafe_config['MFCC'] = f'MFCC ' \
                                   f'CepsIgnoreFirstCoeff={MFCC["CepsIgnoreFirstCoeff"]} ' \
                                   f'CepsNbCoeffs={MFCC["CepsNbCoeffs"]} ' \
                                   f'MelMaxFreq={MFCC["MelMaxFreq"]} ' \
                                   f'MelMinFreq={MFCC["MelMinFreq"]}  ' \
                                   f'MelNbFilters={MFCC["MelNbFilters"]} ' \
                                   f'blockSize={MFCC["block_size"]} ' \
                                   f'stepSize={MFCC["step_size"]} '

        if OBSIR['use']:
            OBSIR = OBSIR["params"]
            yaafe_config['OBSIR'] = f'OBSIR OBSIMinFreq={OBSIR["OBSIMinFreq"]} blockSize={OBSIR["block_size"]} ' \
                                    f'stepSize={OBSIR["step_size"]}'

        if perceptual_sharpness['use']:
            perceptual_sharpness = perceptual_sharpness["params"]
            yaafe_config['PerceptualSharpness'] = f'PerceptualSharpness ' \
                                                  f'blockSize={perceptual_sharpness["block_size"]} ' \
                                                  f'stepSize={perceptual_sharpness["step_size"]}'

        if perceptual_spread['use']:
            perceptual_spread = perceptual_spread["params"]
            yaafe_config['PerceptualSpread'] = f'PerceptualSpread ' \
                                               f'blockSize={perceptual_spread["block_size"]} ' \
                                               f'stepSize={perceptual_spread["step_size"]}'

        if spectral_crest['use']:
            spectral_crest = spectral_crest["params"]
            yaafe_config['SpectralCrestFactorPerBand'] = f'SpectralCrestFactorPerBand ' \
                                                         f'blockSize={spectral_crest["block_size"]} ' \
                                                         f'stepSize={spectral_crest["step_size"]}'

        if spectral_irregularity['use']: # f'SpectralIrregularity CQTBinsPerOctave=36  CQTMinFreq=73.42  CQTNbOctaves=3  stepSize={step_size}',
            spectral_irregularity = spectral_irregularity["params"]
            yaafe_config['SpectralIrregularity'] = f'SpectralIrregularity ' \
                                                   f'CQTBinsPerOctave={spectral_irregularity["CQTBinsPerOctave"]} ' \
                                                   f'CQTMinFreq={spectral_irregularity["CQTMinFreq"]} ' \
                                                   f'CQTNbOctaves={spectral_irregularity["CQTNbOctaves"]} ' \
                                                   f'stepSize={spectral_irregularity["step_size"]}'

        if spectral_decrease['use']:
            spectral_decrease = spectral_decrease["params"]
            yaafe_config['PerceptualSpread'] = f'PerceptualSpread ' \
                                               f'blockSize={spectral_decrease["block_size"]} ' \
                                               f'stepSize={spectral_decrease["step_size"]}'

        if spectral_flatness['use']:
            spectral_flatness = spectral_flatness["params"]
            yaafe_config['SpectralFlatness'] = f'SpectralFlatness ' \
                                               f'blockSize={spectral_flatness["block_size"]} ' \
                                               f'stepSize={spectral_flatness["step_size"]}'

        if spectral_flux['use']:
            spectral_flux = spectral_flux["params"]
            yaafe_config['SpectralFlux'] = f'SpectralFlux ' \
                                           f'blockSize={spectral_flux["block_size"]} ' \
                                           f'stepSize={spectral_flux["step_size"]}'

        if spectral_rolloff['use']:
            spectral_rolloff = spectral_rolloff["params"]
            yaafe_config['SpectralRolloff'] = f'SpectralRolloff ' \
                                              f'blockSize={spectral_rolloff["block_size"]} ' \
                                              f'stepSize={spectral_rolloff["step_size"]}'

        if spectral_variation['use']:
            spectral_variation = spectral_variation["params"]
            yaafe_config['SpectralVariation'] = f'SpectralVariation ' \
                                                f'blockSize={spectral_variation["block_size"]} ' \
                                                f'stepSize={spectral_variation["step_size"]}'

        if spectral_slope['use']:
            spectral_slope = spectral_slope["params"]
            yaafe_config['SpectralSlope'] = f'SpectralSlope ' \
                                            f'blockSize={spectral_slope["block_size"]} ' \
                                            f'stepSize={spectral_slope["step_size"]}'

        if ZCR['use']:
            ZCR = ZCR["params"]
            yaafe_config['ZCR'] = f'ZCR ' \
                                  f'blockSize={ZCR["block_size"]} ' \
                                  f'stepSize={ZCR["step_size"]}'

        if yaafe_config:
            feature_plan = yaafelib.FeaturePlan(sample_rate=fs, normalize=True)
            for feature_name, setting in yaafe_config.items():
                feature_plan.addFeature(feature_name + ': ' + setting)
            data_flow = feature_plan.getDataFlow()
            self.engine = yaafelib.Engine()
            self.engine.load(data_flow)
        else:
            self.engine = None

    def get_features(self, audio_data: np.ndarray) -> dict:
        features = self.engine.processAudio(audio_data.reshape(1, -1).astype('float64'))
        return features



class YaafeWrapper(object):

    def __init__(self, fs: int, block_size=1024, step_size=None, selected_features='all'):
        if not step_size:
            step_size = block_size // 2

        features_config = {
            'Chroma': f'Chroma2 CQTAlign=c  CQTBinsPerOctave=48  CQTMinFreq=27.5  CQTNbOctaves=7  CZBinsPerSemitone=1  CZNbCQTBinsAggregatedToPCPBin=-1  CZTuning=440  stepSize={step_size}',
            'LPC': f'LPC LPCNbCoeffs=1  blockSize={block_size}  stepSize={step_size}',
            'LSF': f'LSF blockSize={block_size}  stepSize={step_size}',
            'MFCC': f'MFCC CepsIgnoreFirstCoeff=1  CepsNbCoeffs=13  FFTWindow=Hanning  MelMaxFreq=6000.0  MelMinFreq=400.0  MelNbFilters=40  blockSize={block_size}  stepSize={step_size}',
            'OBSI': f'OBSI FFTLength=0  FFTWindow=Hanning  OBSIMinFreq=27.5  blockSize={block_size}  stepSize={step_size}',
            'PerceptualSharpness': f'PerceptualSharpness blockSize={block_size}  stepSize={step_size}',
            'PerceptualSpread': f'PerceptualSpread blockSize={block_size}  stepSize={step_size}',
            'SpectralCrestFactorPerBand': f'SpectralCrestFactorPerBand FFTLength=0  FFTWindow=Hanning  blockSize={block_size}  stepSize={step_size}',
            'SpectralIrregularity': f'SpectralIrregularity CQTBinsPerOctave=36  CQTMinFreq=73.42  CQTNbOctaves=3  stepSize={step_size}',
            'SpectralDecrease': f'SpectralDecrease FFTLength=0  FFTWindow=Hanning  blockSize={block_size}  stepSize={step_size}',
            'SpectralFlatness': f'SpectralFlatness FFTLength=0  FFTWindow=Hanning  blockSize={block_size}  stepSize={step_size}',
            'SpectralFlux': f'SpectralFlux FFTLength=0  FFTWindow=Hanning  FluxSupport=All  blockSize={block_size}  stepSize={step_size}',
            'SpectralRolloff': f'SpectralRolloff FFTLength=0  FFTWindow=Hanning  blockSize={block_size}  stepSize={step_size}',
            'SpectralVariation': f'SpectralVariation FFTLength=0  FFTWindow=Hanning  blockSize={block_size}  stepSize={step_size}',
            'SpectralSlope': f'SpectralSlope blockSize={block_size}  stepSize={step_size}',
            'ZCR': f'ZCR blockSize={block_size}  stepSize={step_size}'
        }

        self.fs = fs
        if selected_features == 'all':
            selected_features = features_config.keys()
        feature_plan = yaafelib.FeaturePlan(sample_rate=fs, normalize=True)
        for feature_name, setting in features_config.items():
            if feature_name in selected_features:
                feature_plan.addFeature(feature_name + ': ' + setting)
        data_flow = feature_plan.getDataFlow()
        self.engine = yaafelib.Engine()
        self.engine.load(data_flow)

    def get_features(self, audio_data: np.ndarray) -> dict:
        features = self.engine.processAudio(audio_data.reshape(1, -1).astype('float64'))
        return features

    def get_features_as_df(self, audio_data: np.ndarray):
        features = self.engine.processAudio(audio_data.reshape(1, -1).astype('float64'))
        dfs = []
        for name, array in features.items():
            if array.shape[1] == 1:
                df = pd.DataFrame(array, columns=[name])
            else:
                column_names = [f'{name}.{idx}' for idx in range(array.shape[1])]
                df = pd.DataFrame(array, columns=column_names)
            dfs.append(df)
        dfs = pd.concat(dfs, axis=1)
        return dfs

    def features_to_hdf5(self, audio_data: np.ndarray, filename) -> None:
        features = self.engine.processAudio(audio_data.reshape(1, -1).astype('float64'))
        for name, array in features.items():
            out = filename + '.hdf5'
            pd.DataFrame(array).to_hdf(out, key=name)

    def get_mean_features(self, audio_data: np.ndarray) -> dict:
        features = self.engine.processAudio(audio_data.reshape(1, -1).astype('float64'))

        flat_dict = {}
        for name, values in features.items():
            if values.shape[1] == 1:
                flat_dict[f'{name}'] = values.mean()
            else:
                d = {f'{name}.{idx}': value for idx, value in enumerate(list(values.mean(axis=0)))}
                flat_dict.update(d)
        return flat_dict

    def get_mean_features_as_series(self, audio_data: np.ndarray) -> pd.Series:
        flat_dict = self.get_mean_features(audio_data)
        return pd.Series(flat_dict)

