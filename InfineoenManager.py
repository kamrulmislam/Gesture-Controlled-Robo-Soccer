import pprint
import numpy as np
from ifxradarsdk import get_version_full
from ifxradarsdk.fmcw import DeviceFmcw
from ifxradarsdk.fmcw.types import create_dict_from_sequence,  FmcwSimpleSequenceConfig, FmcwSequenceChirp, FmcwMetrics


class InfineonManager():
    def __init__(self):
        print(get_version_full())
        print(DeviceFmcw.get_list())
        self.device = None
        self.config = None
    
    
    def init_device_fmcw(self, cfg_seq, cfg_chirp):
        if self.device:  # close existing device
            self.close()

        self.device = DeviceFmcw()
        self.cfg_seq = cfg_seq
        self.cfg_chirp = cfg_chirp
        if self.config is None:
            self.config = self.get_config()

        # self.params = self.get_params({**self.cfg_chirp, **self.cfg_seq})

        # sequence = self.device.get_acquisition_sequence()
        # sequence.loop.repetition_time_s = 0.1                 # change frame period to 100ms
        # sequence = self.device.set_acquisition_sequence(sequence)  # apply the new settings
        
        # config = FmcwSimpleSequenceConfig(**cfg_seq,
                                        #    chirp = FmcwSequenceChirp(**cfg_chirp))
                                            
        sequence = self.device.create_simple_sequence(self.config)
        self.device.set_acquisition_sequence(sequence)
        # print('Radar device initiated successfully.')

        
    def fetch_n_frames(self, num_frames):
        # A loop for fetching a finite number of frames
        frames = []
        
        for _ in range(num_frames):
            frame_contents = self.device.get_next_frame()
            frames.append(frame_contents[0])

        return np.array(frames)
    
    def get_config(self):
        # config = FmcwSimpleSequenceConfig(
        #     frame_repetition_time_s=30.303e-3,  # Frame repetition time ### 33 Hz
        #     chirp_repetition_time_s=300e-6,  # Chirp repetition time
        #     num_chirps=32,  # chirps per frame
        #     tdm_mimo=False,  # set True to enable MIMO mode, which is only valid for sensors with 2 Tx antennas
        #     chirp=FmcwSequenceChirp(
        #         start_frequency_Hz=58.5e9,  # start RF frequency, where Tx is ON
        #         end_frequency_Hz=62.5e9,  # stop RF frequency, where Tx is OFF
        #         sample_rate_Hz=2e6,  # ADC sample rate
        #         num_samples=64,  # samples per chirp
        #         rx_mask=7,  # RX mask is a 4-bit, each bit set enables that RX e.g. [1,3,7,15]
        #         tx_mask=1,  # TX antenna mask is a 2-bit (use value 3 for MIMO)
        #         tx_power_level=31,  # TX power level of 31
        #         lp_cutoff_Hz=500000,  # Anti-aliasing filter cutoff frequency, select value from data-sheet
        #         hp_cutoff_Hz=80000,  # High-pass filter cutoff frequency, select value from data-sheet
        #         if_gain_dB=30,  # IF-gain
        #     ),
        # )

        config = FmcwSimpleSequenceConfig(**self.cfg_seq,
                                           chirp = FmcwSequenceChirp(**self.cfg_chirp))

        return config
    
    def get_params(self, cfg):

        num_rx_converter = {1:1, 2:1, 4:1, 3:2, 5:2, 6:2, 7:3}  # binary to int

        params = {}
        params['bw'] = cfg['end_frequency_Hz'] - cfg['start_frequency_Hz'] # bandwidth
        params['Tc'] = cfg['chirp_repetition_time_s']  # chirp duration
        params['prf'] = 1 // params['Tc']
        params['slope'] = params['bw'] / params['Tc'] # chirp slope
        params['c'] = 299792458 # speed of light
        params['range_resolution'] = params['c'] / (2*params['bw'])
        params['Fs'] = cfg['sample_rate_Hz'] # sampling rate
        # params['max_range'] = params['Fs']*params['c'] / (2*params['slope'])
        params['max_range'] = params['range_resolution'] * cfg['num_samples'] / 2
        params['lambda'] = params['c'] / ((cfg['end_frequency_Hz'] + cfg['start_frequency_Hz']) / 2)
        params['max_vel'] = params['lambda'] / (4*params['Tc'])
        params['vel_resolution'] = 2*params['max_vel'] / cfg['num_chirps']
        # params['num_rx_antennas'] = num_rx_converter[cfg['rx_mask']]
        params['num_rx_antennas'] = cfg['num_rx']
        params["num_samples"] = cfg['num_samples']
        params["num_chirps"] = cfg['num_chirps']
        return params
    
    def close(self):
        self.device._close()
        del self.device
        self.device = None