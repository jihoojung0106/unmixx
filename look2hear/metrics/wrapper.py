import csv
import torch
import numpy as np
import logging

from torch_mir_eval.separation import bss_eval_sources
import fast_bss_eval
from ..losses import (
    PITLossWrapper,
    pairwise_neg_sisdr,
    pairwise_neg_snr,
    singlesrc_neg_sisdr,
    PairwiseNegSDR,
)

logger = logging.getLogger(__name__)


class MetricsTracker:
    def __init__(self, save_file: str = ""):
        self.all_sdrs = []
        self.all_sdrs_i = []
        self.all_sisnrs = []
        self.all_sisnrs_i = []
        csv_columns = ["snt_id", "sdr", "sdr_first","sdr_i", "sdr_i_first","si-snr", 'si-snr_first',"si-snr_i",'si-snr_i_first']
           
        #csv_columns = ["snt_id", "sdr", "sdr_i", "si-snr", "si-snr_i"]
        self.results_csv = open(save_file, "w")
        self.writer = csv.DictWriter(self.results_csv, fieldnames=csv_columns)
        self.writer.writeheader()
        self.pit_sisnr = PITLossWrapper(
            PairwiseNegSDR("sisdr", zero_mean=False), pit_from="pw_mtx"
        )
        self.pit_snr = PITLossWrapper(
            PairwiseNegSDR("snr", zero_mean=False), pit_from="pw_mtx"
        )

    def __call__(self, mix, clean, estimate, key,estimate1=None):
        # sisnr
        sisnr = self.pit_sisnr(estimate.unsqueeze(0), clean.unsqueeze(0))
        if estimate1 is not None:
            sisnr1 = self.pit_sisnr(estimate1.unsqueeze(0), clean.unsqueeze(0))
        mix = torch.stack([mix] * clean.shape[0], dim=0)
        sisnr_baseline = self.pit_sisnr(mix.unsqueeze(0), clean.unsqueeze(0))
        sisnr_i = sisnr - sisnr_baseline
        if estimate1 is not None:
            sisnr_i1 = sisnr_i - sisnr1

        # sdr
        sdr = -fast_bss_eval.sdr_pit_loss(estimate, clean).mean()
        if estimate1 is not None:
            sdr1 = -fast_bss_eval.sdr_pit_loss(estimate1, clean).mean()
        sdr_baseline = -fast_bss_eval.sdr_pit_loss(mix, clean).mean()
        sdr_i = sdr - sdr_baseline
        if estimate1 is not None:
            sdr_i1 = sdr_i - sdr1
        # import pdb; pdb.set_trace()
        if estimate1 is None:
            row = {
                "snt_id": key,
                "sdr": sdr.item(),
                "sdr_i": sdr_i.item(),
                "si-snr": -sisnr.item(),
                "si-snr_i": -sisnr_i.item(),
            }
        else:
            row = {
            "snt_id": key,
            "sdr": sdr.item(),
            'sdr_first': sdr1.item(),
            "sdr_i": sdr_i.item(),
            'sdr_i_first': sdr_i1.item(),
            "si-snr": -sisnr.item(),
            'si-snr_first': -sisnr1.item(),
            "si-snr_i": -sisnr_i.item(),
            'si-snr_i_first': -sisnr_i1.item(),
         }
         
        self.writer.writerow(row)
        # Metric Accumulation
        self.all_sdrs.append(sdr.item())
        self.all_sdrs_i.append(sdr_i.item())
        self.all_sisnrs.append(-sisnr.item())
        self.all_sisnrs_i.append(-sisnr_i.item())
    
    def update(self, ):
        return {"sdr_i": np.array(self.all_sdrs_i).mean(),
                "si-snr_i": np.array(self.all_sisnrs_i).mean()
                }

    def final(self,):
        row = {
            "snt_id": "avg",
            "sdr": np.array(self.all_sdrs).mean(),
            "sdr_i": np.array(self.all_sdrs_i).mean(),
            "si-snr": np.array(self.all_sisnrs).mean(),
            "si-snr_i": np.array(self.all_sisnrs_i).mean(),
        }
        self.writer.writerow(row)
        row = {
            "snt_id": "std",
            "sdr": np.array(self.all_sdrs).std(),
            "sdr_i": np.array(self.all_sdrs_i).std(),
            "si-snr": np.array(self.all_sisnrs).std(),
            "si-snr_i": np.array(self.all_sisnrs_i).std(),
        }
        self.writer.writerow(row)
        self.results_csv.close()
