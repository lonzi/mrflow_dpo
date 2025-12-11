# MR-FlowDPO
Official source code for "MR-FLOWDPO: Multi-Reward Direct Preference Optimization for Flow-Matching Text-to-Music Generation" - by Alon Ziv, Sanyuan Chen, Andros Tjandra, Yossi Adi, Wei-Ning Hsu, and Bowen Shi.

Paper:
https://arxiv.org/abs/TODO

Project Page:
https://lonzi.github.io/mr_flowdpo_demopage/

Note: 
In order to use this code you should clone two additional repositories:
1. Audiocraft by Meta: https://github.com/facebookresearch/audiocraft/tree/main.
2. MusicFM: https://github.com/minzwon/musicfm - A Foundation Model for Music Informatics, ICASSP 2024, Minz Won, Yun-Ning Hung, and Duc Le.

## Citation for this work:
MR-FlowDPO: 
```
TODO
```
## Citation for external repos:
1. Audiocraft:
```
@inproceedings{copet2023simple,
    title={Simple and Controllable Music Generation},
    author={Jade Copet and Felix Kreuk and Itai Gat and Tal Remez and David Kant and Gabriel Synnaeve and Yossi Adi and Alexandre Défossez},
    booktitle={Thirty-seventh Conference on Neural Information Processing Systems},
    year={2023},
}
```
2. MusicFM: 
```
@misc{won2023foundationmodelmusicinformatics,
      title={A Foundation Model for Music Informatics}, 
      author={Minz Won and Yun-Ning Hung and Duc Le},
      year={2023},
      eprint={2311.03318},
      archivePrefix={arXiv},
      primaryClass={cs.SD},
      url={https://arxiv.org/abs/2311.03318}, 
}
```

## MR-FlowDPO - HOW-TO
### Stage 0 - Setup
- [ ] Add requirements.txt

### Stage 1 - Reference Model Sampling
- [ ] Multiple generations per prompt
```
cd YOUR_LOCAL_AUDIOCRAFT_REPO_PATH

dora run solver=jasco/chords_drums dataset.batch_size=2 dataset.num_workers=0 logging.log_updates=400 continue_from=//pretrained/facebook/jasco-chords-drums-400M execute_only=generate dataset.generate.num_samples=10000 generate.lm.cfg_coef_all=3.0 generate.lm.cfg_coef_txt=0.0
```

### Stage 2 - Preference Data Creation with Multi Reward Strong Domination (MRSD)
#### Rewards Extraction
- [x] CLAP
- [x] Audiobox aesthetics
- [x] Semantic Consistency Reward (MusicFM based)
- [x] MRSD impl.

```
python src/construct_mrsd_dataset.py --samples_dir AUDIOCRAFT_XP_OF_STAGE_1/samples/1/
```

### Stage 3 - DPO Solver
- [x] DPO impl.
- [ ] Reward prompting.

[note: first you need to copy the content from dpo_audiocraft dir into your local audiocraft repo]

```
cd YOUR_LOCAL_AUDIOCRAFT_REPO_PATH

dora run solver=flow_dpo/flow_dpo_jasco continue_from=//pretrained/facebook/jasco-chords-drums-400M
```

### Stage 4 - Metrics
- [ ] BPM std
