# MR-FlowDPO
Source code for "MR-FLOWDPO: Multi-Reward Direct Preference Optimization for Flow-Matching Text-to-Music Generation".

### Stage 1 - Reference Model Sampling
- [ ] Multiple generations per prompt
```
dora run solver=jasco/chords_drums dataset.batch_size=2 dataset.num_workers=0 logging.log_updates=400 continue_from=//pretrained/facebook/jasco-chords-drums-400M execute_only=generate dataset.generate.num_samples=10000 generate.lm.cfg_coef_all=3.0 generate.lm.cfg_coef_txt=0.0
```

### Stage 2 - Preference Data Creation with Multi Reward Strong Domination (MRSD)
#### Rewards Extraction
- [x] CLAP
- [x] Audiobox aesthetics
- [x] HuBERT likelihood
- [x] MRSD impl.

```
python -m src/construct_mrsd_dataset.py --samples_dir AUDIOCRAFT_XP_OF_STAGE_1/samples/1/
```

### Stage 4 - DPO Solver
- [ ] DPO impl.

### Stage 5 - Metrics
- [ ] BPM std

