import argparse
import json
import os
import pickle
import random
import sys

import numpy as np
import torch
import tqdm
from audiobox_aesthetics.infer import (
    initialize_predictor as initialize_ab_predictor,
)
from audiocraft.data.audio import audio_read
from audiocraft.metrics import CLAPTextConsistencyMetric
from semantic_consistency_reward import SemanticConsistencyReward

import wandb


def extract_clap_scores(args) -> dict[str, float]:
    clap_scores_json_fpath = os.path.join(args.samples_dir, "rewards/clap_scores.json")
    os.makedirs(os.path.dirname(clap_scores_json_fpath), exist_ok=True)
    print(f"clap_scores_json_fpath: {clap_scores_json_fpath}")
    if os.path.exists(clap_scores_json_fpath):
        print(f"CLAP scores already extracted and saved to {clap_scores_json_fpath}")
        with open(clap_scores_json_fpath) as f:
            return json.load(f)
    clap_model_path = "//reference/clap/music_audioset_epoch_15_esc_90.14.pt"
    clap_model_arch = "HTSAT-base"
    text_consistency_metric = CLAPTextConsistencyMetric(model_path=clap_model_path, model_arch=clap_model_arch)
    text_consistency_metric.eval()

    sample_id_2_clap_scores = {}
    # iterate over all json files in the samples_dir
    texts = []
    audios = []
    srs = []
    sizes = []
    sample_ids = []
    total_processed = 0
    for json_file in tqdm.tqdm(f for f in os.listdir(args.samples_dir) if f.endswith(".json")):
        sample_id = json_file.split(".")[0]
        json_fpath = os.path.join(args.samples_dir, json_file)

        with open(json_fpath) as f:
            data = json.load(f)
            text = data["conditioning"]["description"]
            audio_fpath = json_fpath.replace(".json", ".wav")
            assert os.path.exists(audio_fpath), f"Audio file {audio_fpath} does not exist"
            audio, sr = audio_read(audio_fpath)
            texts.append(text)
            audios.append(audio)
            srs.append(sr)
            sizes.append(audio.shape[1])
            sample_ids.append(sample_id)
        if len(texts) == args.batch_size:
            clap_scores = text_consistency_metric.extract_scores(
                torch.stack(audios),
                texts,
                torch.tensor(sizes),
                torch.tensor(srs),
            )
            total_processed += len(sample_ids)
            sample_id_2_clap_scores.update(
                {sample_id: score.item() for sample_id, score in zip(sample_ids, clap_scores, strict=False)}
            )
            texts = []
            audios = []
            srs = []
            sizes = []
            sample_ids = []

    print(f"finished processing {total_processed} samples for CLAP scores")
    with open(clap_scores_json_fpath, "w") as f:
        print(f"saving {len(sample_id_2_clap_scores)} CLAP scores to {clap_scores_json_fpath}")
        json.dump(sample_id_2_clap_scores, f)
    return sample_id_2_clap_scores


def extract_audiobox_aesthetics_scores(args) -> dict[str, float]:
    audiobox_aesthetics_json_fpath = os.path.join(args.samples_dir, "rewards/audiobox_aesthetics.json")
    os.makedirs(os.path.dirname(audiobox_aesthetics_json_fpath), exist_ok=True)
    print(f"audiobox_aesthetics_json_fpath: {audiobox_aesthetics_json_fpath}")
    if os.path.exists(audiobox_aesthetics_json_fpath):
        print(f"Audiobox aesthetics scores already extracted and saved to {audiobox_aesthetics_json_fpath}")
        with open(audiobox_aesthetics_json_fpath) as f:
            return json.load(f)
    ab_predictor = initialize_ab_predictor()
    sample_id_2_ab_scores = {}
    # iterate over all json files in the samples_dir
    audios = []
    srs = []
    sample_ids = []
    total_processed = 0
    for json_file in tqdm.tqdm(f for f in os.listdir(args.samples_dir) if f.endswith(".json")):
        sample_id = json_file.split(".")[0]
        json_fpath = os.path.join(args.samples_dir, json_file)
        with open(json_fpath) as f:
            audio_fpath = json_fpath.replace(".json", ".wav")
            assert os.path.exists(audio_fpath), f"Audio file {audio_fpath} does not exist"
            audio, sr = audio_read(audio_fpath)
            audios.append(audio)
            srs.append(sr)
            sample_ids.append(sample_id)
        if len(sample_ids) == args.batch_size:
            ab_input_list = [{"path": audio, "sample_rate": sr} for audio, sr in zip(audios, srs, strict=False)]
            ab_scores = ab_predictor.forward(ab_input_list)
            total_processed += len(sample_ids)
            sample_id_2_ab_scores.update(
                {sample_id: ab_scores_dict for sample_id, ab_scores_dict in zip(sample_ids, ab_scores, strict=False)}
            )
            audios = []
            srs = []
            sample_ids = []

    print(f"finished processing {total_processed} samples for Audiobox aesthetics scores")
    with open(audiobox_aesthetics_json_fpath, "w") as f:
        print(f"saving {len(sample_id_2_ab_scores)} Audiobox aesthetics scores to {audiobox_aesthetics_json_fpath}")
        json.dump(sample_id_2_ab_scores, f)
    return sample_id_2_ab_scores


def extract_semantic_consistency_scores(args) -> dict[str, float]:
    semantic_consistency_json_fpath = os.path.join(args.samples_dir, "rewards/semantic_consistency.json")
    os.makedirs(os.path.dirname(semantic_consistency_json_fpath), exist_ok=True)
    print(f"semantic_consistency_json_fpath: {semantic_consistency_json_fpath}")
    if os.path.exists(semantic_consistency_json_fpath):
        print(f"Semantic consistency scores already extracted and saved to {semantic_consistency_json_fpath}")
        with open(semantic_consistency_json_fpath) as f:
            return json.load(f)
    semantic_consistency_reward = SemanticConsistencyReward()
    sample_id_2_semantic_consistency_scores = {}
    # iterate over all json files in the samples_dir
    audios = []
    srs = []
    sample_ids = []
    total_processed = 0
    for json_file in tqdm.tqdm(f for f in os.listdir(args.samples_dir) if f.endswith(".json")):
        sample_id = json_file.split(".")[0]
        json_fpath = os.path.join(args.samples_dir, json_file)
        with open(json_fpath) as f:
            audio_fpath = json_fpath.replace(".json", ".wav")
            assert os.path.exists(audio_fpath), f"Audio file {audio_fpath} does not exist"
            audio, sr = audio_read(audio_fpath)
            audios.append(audio)
            srs.append(sr)
            sample_ids.append(sample_id)
        if len(sample_ids) == args.batch_size:
            scores_dicts = semantic_consistency_reward(audios, src_sr=srs)
            total_processed += len(sample_ids)
            sample_id_2_semantic_consistency_scores.update(
                {sample_id: scores_dict_i for sample_id, scores_dict_i in zip(sample_ids, scores_dicts, strict=False)}
            )
            audios = []
            srs = []
            sample_ids = []

    print(f"finished processing {total_processed} samples for Semantic consistency scores")
    with open(semantic_consistency_json_fpath, "w") as f:
        print(
            f"saving "
            f"{len(sample_id_2_semantic_consistency_scores)} Semantic "
            f"consistency scores to {semantic_consistency_json_fpath}"
        )
        print(sample_id_2_semantic_consistency_scores)
        json.dump(sample_id_2_semantic_consistency_scores, f)
    return sample_id_2_semantic_consistency_scores


def extract_rewards(args) -> dict[str, float]:
    rewards_json_fpath = os.path.join(args.samples_dir, "rewards/rewards.json")
    os.makedirs(os.path.dirname(rewards_json_fpath), exist_ok=True)
    print(f"rewards_json_fpath: {rewards_json_fpath}")
    if os.path.exists(rewards_json_fpath):
        print(f"Rewards already extracted and saved to {rewards_json_fpath}")
        with open(rewards_json_fpath) as f:
            return json.load(f)
    assert "clap" in args.rewards, "CLAP is required for MRSD"
    assert "audiobox_aesthetics" in args.rewards, "Audiobox aesthetics is required for MRSD"
    assert "semantic_consistency" in args.rewards, "Semantic consistency is required for MRSD"
    sample_id_2_clap_scores = extract_clap_scores(args)
    sample_id_2_ab_scores = extract_audiobox_aesthetics_scores(args)
    sample_id_2_semantic_consistency_scores = extract_semantic_consistency_scores(args)
    # merge into one dictionary mapping sample_id to a dictionary of rewards
    sample_id_2_rewards = {}
    # find the intersection of the keys in the three dictionaries
    common_sample_ids = (
        set(sample_id_2_clap_scores.keys())
        & set(sample_id_2_ab_scores.keys())
        & set(sample_id_2_semantic_consistency_scores.keys())
    )
    for sample_id in tqdm.tqdm(common_sample_ids):
        sample_id_2_rewards[sample_id] = {
            "clap": sample_id_2_clap_scores[sample_id],
            "audiobox_aesthetics": (sample_id_2_ab_scores[sample_id]),
            "semantic_consistency": (sample_id_2_semantic_consistency_scores[sample_id]),
        }
    with open(rewards_json_fpath, "w") as f:
        json.dump(sample_id_2_rewards, f)
    return sample_id_2_rewards


def bin_by_prompt(args, sample_id_2_rewards: dict[str, dict[str, float]]) -> dict[str, list[dict[str, float]]]:
    samples_by_prompt_fpath = os.path.join(args.samples_dir, "rewards/samples_by_prompt.json")
    os.makedirs(os.path.dirname(samples_by_prompt_fpath), exist_ok=True)
    print(f"samples_by_prompt_fpath: {samples_by_prompt_fpath}")
    if os.path.exists(samples_by_prompt_fpath):
        print(f"Samples by prompt already extracted and saved to {samples_by_prompt_fpath}")
        with open(samples_by_prompt_fpath) as f:
            return json.load(f)
    samples_by_prompt = {}
    for sample_id, rewards in tqdm.tqdm(sample_id_2_rewards.items()):
        sample_json_fpath = os.path.join(args.samples_dir, f"{sample_id}.json")
        with open(sample_json_fpath) as f:
            metadata = json.load(f)
        prompt = metadata["conditioning"]["description"]
        if prompt not in samples_by_prompt:
            samples_by_prompt[prompt] = []
        audio_fpath = sample_json_fpath.replace(".json", ".wav")
        samples_by_prompt[prompt].append(
            {
                "audio_fpath": audio_fpath,
                "rewards": rewards,
            }
        )
    with open(samples_by_prompt_fpath, "w") as f:
        json.dump(samples_by_prompt, f)
    return samples_by_prompt


def get_reward_value(sample: dict, reward: str) -> float:
    rewards_dict = sample["rewards"]
    if reward == "audiobox_aesthetics":
        return rewards_dict[reward]["PQ"]
    elif reward == "semantic_consistency":
        return rewards_dict[reward]["unmasked_semantic_consistency"]
    else:
        return rewards_dict[reward]


def extract_rewards2thresholds(args, samples_by_prompt: dict[str, list[dict[str, float]]]) -> dict[str, float]:
    rewards2abs_diffs_fpath = os.path.join(args.samples_dir, "rewards/rewards2abs_diffs.json")
    os.makedirs(os.path.dirname(rewards2abs_diffs_fpath), exist_ok=True)
    print(f"rewards2abs_diffs_fpath: {rewards2abs_diffs_fpath}")
    if os.path.exists(rewards2abs_diffs_fpath):
        print(f"Rewards2abs diffs already extracted and saved to {rewards2abs_diffs_fpath}")
        with open(rewards2abs_diffs_fpath) as f:
            reward2abs_diffs = json.load(f)
    else:
        reward2abs_diffs = {r: [] for r in args.rewards}
        for _prompt, samples in tqdm.tqdm(samples_by_prompt.items()):
            for i, sample in enumerate(samples):
                for j in range(i + 1, len(samples)):
                    sample2 = samples[j]
                    for reward in args.rewards:
                        r_abs_diff = abs(get_reward_value(sample, reward) - get_reward_value(sample2, reward))
                        reward2abs_diffs[reward].append(r_abs_diff)

        with open(rewards2abs_diffs_fpath, "w") as f:
            json.dump(reward2abs_diffs, f)

    rewards2thresholds = {r: None for r in args.rewards}
    for reward in args.rewards:
        primary_axis_th = np.percentile(reward2abs_diffs[reward], args.primary_axis_margin_perc)
        secondary_axis_th = np.percentile(reward2abs_diffs[reward], args.secondary_axis_margin_perc)
        rewards2thresholds[reward] = (primary_axis_th, secondary_axis_th)

    return rewards2thresholds


def find_mrsd_pairs(
    args,
    samples_by_prompt: dict[str, list[dict[str, float]]],
    rewards2thresholds: dict[str, float],
) -> dict[str, list[tuple[str, dict, dict]]]:
    max_samples_per_prompt = args.max_samples_per_prompt
    mrsd_pairs_pkl_fpath = os.path.join(
        args.samples_dir,
        f"rewards/mrsd_pairs_margins_{args.primary_axis_margin_perc}_{args.secondary_axis_margin_perc}.pkl",
    )
    os.makedirs(os.path.dirname(mrsd_pairs_pkl_fpath), exist_ok=True)
    print(f"mrsd_pairs_pkl_fpath: {mrsd_pairs_pkl_fpath}")
    if os.path.exists(mrsd_pairs_pkl_fpath):
        print(f"MRSD pairs already extracted and saved to {mrsd_pairs_pkl_fpath}")
        with open(mrsd_pairs_pkl_fpath, "rb") as f:
            return pickle.load(f)

    primary_reward_2_mrsd_pairs = {r: [] for r in args.rewards}
    for prompt, samples in tqdm.tqdm(samples_by_prompt.items()):
        # if there are more than max_samples_per_prompt samples,
        # randomly sample max_samples_per_prompt samples.
        if len(samples) <= 1:
            continue
        if len(samples) > max_samples_per_prompt:
            samples = random.sample(samples, max_samples_per_prompt)
        for primary_reward in args.rewards:
            for sample in samples:
                for sample2 in samples:
                    if sample2["audio_fpath"] == sample["audio_fpath"]:
                        continue
                    # primary reward criterion
                    if (
                        get_reward_value(sample, primary_reward)
                        < get_reward_value(sample2, primary_reward) + rewards2thresholds[primary_reward][0]
                    ):
                        continue

                    # secondary rewards criteria
                    secondary_reward_conflict = False
                    for secondary_reward in args.rewards:
                        if secondary_reward == primary_reward:
                            continue
                        if (
                            get_reward_value(sample, secondary_reward)
                            < get_reward_value(sample2, secondary_reward) + rewards2thresholds[secondary_reward][1]
                        ):
                            secondary_reward_conflict = True
                            break

                    if secondary_reward_conflict:
                        continue

                    # create mrsd pair
                    primary_reward_2_mrsd_pairs[primary_reward].append((prompt, sample, sample2))

    for primary_reward in args.rewards:
        print(
            f"created {len(primary_reward_2_mrsd_pairs[primary_reward])} mrsd pairs for primary reward {primary_reward}"
        )
    with open(mrsd_pairs_pkl_fpath, "wb") as f:
        pickle.dump(primary_reward_2_mrsd_pairs, f)
    return primary_reward_2_mrsd_pairs


def log_pairs_to_wandb(args, primary_reward_2_mrsd_pairs: dict[str, list[tuple[dict, dict]]]):
    wandb.init(
        project="mrflow_dpo",
        name=(f"mrsd_{args.samples_dir}_margins_{args.primary_axis_margin_perc}_{args.secondary_axis_margin_perc}"),
    )
    # table of mrsd pairs
    sample_table = wandb.Table(
        columns=[
            "primary_reward",
            "prompt",
            "pos_sample",
            "neg_sample",
            "pos_reward",
            "neg_reward",
        ]
    )
    for primary_reward in args.rewards:
        # randomly sample num_samples_to_upload_per_reward mrsd pairs
        mrsd_pairs = random.sample(
            primary_reward_2_mrsd_pairs[primary_reward],
            args.num_samples_to_upload_per_reward,
        )
        for prompt, sample, sample2 in mrsd_pairs:
            sample_table.add_data(
                primary_reward,
                prompt,
                wandb.Audio(sample["audio_fpath"]),
                wandb.Audio(sample2["audio_fpath"]),
                get_reward_value(sample, primary_reward),
                get_reward_value(sample2, primary_reward),
            )
    wandb.log({"mrsd_pairs": sample_table})
    wandb.finish()


def balance_primary_axes(
    args, primary_reward_2_mrsd_pairs: dict[str, list[tuple[str, dict, dict]]]
) -> dict[str, list[tuple[str, dict, dict]]]:
    # find the primary axis with the least mrsd pairs
    min_mrsd_pairs = min(len(mrsd_pairs) for mrsd_pairs in primary_reward_2_mrsd_pairs.values())
    # for each primary axis, sample the same number of mrsd pairs as the
    # primary axis with the least mrsd pairs
    primary_reward_2_mrsd_pairs_balanced = {r: [] for r in args.rewards}
    for primary_reward in args.rewards:
        mrsd_pairs = primary_reward_2_mrsd_pairs[primary_reward]
        mrsd_pairs_balanced = random.sample(mrsd_pairs, min_mrsd_pairs)
        primary_reward_2_mrsd_pairs_balanced[primary_reward] = mrsd_pairs_balanced
    return primary_reward_2_mrsd_pairs_balanced


def write_dataset_json_lines(
    args,
    primary_reward_2_mrsd_pairs: dict[str, list[tuple[str, dict, dict]]],
    test_fraction: float = 0.05,
    val_fraction: float = 0.05,
):
    """Create line per mrsd pair in a new jsonl file
    The format of each line is like in the audiocraft dataset example:
    {"path": "dataset/example/electro_1.mp3", "duration": 15.024,
    "sample_rate": 48000, "amplitude": null, "weight": null,
    "info_path": null}
    {"path": "dataset/example/electro_2.mp3", "duration": 20.035918367346937,
    "sample_rate": 44100, "amplitude": null, "weight": null,
    "info_path": null}
    But now as a dpo pair:
    {"path": "dataset/example/electro_1.mp3", "duration": 15.024,
    "sample_rate": 48000, "rewards:{
        "clap": 0.5,
        "audiobox_aesthetics": 0.6,
        "semantic_consistency": 0.7
    }, "negative_path": "dataset/example/electro_2.mp3",
    "negative_duration": 20.035918367346937, "negative_sample_rate": 44100,
    "negative_rewards": {
        "clap": 0.5,
        "audiobox_aesthetics": 0.6,
        "semantic_consistency": 0.7
    }, "weight": null, "info_path": null}

    something like this.

    Create train, test, and val sets according to the fractions provided.
    """
    # Create output directory
    output_dir = os.path.join(args.samples_dir, "mrsd_dataset")
    os.makedirs(output_dir, exist_ok=True)

    # Collect all pairs from all primary rewards
    all_pairs = []
    for primary_reward, mrsd_pairs in primary_reward_2_mrsd_pairs.items():
        for prompt, pos_sample, neg_sample in mrsd_pairs:
            all_pairs.append((primary_reward, prompt, pos_sample, neg_sample))

    # Shuffle all pairs
    random.shuffle(all_pairs)

    # Calculate split indices
    total_pairs = len(all_pairs)
    test_size = int(total_pairs * test_fraction)
    val_size = int(total_pairs * val_fraction)
    train_size = total_pairs - test_size - val_size

    # Split the data
    train_pairs = all_pairs[:train_size]
    val_pairs = all_pairs[train_size : train_size + val_size]
    test_pairs = all_pairs[train_size + val_size :]

    print(f"Total pairs: {total_pairs}")
    print(f"Train pairs: {len(train_pairs)}")
    print(f"Val pairs: {len(val_pairs)}")
    print(f"Test pairs: {len(test_pairs)}")

    # Helper function to get audio metadata
    def get_audio_metadata(audio_fpath):
        audio, sr = audio_read(audio_fpath)
        duration = audio.shape[1] / sr
        return {"duration": duration, "sample_rate": sr}

    # Helper function to create a JSONL entry
    def create_jsonl_entry(primary_reward, prompt, pos_sample, neg_sample):
        pos_audio_path = pos_sample["audio_fpath"]
        neg_audio_path = neg_sample["audio_fpath"]

        pos_metadata = get_audio_metadata(pos_audio_path)
        neg_metadata = get_audio_metadata(neg_audio_path)

        entry = {
            "path": pos_audio_path,
            "duration": pos_metadata["duration"],
            "sample_rate": pos_metadata["sample_rate"],
            "amplitude": None,
            "rewards": pos_sample["rewards"],
            "negative_path": neg_audio_path,
            "negative_duration": neg_metadata["duration"],
            "negative_sample_rate": neg_metadata["sample_rate"],
            "negative_rewards": neg_sample["rewards"],
            "weight": None,
            "info_path": None,
            "primary_reward": primary_reward,
            "prompt": prompt,
        }
        return entry

    # Write splits to files
    splits = {"train": train_pairs, "val": val_pairs, "test": test_pairs}

    for split_name, split_pairs in splits.items():
        output_fpath = os.path.join(output_dir, f"{split_name}.jsonl")
        print(f"Writing {len(split_pairs)} pairs to {output_fpath}")

        with open(output_fpath, "w") as f:
            for primary_reward, prompt, pos_sample, neg_sample in tqdm.tqdm(split_pairs, desc=f"Writing {split_name}"):
                entry = create_jsonl_entry(primary_reward, prompt, pos_sample, neg_sample)
                f.write(json.dumps(entry) + "\n")

        print(f"Finished writing {split_name} set to {output_fpath}")

    print(f"Dataset written to {output_dir}")


def construct_mrsd_dataset(args):
    sample_id_2_rewards = extract_rewards(args)
    samples_by_prompt = bin_by_prompt(args, sample_id_2_rewards)
    rewards2thresholds = extract_rewards2thresholds(args, samples_by_prompt)
    primary_reward_2_mrsd_pairs: dict[str, list[tuple[str, dict, dict]]] = find_mrsd_pairs(
        args, samples_by_prompt, rewards2thresholds
    )
    primary_reward_2_mrsd_pairs = balance_primary_axes(args, primary_reward_2_mrsd_pairs)
    write_dataset_json_lines(args, primary_reward_2_mrsd_pairs)
    log_pairs_to_wandb(args, primary_reward_2_mrsd_pairs)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples_dir", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--max_samples_per_prompt", type=int, default=8)
    parser.add_argument(
        "--rewards",
        type=list,
        default=["clap", "audiobox_aesthetics", "semantic_consistency"],
    )
    parser.add_argument("--primary_axis_margin_perc", type=int, default=95)
    parser.add_argument("--secondary_axis_margin_perc", type=int, default=50)
    parser.add_argument("--num_samples_to_upload_per_reward", type=int, default=20)
    args = parser.parse_args()
    return args


def main():
    audiocraft_repo_path = "YOUR_LOCAL_AUDIOCRAFT_REPO_PATH"
    sys.path.append(os.path.dirname(audiocraft_repo_path))

    args = parse_args()
    construct_mrsd_dataset(args)
    print("Done")


if __name__ == "__main__":
    main()
