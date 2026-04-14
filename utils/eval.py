import os 
import numpy as np
from scipy.ndimage import gaussian_filter1d
from sklearn.metrics import roc_auc_score, precision_recall_curve, roc_curve,auc
import csv
from tqdm import tqdm
from dataset import shanghaitech_hr_skip
from sklearn.preprocessing import StandardScaler
import torch

target_fnr = 0.1

def combined_score_dataset(score_c, score_f, metadata_c, metadata_f, args=None):
    if args is not None and (getattr(args, "no_metrics", False) or not getattr(args, "mask_root", None)):
        return 0.0, 0.0, 1.0, 0.0, 1.0, 0.0
    args.branch = "SPARTA_C"
    gt_arr_c, scores_arr_c = get_dataset_scores(score_c, metadata_c, args=args)
    if len(scores_arr_c) == 0:
        print("No clips to score for branch C; returning default metrics.")
        return 0.0, 0.0, 1.0, 0.0, 1.0, 0.0
    scores_arr_c = smooth_scores(scores_arr_c)
    gt_np_c = np.concatenate(gt_arr_c)
    scores_np_c = np.concatenate(scores_arr_c)
    
    args.branch = "SPARTA_F"
    gt_arr_f, scores_arr_f = get_dataset_scores(score_f, metadata_f, args=args)
    if len(scores_arr_f) == 0:
        print("No clips to score for branch F; returning default metrics.")
        return 0.0, 0.0, 1.0, 0.0, 1.0, 0.0
    scores_arr_f = smooth_scores(scores_arr_f)
    gt_np_f = np.concatenate(gt_arr_f)
    scores_np_f = np.concatenate(scores_arr_f)
    
    scaler1 = StandardScaler()
    scaler2 = StandardScaler()

    scaled_c = scaler1.fit_transform(scores_np_c.reshape(-1, 1))
    scaled_f = scaler2.fit_transform(scores_np_f.reshape(-1, 1))
    combined_score = scaled_c + scaled_f
    auc_roc, auc_precision_recall, EER, eer_threshold, fpr_at_target_fnr, threshold_at_target_fnr = score_auc(combined_score, gt_np_f)
    return auc_roc, auc_precision_recall, EER, eer_threshold, fpr_at_target_fnr, threshold_at_target_fnr


def score_dataset(score, metadata, args=None):
    if args is not None and (getattr(args, "no_metrics", False) or not getattr(args, "mask_root", None)):
        return 0.0, 0.0, 1.0, 0.0, 1.0, 0.0
    # Pass 1: Get the raw scores to calculate the global EER threshold
    gt_arr, scores_arr = get_dataset_scores(score, metadata, args=args, save_mode=False)
    
    if len(scores_arr) == 0:
        return 0.0, 0.0, 1.0, 0.0, 1.0, 0.0
        
    scores_arr_smoothed = smooth_scores(scores_arr)
    gt_np = np.concatenate(gt_arr)
    scores_np = np.concatenate(scores_arr_smoothed)
    
    # Calculate EER and other metrics
    auc_roc, auc_pr, EER, eer_threshold, fpr_at_fnr, th_at_fnr = score_auc(scores_np, gt_np)
    
    # Pass 2: Now that we have the 'eer_threshold', run it again to save CSVs
    if args.save_results:
        print(f"Finalizing CSVs with EER Threshold: {eer_threshold:.4f}")
        get_dataset_scores(score, metadata, args=args, save_mode=True, threshold=eer_threshold)
        
    return auc_roc, auc_pr, EER, eer_threshold, fpr_at_fnr, th_at_fnr


def get_dataset_scores(scores, metadata, args=None, save_mode=False, threshold=None):
    dataset_gt_arr = []
    dataset_scores_arr = []
    metadata_np = np.array(metadata)
    # If metadata comes in as a 1-D object array of lists, force it to 2-D
    if metadata_np.ndim == 1 and metadata_np.size > 0:
        try:
            metadata_np = np.vstack(metadata_np)
        except Exception:
            pass

    # Validate that metadata is a 2-D table with the expected columns
    # Columns are: scene_id, clip_id, person_id, start_frame
    if metadata_np.ndim != 2 or metadata_np.size == 0 or metadata_np.shape[1] < 4:
        print("No metadata available for scoring; skipping clip scoring.")
        return [], []

    clip_list = os.listdir(args.mask_root)
    clip_list = sorted(fn for fn in clip_list if fn.endswith('.npy'))

    print("Scoring {} clips".format(len(clip_list)))
    if len(clip_list) == 0:
        return [], []

    # Ensure we have a place to save CSVs when requested
    if save_mode and args.save_results:
        # Default directory if the user did not pass --save_results_dir
        if not getattr(args, "save_results_dir", None):
            fallback_dir = args.model_save_dir if getattr(args, "model_save_dir", None) else "."
            args.save_results_dir = os.path.join(fallback_dir, "evaluation_results")
        os.makedirs(args.save_results_dir, exist_ok=True)

    for clip in tqdm(clip_list, desc="Saving CSVs" if save_mode else "Loading Scores"):
        clip_gt, clip_score = get_clip_score(scores, clip, metadata_np, metadata, args.mask_root, args)
        
        if clip_score is not None:
            # Replace invalid scores (-inf/inf) with the minimum finite score in the clip
            finite_mask = np.isfinite(clip_score)
            if finite_mask.any():
                fill_val = clip_score[finite_mask].min()
                clip_score = np.where(finite_mask, clip_score, fill_val)
            else:
                # Skip clips with no valid scores
                continue

            dataset_gt_arr.append(clip_gt)
            dataset_scores_arr.append(clip_score)
            
            if save_mode and args.save_results:
                clip_pred = np.full_like(clip_score, -np.inf)
                valid_mask = (clip_score != -np.inf)
                
                if threshold is not None:
                    clip_pred[valid_mask] = (clip_score[valid_mask] > threshold).astype(int)
                
                # clip looks like "01_0209.npy"; we need the stem without the extension
                file_basename = os.path.splitext(clip)[0]
                save_fn = os.path.join(args.save_results_dir, file_basename + ".csv")
                
                with open(save_fn, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(["Actual", "Predicted", "Score"])
                    writer.writerows(zip(clip_gt, clip_pred, clip_score))

    return dataset_gt_arr, dataset_scores_arr

def score_auc(scores_np, gt):
    scores_np[scores_np == np.inf] = scores_np[scores_np != np.inf].max() if np.any(scores_np != np.inf) else 0
    scores_np[scores_np == -1 * np.inf] = scores_np[scores_np != -1 * np.inf].min() if np.any(scores_np != -1 * np.inf) else 0

    # If all scores are identical, ROC/PR are undefined; return safe defaults
    if np.ptp(scores_np) == 0:
        auc_roc = 0.5
        auc_precision_recall = 0.0
        EER = 1.0
        eer_threshold = 0.0
        fpr_at_target_fnr = 1.0
        threshold_at_target_fnr = 0.0
        return auc_roc, auc_precision_recall, EER, eer_threshold, fpr_at_target_fnr, threshold_at_target_fnr

    # Handle degenerate GT (all 0 or all 1) gracefully
    if len(np.unique(gt)) < 2:
        # No positive or no negative class: metrics are undefined. Return zeros and safe thresholds.
        auc_roc = 0.0
        auc_precision_recall = 0.0
        EER = 1.0
        eer_threshold = 0.0
        fpr_at_target_fnr = 1.0
        threshold_at_target_fnr = 0.0
        return auc_roc, auc_precision_recall, EER, eer_threshold, fpr_at_target_fnr, threshold_at_target_fnr

    auc_roc = roc_auc_score(gt, scores_np)
    precision, recall, thresholds = precision_recall_curve(gt, scores_np)
    auc_precision_recall = auc(recall, precision)

    fpr, tpr, threshold = roc_curve(gt, scores_np, pos_label=1)
    fnr = 1 - tpr
    diff = np.absolute(fnr - fpr)
    if np.all(np.isnan(diff)):
        eer_threshold = 0.0
        EER = 1.0
    else:
        eer_idx = np.nanargmin(diff)
        eer_threshold = threshold[eer_idx]
        EER = fpr[eer_idx]

    idx_closest_fnr = np.nanargmin(np.abs(fnr - target_fnr)) if not np.all(np.isnan(fnr)) else 0
    threshold_at_target_fnr = threshold[idx_closest_fnr] if len(threshold) else 0.0
    fpr_at_target_fnr = fpr[idx_closest_fnr] if len(fpr) else 1.0
    return auc_roc, auc_precision_recall, EER, eer_threshold, fpr_at_target_fnr, threshold_at_target_fnr


def smooth_scores(scores_arr, sigma=7):
    for s in range(len(scores_arr)):
        for sig in range(1, sigma):
            scores_arr[s] = gaussian_filter1d(scores_arr[s], sigma=sig)
    return scores_arr


def get_clip_score(scores, clip, metadata_np, metadata, per_frame_scores_root, args):

    name_parts = clip.split('.')[0].split('_')
    if len(name_parts) >= 2:
        scene_id, clip_id = [int(i) for i in name_parts[:2]]
    else:
        # fallback for filenames like 000001.npy (no scene prefix)
        scene_id = 0
        clip_id = int(name_parts[0])
    if shanghaitech_hr_skip((args.dataset == 'ShanghaiTech-HR'), scene_id, clip_id):
        return None, None
    
    clip_metadata_inds = np.where((metadata_np[:, 1] == clip_id) &
                                  (metadata_np[:, 0] == scene_id))[0]
    clip_metadata = metadata[clip_metadata_inds]
    clip_fig_idxs = set([arr[2] for arr in clip_metadata])
    clip_res_fn = os.path.join(per_frame_scores_root, clip)
    clip_gt = np.load(clip_res_fn)

    scores_zeros = np.ones(clip_gt.shape[0]) * np.inf * -1
    if len(clip_fig_idxs) == 0:
        clip_person_scores_dict = {0: np.copy(scores_zeros)}
    else:
        clip_person_scores_dict = {i: np.copy(scores_zeros) for i in clip_fig_idxs}

    for person_id in clip_fig_idxs:
        person_metadata_inds = \
            np.where(
                (metadata_np[:, 1] == clip_id) & (metadata_np[:, 0] == scene_id) & (metadata_np[:, 2] == person_id))[0]
        pid_scores = scores[person_metadata_inds]

        pid_frame_inds = np.array([metadata[i][3] for i in person_metadata_inds]).astype(int)
        if args.branch == "SPARTA_C":
            clip_person_scores_dict[person_id][pid_frame_inds + int(args.seg_len / 2)] = pid_scores
        else:
            clip_person_scores_dict[person_id][pid_frame_inds + int(args.seg_len / 2)+args.seg_len] = pid_scores

    clip_ppl_score_arr = np.stack(list(clip_person_scores_dict.values()))
    clip_score = np.amax(clip_ppl_score_arr, axis=0)

    return clip_gt, clip_score

def eval (args, model_args, model, tokenizer, loss_func,  loader):
    eval_loss = []
    model.eval()
    with torch.no_grad():
        for i, data_batch in enumerate(tqdm(loader['test'])):
            input_data, target_data = tokenizer.process_data(data_batch)
            if args.branch == "SPARTA_C":
                recon = model.forward(input_data, input_data)
                loss = loss_func.calculate(input_data, recon)
                eval_loss.extend(loss.cpu().numpy())
            elif args.branch == "SPARTA_F":
                pred = model.forward(input_data, target_data)
                loss = loss_func.calculate(target_data, pred)
                eval_loss.extend(loss.cpu().numpy())
    return eval_loss
