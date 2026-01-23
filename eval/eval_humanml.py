from utils.parser_util import evaluation_parser
from utils.fixseed import fixseed
from datetime import datetime
from data_loaders.humanml.motion_loaders.model_motion_loaders import get_mdm_loader  # get_motion_loader
from data_loaders.humanml.utils.metrics import *
from data_loaders.humanml.networks.evaluator_wrapper import EvaluatorMDMWrapper
from collections import OrderedDict
from data_loaders.humanml.scripts.motion_process import *
from data_loaders.humanml.utils.utils import *
from utils.model_util import create_model_and_diffusion, load_saved_model

from diffusion import logger
from utils import dist_util
from data_loaders.get_data import get_dataset_loader
from utils.sampler_util import ClassifierFreeSampleModel
from train.train_platforms import ClearmlPlatform, TensorboardPlatform, NoPlatform, WandBPlatform  # required for the eval operation

# Import geodesic distance functions for rotation quality evaluation
from diffusion.losses import geodesic_distance, rot6d_to_quaternion

torch.multiprocessing.set_sharing_strategy('file_system')

def evaluate_matching_score(eval_wrapper, motion_loaders, file, num_samples_limit=None):
    """
    This function processes BOTH ground truth AND generated samples
    """
    match_score_dict = OrderedDict({})
    R_precision_dict = OrderedDict({})
    activation_dict = OrderedDict({})
    print('========== Evaluating Matching Score ==========')
    
    for motion_loader_name, motion_loader in motion_loaders.items():
        all_motion_embeddings = []
        all_size = 0
        matching_score_sum = 0
        top_k_count = 0
        skipped_batches = 0
        
        with torch.no_grad():
            for idx, batch in enumerate(motion_loader):
                # Apply sample limit
                if num_samples_limit is not None and all_size >= num_samples_limit:
                    break
                
                word_embeddings, pos_one_hots, _, sent_lens, motions, m_lens, _ = batch
                
                # Check for NaN/Inf
                if (torch.isnan(motions).any() or torch.isinf(motions).any() or
                    torch.isnan(word_embeddings).any() or torch.isinf(word_embeddings).any()):
                    skipped_batches += 1
                    continue
                
                text_embeddings, motion_embeddings = eval_wrapper.get_co_embeddings(
                    word_embs=word_embeddings,
                    pos_ohot=pos_one_hots,
                    cap_lens=sent_lens,
                    motions=motions,
                    m_lens=m_lens
                )
                
                if (torch.isnan(text_embeddings).any() or torch.isinf(text_embeddings).any() or
                    torch.isnan(motion_embeddings).any() or torch.isinf(motion_embeddings).any()):
                    skipped_batches += 1
                    continue
                
                # Truncate batch if needed
                batch_size = text_embeddings.shape[0]
                if num_samples_limit is not None:
                    remaining = num_samples_limit - all_size
                    if remaining < batch_size:
                        text_embeddings = text_embeddings[:remaining]
                        motion_embeddings = motion_embeddings[:remaining]
                        batch_size = remaining
                
                text_emb_np = text_embeddings.cpu().numpy()
                motion_emb_np = motion_embeddings.cpu().numpy()
                
                if (np.isnan(text_emb_np).any() or np.isinf(text_emb_np).any() or
                    np.isnan(motion_emb_np).any() or np.isinf(motion_emb_np).any()):
                    skipped_batches += 1
                    continue
                
                dist_mat = euclidean_distance_matrix(text_emb_np, motion_emb_np)
                
                if np.isnan(dist_mat).any() or np.isinf(dist_mat).any():
                    skipped_batches += 1
                    continue
                
                matching_score_sum += dist_mat.trace()
                argsmax = np.argsort(dist_mat, axis=1)
                top_k_mat = calculate_top_k(argsmax, top_k=3)
                top_k_count += top_k_mat.sum(axis=0)
                all_size += batch_size
                all_motion_embeddings.append(motion_emb_np)

            if skipped_batches > 0:
                print(f'Warning: [{motion_loader_name}] Skipped {skipped_batches} batches due to NaN/Inf')
                print(f'Warning: [{motion_loader_name}] Skipped {skipped_batches} batches', file=file, flush=True)
            
            print(f'[{motion_loader_name}] Collected {all_size} samples')
            print(f'[{motion_loader_name}] Collected {all_size} samples', file=file, flush=True)
            
            if all_size == 0:
                matching_score = float('nan')
                R_precision = np.array([float('nan')] * 3)
                all_motion_embeddings = np.array([])
            else:
                all_motion_embeddings = np.concatenate(all_motion_embeddings, axis=0)
                matching_score = matching_score_sum / all_size
                R_precision = top_k_count / all_size
            
            match_score_dict[motion_loader_name] = matching_score
            R_precision_dict[motion_loader_name] = R_precision
            activation_dict[motion_loader_name] = all_motion_embeddings

        print(f'---> [{motion_loader_name}] Matching Score: {matching_score:.4f}')
        print(f'---> [{motion_loader_name}] Matching Score: {matching_score:.4f}', file=file, flush=True)

        line = f'---> [{motion_loader_name}] R_precision: '
        for i in range(len(R_precision)):
            line += '(top %d): %.4f ' % (i+1, R_precision[i])
        print(line)
        print(line, file=file, flush=True)

    return match_score_dict, R_precision_dict, activation_dict



def evaluate_fid(eval_wrapper, groundtruth_loader, activation_dict, file, num_samples_limit=None):
    """
    Fixed version - only computes GT once and compares generated models against it
    """
    eval_dict = OrderedDict({})
    gt_motion_embeddings = []
    total_batches = 0
    skipped_batches = 0
    total_samples = 0
    
    print('========== Evaluating FID ==========')
    
    # Compute ground truth embeddings (the reference distribution)
    with torch.no_grad():
        for idx, batch in enumerate(groundtruth_loader):
            if num_samples_limit is not None and total_samples >= num_samples_limit:
                break
            
            total_batches += 1
            _, _, _, sent_lens, motions, m_lens, _ = batch
            
            if torch.isnan(motions).any() or torch.isinf(motions).any():
                skipped_batches += 1
                print(f'Warning: Skipping GT batch {idx} due to NaN/Inf')
                print(f'Warning: Skipping GT batch {idx} due to NaN/Inf', file=file, flush=True)
                continue
            
            motion_embeddings = eval_wrapper.get_motion_embeddings(
                motions=motions,
                m_lens=m_lens
            )
            
            if torch.isnan(motion_embeddings).any() or torch.isinf(motion_embeddings).any():
                skipped_batches += 1
                print(f'Warning: Skipping GT batch {idx} embeddings due to NaN/Inf')
                print(f'Warning: Skipping GT batch {idx} embeddings due to NaN/Inf', file=file, flush=True)
                continue
            
            # Truncate batch if needed
            batch_size = motion_embeddings.shape[0]
            if num_samples_limit is not None:
                remaining = num_samples_limit - total_samples
                if remaining < batch_size:
                    motion_embeddings = motion_embeddings[:remaining]
                    batch_size = remaining
            
            total_samples += batch_size
            gt_motion_embeddings.append(motion_embeddings.cpu().numpy())
    
    print(f'\n=== FID Ground Truth Diagnostics ===')
    print(f'GT batches processed: {total_batches}')
    print(f'GT batches skipped: {skipped_batches}')
    print(f'GT samples collected: {total_samples}')
    print(f'=================================\n')
    
    if len(gt_motion_embeddings) == 0:
        print('Error: No valid ground truth embeddings!')
        for model_name in activation_dict.keys():
            if model_name != 'ground truth':  # Skip GT entry
                eval_dict[model_name] = float('nan')
        return eval_dict
    
    gt_motion_embeddings = np.concatenate(gt_motion_embeddings, axis=0)
    print(f'GT embeddings shape: {gt_motion_embeddings.shape}')
    print(f'GT embeddings shape: {gt_motion_embeddings.shape}', file=file, flush=True)
    
    gt_mu, gt_cov = calculate_activation_statistics(gt_motion_embeddings)

    # Now compare each GENERATED model against GT
    for model_name, motion_embeddings in activation_dict.items():
        # CRITICAL: Skip the 'ground truth' entry - we don't compare GT to itself!
        if model_name == 'ground truth':
            print(f'Skipping [{model_name}] - not computing FID for GT vs GT')
            print(f'Skipping [{model_name}] - not computing FID for GT vs GT', file=file, flush=True)
            continue
        
        print(f'\n=== {model_name} FID Evaluation ===')
        print(f'Generated embeddings shape: {motion_embeddings.shape}')
        print(f'GT embeddings shape: {gt_motion_embeddings.shape}')
        print(f'Sample count difference: {abs(motion_embeddings.shape[0] - gt_motion_embeddings.shape[0])}')
        
        # Check for NaN/Inf
        if np.isnan(motion_embeddings).any() or np.isinf(motion_embeddings).any():
            print(f'Warning: [{model_name}] contains NaN/Inf, skipping')
            print(f'Warning: [{model_name}] contains NaN/Inf, skipping', file=file, flush=True)
            eval_dict[model_name] = float('nan')
            continue
        
        mu, cov = calculate_activation_statistics(motion_embeddings)
        
        try:
            fid = calculate_frechet_distance(gt_mu, gt_cov, mu, cov)
            print(f'---> [{model_name}] FID: {fid:.4f}')
            print(f'---> [{model_name}] FID: {fid:.4f}', file=file, flush=True)
            eval_dict[model_name] = fid
        except Exception as e:
            print(f'Error calculating FID for [{model_name}]: {str(e)}')
            print(f'Error calculating FID for [{model_name}]: {str(e)}', file=file, flush=True)
            eval_dict[model_name] = float('nan')
    
    return eval_dict


def evaluate_diversity(activation_dict, file, diversity_times):
    eval_dict = OrderedDict({})
    print('========== Evaluating Diversity ==========')
    for model_name, motion_embeddings in activation_dict.items():
        diversity = calculate_diversity(motion_embeddings, diversity_times)
        eval_dict[model_name] = diversity
        print(f'---> [{model_name}] Diversity: {diversity:.4f}')
        print(f'---> [{model_name}] Diversity: {diversity:.4f}', file=file, flush=True)
    return eval_dict


def evaluate_multimodality(eval_wrapper, mm_motion_loaders, file, mm_num_times):
    eval_dict = OrderedDict({})
    print('========== Evaluating MultiModality ==========')
    for model_name, mm_motion_loader in mm_motion_loaders.items():
        mm_motion_embeddings = []
        with torch.no_grad():
            for idx, batch in enumerate(mm_motion_loader):
                # (1, mm_replications, dim_pos)
                motions, m_lens = batch
                motion_embedings = eval_wrapper.get_motion_embeddings(motions[0], m_lens[0])
                mm_motion_embeddings.append(motion_embedings.unsqueeze(0))
        if len(mm_motion_embeddings) == 0:
            multimodality = 0
        else:
            mm_motion_embeddings = torch.cat(mm_motion_embeddings, dim=0).cpu().numpy()
            multimodality = calculate_multimodality(mm_motion_embeddings, mm_num_times)
        print(f'---> [{model_name}] Multimodality: {multimodality:.4f}')
        print(f'---> [{model_name}] Multimodality: {multimodality:.4f}', file=file, flush=True)
        eval_dict[model_name] = multimodality
    return eval_dict


def evaluate_geodesic(gt_loader, gen_loader, file, num_joints=22):
    """
    Evaluate geodesic distance between ground truth and generated rotations.

    For HumanML3D with 22 joints:
    - Rotation data starts at index 4 + (22-1)*3 = 67
    - Rotation data ends at index 67 + (22-1)*6 = 193
    - Each joint has 6D rotation representation

    Returns:
        dict with 'mean_geodesic', 'std_geodesic', 'median_geodesic'
    """
    print('========== Evaluating Geodesic Distance ==========')
    print('========== Evaluating Geodesic Distance ==========', file=file, flush=True)

    # Rotation indices for HumanML3D format
    rot_start_idx = 4 + (num_joints - 1) * 3  # 67 for 22 joints
    rot_end_idx = rot_start_idx + (num_joints - 1) * 6  # 193 for 22 joints
    n_rot_joints = num_joints - 1  # 21 non-root joints

    all_geodesic_distances = []

    with torch.no_grad():
        # Iterate through both loaders in parallel
        for (gt_batch, gen_batch) in zip(gt_loader, gen_loader):
            # Unpack batches
            _, _, _, _, gt_motions, gt_m_lens, _ = gt_batch
            _, _, _, _, gen_motions, gen_m_lens, _ = gen_batch

            batch_size = gt_motions.shape[0]

            for i in range(batch_size):
                # Get valid length (minimum of both to ensure fair comparison)
                gt_len = int(gt_m_lens[i].item()) if torch.is_tensor(gt_m_lens[i]) else int(gt_m_lens[i])
                gen_len = int(gen_m_lens[i].item()) if torch.is_tensor(gen_m_lens[i]) else int(gen_m_lens[i])
                valid_len = min(gt_len, gen_len)

                if valid_len <= 0:
                    continue

                # Extract rotation data: [seq_len, 263] -> [valid_len, 126]
                gt_rot = gt_motions[i, :valid_len, rot_start_idx:rot_end_idx]  # [valid_len, 126]
                gen_rot = gen_motions[i, :valid_len, rot_start_idx:rot_end_idx]  # [valid_len, 126]

                # Check for NaN/Inf
                if torch.isnan(gt_rot).any() or torch.isnan(gen_rot).any():
                    continue
                if torch.isinf(gt_rot).any() or torch.isinf(gen_rot).any():
                    continue

                # Reshape to [valid_len, n_rot_joints, 6]
                gt_rot = gt_rot.reshape(valid_len, n_rot_joints, 6)
                gen_rot = gen_rot.reshape(valid_len, n_rot_joints, 6)

                # Convert to quaternions: [valid_len, n_rot_joints, 4]
                gt_quats = rot6d_to_quaternion(gt_rot)
                gen_quats = rot6d_to_quaternion(gen_rot)

                # Compute geodesic distance: [valid_len, n_rot_joints]
                geo_dist = geodesic_distance(gt_quats, gen_quats)

                # Check for NaN in result
                if torch.isnan(geo_dist).any():
                    continue

                # Convert to degrees and store mean per frame
                geo_dist_degrees = torch.rad2deg(geo_dist)  # Convert radians to degrees
                mean_geo_per_frame = geo_dist_degrees.mean(dim=1)  # Mean across joints per frame
                all_geodesic_distances.append(mean_geo_per_frame.cpu().numpy())

    if len(all_geodesic_distances) == 0:
        print('Warning: No valid geodesic distances computed')
        print('Warning: No valid geodesic distances computed', file=file, flush=True)
        return {'mean': float('nan'), 'std': float('nan'), 'median': float('nan')}

    # Concatenate all distances
    all_distances = np.concatenate(all_geodesic_distances)

    mean_geo = np.mean(all_distances)
    std_geo = np.std(all_distances)
    median_geo = np.median(all_distances)

    print(f'---> Geodesic Distance (degrees): Mean: {mean_geo:.4f}, Std: {std_geo:.4f}, Median: {median_geo:.4f}')
    print(f'---> Geodesic Distance (degrees): Mean: {mean_geo:.4f}, Std: {std_geo:.4f}, Median: {median_geo:.4f}', file=file, flush=True)

    return {'mean': mean_geo, 'std': std_geo, 'median': median_geo}


def get_metric_statistics(values, replication_times):
    mean = np.mean(values, axis=0)
    std = np.std(values, axis=0)
    conf_interval = 1.96 * std / np.sqrt(replication_times)
    return mean, conf_interval


def evaluation(eval_wrapper, gt_loader, eval_motion_loaders, log_file, replication_times,
               diversity_times, mm_num_times, run_mm=False, eval_platform=None, num_samples_limit=None,
               run_geodesic=True):
    with open(log_file, 'w') as f:
        all_metrics = OrderedDict({'Matching Score': OrderedDict({}),
                                   'R_precision': OrderedDict({}),
                                   'FID': OrderedDict({}),
                                   'Diversity': OrderedDict({}),
                                   'MultiModality': OrderedDict({}),
                                   'Geodesic': OrderedDict({})})
        for replication in range(replication_times):
            motion_loaders = {}
            mm_motion_loaders = {}
            motion_loaders['ground truth'] = gt_loader
            for motion_loader_name, motion_loader_getter in eval_motion_loaders.items():
                motion_loader, mm_motion_loader = motion_loader_getter()
                motion_loaders[motion_loader_name] = motion_loader
                mm_motion_loaders[motion_loader_name] = mm_motion_loader

            print(f'==================== Replication {replication} ====================')
            print(f'==================== Replication {replication} ====================', file=f, flush=True)
            print(f'Time: {datetime.now()}')
            print(f'Time: {datetime.now()}', file=f, flush=True)
            mat_score_dict, R_precision_dict, acti_dict = evaluate_matching_score(eval_wrapper, motion_loaders, f)

            print(f'Time: {datetime.now()}')
            print(f'Time: {datetime.now()}', file=f, flush=True)
            fid_score_dict = evaluate_fid(eval_wrapper, gt_loader, acti_dict, f, num_samples_limit=num_samples_limit)

            print(f'Time: {datetime.now()}')
            print(f'Time: {datetime.now()}', file=f, flush=True)
            div_score_dict = evaluate_diversity(acti_dict, f, diversity_times)

            if run_mm:
                print(f'Time: {datetime.now()}')
                print(f'Time: {datetime.now()}', file=f, flush=True)
                mm_score_dict = evaluate_multimodality(eval_wrapper, mm_motion_loaders, f, mm_num_times)

            # Evaluate geodesic distance for rotation quality
            geo_score_dict = {}
            if run_geodesic:
                print(f'Time: {datetime.now()}')
                print(f'Time: {datetime.now()}', file=f, flush=True)
                for model_name, gen_loader in motion_loaders.items():
                    if model_name != 'ground truth':
                        geo_result = evaluate_geodesic(gt_loader, gen_loader, f)
                        geo_score_dict[model_name] = geo_result['mean']

            print(f'!!! DONE !!!')
            print(f'!!! DONE !!!', file=f, flush=True)

            for key, item in mat_score_dict.items():
                if key not in all_metrics['Matching Score']:
                    all_metrics['Matching Score'][key] = [item]
                else:
                    all_metrics['Matching Score'][key] += [item]

            for key, item in R_precision_dict.items():
                if key not in all_metrics['R_precision']:
                    all_metrics['R_precision'][key] = [item]
                else:
                    all_metrics['R_precision'][key] += [item]

            for key, item in fid_score_dict.items():
                if key not in all_metrics['FID']:
                    all_metrics['FID'][key] = [item]
                else:
                    all_metrics['FID'][key] += [item]

            for key, item in div_score_dict.items():
                if key not in all_metrics['Diversity']:
                    all_metrics['Diversity'][key] = [item]
                else:
                    all_metrics['Diversity'][key] += [item]
            if run_mm:
                for key, item in mm_score_dict.items():
                    if key not in all_metrics['MultiModality']:
                        all_metrics['MultiModality'][key] = [item]
                    else:
                        all_metrics['MultiModality'][key] += [item]

            if run_geodesic:
                for key, item in geo_score_dict.items():
                    if key not in all_metrics['Geodesic']:
                        all_metrics['Geodesic'][key] = [item]
                    else:
                        all_metrics['Geodesic'][key] += [item]

        # print(all_metrics['Diversity'])
        mean_dict = {}
        for metric_name, metric_dict in all_metrics.items():
            print('========== %s Summary ==========' % metric_name)
            print('========== %s Summary ==========' % metric_name, file=f, flush=True)
            for model_name, values in metric_dict.items():
                # print(metric_name, model_name)
                mean, conf_interval = get_metric_statistics(np.array(values), replication_times)
                mean_dict[metric_name + '_' + model_name] = mean
                # print(mean, mean.dtype)
                if isinstance(mean, np.float64) or isinstance(mean, np.float32):
                    print(f'---> [{model_name}] Mean: {mean:.4f} CInterval: {conf_interval:.4f}')
                    print(f'---> [{model_name}] Mean: {mean:.4f} CInterval: {conf_interval:.4f}', file=f, flush=True)
                elif isinstance(mean, np.ndarray):
                    line = f'---> [{model_name}]'
                    for i in range(len(mean)):
                        line += '(top %d) Mean: %.4f CInt: %.4f;' % (i+1, mean[i], conf_interval[i])
                    print(line)
                    print(line, file=f, flush=True)
                    
        # log results
        if eval_platform is not None:
            for k, v in mean_dict.items():
                if k.startswith('R_precision'):
                    for i in range(len(v)):
                        eval_platform.report_scalar(name=f'top{i + 1}_' + k, value=v[i],
                                                            iteration=1, group_name='Eval')
                else:
                    eval_platform.report_scalar(name=k, value=v, iteration=1, group_name='Eval')
        
        return mean_dict


if __name__ == '__main__':
    args = evaluation_parser()
    fixseed(args.seed)
    args.batch_size = 32 # This must be 32! Don't change it! otherwise it will cause a bug in R precision calc!
    name = os.path.basename(os.path.dirname(args.model_path))
    niter = os.path.basename(args.model_path).replace('model', '').replace('.pt', '')
    log_name = 'eval_humanml_{}_{}'.format(name, niter)
    if args.guidance_param != 1.:
        log_name += f'_gscale{args.guidance_param}'
    log_name += f'_{args.eval_mode}'
    log_file = os.path.join(os.path.dirname(args.model_path), log_name + '.log')
    save_dir = os.path.dirname(log_file)  # has not been tested with WandB

    print(f'Will save to log file [{log_file}]')

    eval_platform_type = eval(args.train_platform_type)
    eval_platform = eval_platform_type(save_dir, name=log_name)
    eval_platform.report_args(args, name='Args')

    print(f'Eval mode [{args.eval_mode}]')
    if args.eval_mode == 'debug':
        num_samples_limit = 1000  # None means no limit (eval over all dataset)
        run_mm = False
        mm_num_samples = 0
        mm_num_repeats = 0
        mm_num_times = 0
        diversity_times = 300
        replication_times = 5  # about 3 Hrs
    elif args.eval_mode == 'wo_mm':
        num_samples_limit = 1000
        run_mm = False
        mm_num_samples = 0
        mm_num_repeats = 0
        mm_num_times = 0
        diversity_times = 300
        replication_times = 20 # about 12 Hrs
    elif args.eval_mode == 'mm_short':
        num_samples_limit = 1000
        run_mm = True
        mm_num_samples = 100
        mm_num_repeats = 30
        mm_num_times = 10
        diversity_times = 300
        replication_times = 5  # about 15 Hrs
    else:
        raise ValueError()


    dist_util.setup_dist(args.device)
    logger.configure()

    logger.log("creating data loader...")
    split = 'test'
    gt_loader = get_dataset_loader(name=args.dataset, batch_size=args.batch_size, num_frames=None, split=split, hml_mode='gt')
    # gen_loader = get_dataset_loader(name=args.dataset, batch_size=args.batch_size, num_frames=None, split=split, hml_mode='eval')
    # added new features + support for prefix completion:
    gen_loader = get_dataset_loader(name=args.dataset, batch_size=args.batch_size, num_frames=None, split=split, hml_mode='eval',
                                    fixed_len=args.context_len+args.pred_len, pred_len=args.pred_len, device=dist_util.dev(),
                                    autoregressive=args.autoregressive)

    num_actions = gen_loader.dataset.num_actions

    logger.log("Creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(args, gen_loader)

    logger.log(f"Loading checkpoints from [{args.model_path}]...")
    load_saved_model(model, args.model_path, use_avg=args.use_ema)

    if args.guidance_param != 1:
        model = ClassifierFreeSampleModel(model)   # wrapping model with the classifier-free sampler
    model.to(dist_util.dev())
    model.eval()  # disable random masking

    eval_motion_loaders = {
        ################
        ## HumanML3D Dataset##
        ################
        'vald': lambda: get_mdm_loader(args,
            model=model, diffusion=diffusion, batch_size=args.batch_size,
            ground_truth_loader=gen_loader, mm_num_samples=mm_num_samples, mm_num_repeats=mm_num_repeats, 
            max_motion_length=gt_loader.dataset.opt.max_motion_length, num_samples_limit=num_samples_limit, 
            scale=args.guidance_param
        )
    }

    eval_wrapper = EvaluatorMDMWrapper(args.dataset, dist_util.dev())
    evaluation(eval_wrapper, gt_loader, eval_motion_loaders, log_file, replication_times, 
               diversity_times, mm_num_times, run_mm=run_mm, eval_platform=eval_platform, num_samples_limit=num_samples_limit)
    eval_platform.close()
