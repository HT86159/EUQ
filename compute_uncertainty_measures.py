"""Compute uncertainty measures after generating answers."""
from collections import defaultdict
import logging
import os
import pickle
import numpy as np
import wandb

import torch
from analyze_results import analyze_run
from uncertainty.data.data_utils import load_ds
from uncertainty.uncertainty_measures.envidence import EvidenceModel
from uncertainty.utils import utils

from tqdm import tqdm
utils.setup_logger()
from transformers import AutoTokenizer, AutoModelForCausalLM
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EXP_DETAILS = 'experiment_details.pkl'


def main(args):

    if args.train_wandb_runid is None:
        args.train_wandb_runid = args.eval_wandb_runid
    user = os.environ['USER']
    scratch_dir = os.getenv('SCRATCH_DIR', '.')
    wandb_dir = f'{scratch_dir}/{user}/uncertainty'
    slurm_jobid = os.getenv('SLURM_JOB_ID', None)
    project = "EUQ" if not args.debug else "EUQ_debug"
    if args.assign_new_wandb_id:
        logging.info('Assign new wandb_id.')
        api = wandb.Api()
        old_run = api.run(f'{args.restore_entity_eval}/{project}/{args.eval_wandb_runid}')
        wandb.init(
            entity=args.entity,
            project=project,
            dir=wandb_dir,
            notes=f'slurm_id: {slurm_jobid}, experiment_lot: {args.experiment_lot}',
            # For convenience, keep any 'generate_answers' configs from old run,
            # but overwrite the rest!
            # NOTE: This means any special configs affecting this script must be
            # called again when calling this script!
            config={**old_run.config, **args.__dict__},
        )

        def restore(filename):
            old_run.file(filename).download(
                replace=True, exist_ok=False, root=wandb.run.dir)

            class Restored:
                name = f'{wandb.run.dir}/{filename}'

            return Restored
    else:
        logging.info('Reuse active wandb id.')

        def restore(filename):
            class Restored:
                name = f'{wandb.run.dir}/{filename}'
            return Restored

    if args.train_wandb_runid != args.eval_wandb_runid:
        logging.info(
            "Distribution shift for p_ik. Training on embeddings from run %s but evaluating on run %s",
            args.train_wandb_runid, args.eval_wandb_runid)

        is_ood_eval = True  # pylint: disable=invalid-name
        api = wandb.Api()
        old_run_train = api.run(f'{args.restore_entity_train}/EUQ/{args.train_wandb_runid}')
        filename = 'train_generations.pkl'
        old_run_train.file(filename).download(
            replace=True, exist_ok=False, root=wandb.run.dir)
        with open(f'{wandb.run.dir}/{filename}', "rb") as infile:
            train_generations = pickle.load(infile)
        wandb.config.update(
            {"ood_training_set": old_run_train.config['dataset']}, allow_val_change=True)
    else:
        is_ood_eval = False  # pylint: disable=invalid-name
    wandb.config.update({"is_ood_eval": is_ood_eval}, allow_val_change=True)

    if args.recompute_accuracy:
        # This is usually not enabled.
        logging.warning('Recompute accuracy enabled. This does not apply to precomputed p_true!')
        metric = utils.get_metric(args.metric)
    # Restore outputs from `generate_answrs.py` run.
    result_dict_pickle = restore('uncertainty_measures.pkl')
    with open(result_dict_pickle.name, "rb") as infile:
        result_dict = pickle.load(infile)
    result_dict['type'] = []


    validation_generations_pickle = restore('validation_generations.pkl')
    with open(validation_generations_pickle.name, 'rb') as infile:
        validation_generations = pickle.load(infile)

    entropies = defaultdict(list)
    validation_embeddings, validation_is_true, validation_answerable, validation_llm_head_inputs, validation_down_proj_inputs, responses_list = [], [], [], [], [], []
    
    count = 0 

    def is_answerable(generation):
        return len(generation['reference']['answers']['text']) > 0

    if args.model_name == 'Qwen2.5':
        state_dict = torch.load("../weights/Qwen2.5_attention_weights.pth")
        head_state_dict = torch.load("../weights/Qwen2.5_head_weights.pth")
    evidence_model = EvidenceModel(state_dict) 
    head_evidence_model = EvidenceModel(head_state_dict) 

    processed_feature = []
    processed_feature_llm = []

    for idx, tid in tqdm(enumerate(validation_generations)):
        example = validation_generations[tid]
        question = example['question']
        context = example['context']
        full_responses = example["responses"]
        most_likely_answer = example['most_likely_answer']
        result_dict['type'].append(example['reference']['id'])
        
        if not args.use_all_generations:
            if args.use_num_generations == -1:
                raise ValueError
            responses = [fr[0] for fr in full_responses[:args.use_num_generations]]
        else:
            responses = [fr[0] for fr in full_responses]

        if args.recompute_accuracy:
            logging.info('Recomputing accuracy!')
            if is_answerable(example):
                acc = metric(most_likely_answer['response'], example, None)
            else:
                acc = 0.0  
            validation_is_true.append(acc)
            logging.info('Recomputed accuracy!')

        else:
            validation_is_true.append(most_likely_answer['accuracy'])

        validation_answerable.append(is_answerable(example))
        validation_embeddings.append(most_likely_answer['embedding'])
        responses_list.append(responses)
        validation_down_proj_inputs.append(most_likely_answer['down_proj_inputs'])
        validation_llm_head_inputs.append(most_likely_answer['llm_head_inputs'])
        logging.info('validation_is_true: %f', validation_is_true[-1])
        logging.info('validation_is_true: %f', validation_is_true[-1])
        if args.compute_predictive_entropy:
            
            conflict_value = 0
            ig_value = 0
            head_conflict_value = 0
            head_ig_value = 0

            for feature in most_likely_answer['down_proj_inputs']:
                processed_feature.append(feature.squeeze(0))
            for feature in most_likely_answer['llm_head_inputs']:
                processed_feature_llm.append(feature)

            for feature in most_likely_answer['down_proj_inputs']:
                evidence_weights = evidence_model.get_evidence_weights(feature.squeeze(0).T)
                conflict_value += evidence_model.get_evidence_conflict()
                ig_value += evidence_model.get_evidence_ignorance()

            for feature in most_likely_answer['llm_head_inputs']:
                head_evidence_weights = head_evidence_model.get_evidence_weights(feature.T)
                head_conflict_value += head_evidence_model.get_evidence_conflict()
                head_ig_value += head_evidence_model.get_evidence_ignorance()

            entropies['attention_conflict'].append(conflict_value.item()/len(most_likely_answer['down_proj_inputs']))
            entropies['attention_ignorance'].append(ig_value.item()/len(most_likely_answer['down_proj_inputs']))
            entropies['llm_head_conflict'].append(head_conflict_value.item()/len(most_likely_answer['llm_head_inputs']))
            entropies['llm_head_ignorance'].append(head_ig_value.item()/len(most_likely_answer['llm_head_inputs']))

            log_str = 'entropies: %s'
            entropies_fmt = ', '.join([f'{i}:{j[-1]:.2f}' for i, j in entropies.items()])

            logging.info(80*'#')
            logging.info('NEW ITEM %d at id=`%s`.', idx, tid)
            logging.info('Context:')
            logging.info(example['context'])
            logging.info('Question:')
            logging.info(question)
            logging.info('True Answers:')
            logging.info(example['reference'])
            logging.info('Low Temperature Generation:')
            logging.info(most_likely_answer['response'])
            logging.info('Low Temperature Generation Accuracy:')
            logging.info(most_likely_answer['accuracy'])
            logging.info('High Temp Generation:')
            logging.info([r[0] for r in full_responses])
            logging.info('High Temp Generation:')
            logging.info(log_str, entropies_fmt)

        count += 1
        if count >= args.num_eval_samples:
            logging.info('Breaking out of main loop.')
            break

    logging.info('Accuracy on original task: %f', np.mean(validation_is_true))
    validation_is_false = [1.0 - is_t for is_t in validation_is_true]
    result_dict['validation_is_false'] = validation_is_false

    validation_unanswerable = [1.0 - is_a for is_a in validation_answerable]
    result_dict['validation_unanswerable'] = validation_unanswerable
    logging.info('Unanswerable prop on validation: %f', np.mean(validation_unanswerable))

    if 'uncertainty_measures' not in result_dict:
        result_dict['uncertainty_measures'] = dict()

    if args.compute_predictive_entropy:
        result_dict['uncertainty_measures'].update(entropies)
    
    utils.save(result_dict, 'uncertainty_measures.pkl')
    logging.info('Finished computing uncertainty measures.')

    if args.analyze_run:
        # Follow up with computation of aggregate performance metrics.
        logging.info(50 * '#X')
        logging.info('STARTING `analyze_run`!')
        analyze_run(wandb.run.id)
        logging.info(50 * '#X')
        logging.info('FINISHED `analyze_run`!')


if __name__ == '__main__':
    parser = utils.get_parser(stages=['compute'])
    args, unknown = parser.parse_known_args()
    if unknown:
        raise ValueError(f'Unkown args: {unknown}')

    logging.info("Args: %s", args)

    main(args)
