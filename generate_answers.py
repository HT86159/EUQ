import gc
import os
import logging
import random
from tqdm import tqdm

import numpy as np
import wandb

from uncertainty.data.data_utils import load_ds
from uncertainty.utils import utils
from compute_uncertainty_measures import main as main_compute
import torch
utils.setup_logger()
import sys
sys.path.insert(0, "../uncertainty/uncertainty_measures")

def main(args):
    # Setup run.
    experiment_details = {'args': args}
    random.seed(args.random_seed)
    user = os.environ['USER']
    slurm_jobid = os.getenv('SLURM_JOB_ID', None)
    scratch_dir = os.getenv('SCRATCH_DIR', '.')
    if not os.path.exists(f"{scratch_dir}/{user}/uncertainty"):
        os.makedirs(f"{scratch_dir}/{user}/uncertainty")

    wandb.init(
        entity=args.entity,
        project="EUQ" if not args.debug else "EUQ_debug",
        dir=f"{scratch_dir}/{user}/uncertainty",
        config=args,
        notes=f'slurm_id: {slurm_jobid}, experiment_lot: {args.experiment_lot}',
    )
    logging.info('Finished wandb init.')
    metric = utils.get_metric('squad')
    train_dataset, validation_dataset = load_ds(
        args.dataset, add_options=args.use_mc_options, seed=args.random_seed)
    if args.ood_train_dataset is not None:
        logging.warning(
            'Using OOD dataset %s to construct few-shot prompts and train p_ik.',
            args.ood_train_dataset)
        # Get indices of answerable and unanswerable questions and construct prompt.
        train_dataset, _ = load_ds(args.ood_train_dataset, add_options=args.use_mc_options)
    if not isinstance(train_dataset, list):
        logging.info('Train dataset: %s', train_dataset)

    # Get indices of answerable and unanswerable questions and construct prompt.
    answerable_indices, unanswerable_indices = utils.split_dataset(train_dataset)
    if args.answerable_only:
        unanswerable_indices = []
        val_answerable, val_unanswerable = utils.split_dataset(validation_dataset)
        del val_unanswerable
        validation_dataset = [validation_dataset[i] for i in val_answerable]

    prompt_indices = random.sample(answerable_indices, args.num_few_shot)
    experiment_details['prompt_indices'] = prompt_indices
    remaining_answerable = list(set(answerable_indices) - set(prompt_indices))

    # Create Few-Shot prompt.
    make_prompt = utils.get_make_prompt(args)
    BRIEF = utils.BRIEF_PROMPTS[args.brief_prompt]
    arg = args.brief_always if args.enable_brief else True
    prompt = ""
    experiment_details['prompt'] = prompt
    experiment_details['BRIEF'] = BRIEF
    logging.info('Prompt is: %s', prompt)

    # Initialize model.
    model = utils.init_model(args)
    if isinstance(model, tuple):
        tokenizer, model, image_processor, context_len = model

    def down_proj_hook(module, input, output):
        intermediate = input[0]
        down_proj_features.append(intermediate.detach().cpu())
    
    def lm_head_hook(module, input, output):
        full_hidden = input[0].detach().cpu()
        last_token_hidden = full_hidden[:, -1, :]
        llm_head_features.append(last_token_hidden)

    # import pdb;pdb.set_trace()
    weights_path = "../weights"
    head_weights_path = f"{weights_path}/{args.model_name}_head_weights.pth"
    attention_weights_path = f"{weights_path}/{args.model_name}_attention_weights.pth"
    if not os.path.exists(weights_path):
        import torch
        os.makedirs(weights_path, exist_ok=True)
        head_weight = model.lm_head.weight
        head_weight_cpu = head_weight.cpu()
        torch.save(head_weight_cpu, head_weights_path)
        # attention_weight = model.model.layers[0].mlp.down_proj.weight
        attention_weight = model.model.language_model.layers[0].mlp.down_proj.weight
        attention_weight_cpu = attention_weight.cpu()
        torch.save(attention_weight_cpu, attention_weights_path)

    # Start answer generation.
    logging.info(80 * '=')
    logging.info('Generating answers: ')
    logging.info(80 * '=')
    # for dataset_split in ['train', 'validation']:
    for dataset_split in ['validation']:
        logging.info(80 * 'x')
        logging.info('Starting with dataset_split %s.', dataset_split)
        logging.info(80 * 'x')

        # This will store all input data and model predictions.
        accuracies, generations, results_dict, p_trues = [], {}, {}, []

        if dataset_split == 'train':
            if not args.get_training_set_generations:
                logging.info('Skip training data.')
                continue
            dataset = train_dataset
            possible_indices = list(set(remaining_answerable) | set(unanswerable_indices))

        else:
            dataset = validation_dataset
            if(args.dataset=="mmvp"):
                possible_indices = range(0, len(dataset)) 
            else:
                possible_indices = [item['id'] for item in dataset]

        # Evaluate over random subset of the datasets.
        print(len(dataset))
        indices = random.sample(possible_indices, min(args.num_samples, len(possible_indices)))
        experiment_details[dataset_split] = {'indices': indices}

        if args.num_samples > len(dataset):
            logging.warning('Not enough samples in dataset. Using all %d samples.', len(dataset))

        it = 0
        all_ans = 0
        for index in tqdm(indices):
            if (it + 1 % 10) == 0:
                gc.collect()
                torch.cuda.empty_cache()
            it += 1
            for item in dataset:
                if item['id'] == index:
                    found_item = item

            # Grab example at index.
            example = found_item
            example['id'] =  str(example['id']) 
            question, context = example["question"], example['context']
            quest_id = str(example["question"])
            rural_example_id = example['id']
            if args.dataset == "hallucination":
                example['id'] += quest_id 
            generations[example['id']] = {'question': question, 'context': context}
            example['id'] = rural_example_id
            correct_answer = example['answers']['text']
            if args.model_name == "Qwen2.5":
                from qwen_vl_utils import process_vision_info
                current_input = make_prompt(context, question, None, BRIEF, args.brief_always and args.enable_brief)
                local_prompt = prompt + current_input
                if args.dataset == "hallucination":
                    image_path = os.path.join(args.image_path, 'images', example['id'])
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image":image_path},
                            {"type": "text", "text": local_prompt}
                        ]
                    }
                ]
                text = image_processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                image_inputs, video_inputs = process_vision_info(messages)
                inputs = image_processor(
                    text=[text],
                    images=image_inputs,
                    videos=video_inputs,
                    padding=True,
                    return_tensors="pt",
                )
                inputs = inputs.to(model.device)

            if args.dataset == "hallucination":
                example['id'] += quest_id
            logging.info('Current input: '.ljust(15) + local_prompt)


            full_responses = []

            if dataset_split == 'train' and args.get_training_set_generations_most_likely_only:
                num_generations = 1
            else:
                num_generations = args.num_generations + 1
            for i in range(num_generations):
                import torch
                temperature = 0.1 if i == 0 else args.temperature

                if args.model_name=="Qwen2.5":
                    down_proj_features = []
                    llm_head_features = []
                    down_proj_handle = model.model.language_model.layers[0].mlp.down_proj.register_forward_hook(down_proj_hook)
                    lm_head_handle = model.lm_head.register_forward_hook(lm_head_hook)
                    output_ids = model.generate(
                        **inputs, 
                        max_new_tokens=128, 
                        output_hidden_states=True,
                        do_sample=True,
                        temperature=temperature,
                        )
                    generated_ids_trimmed = [
                        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, output_ids)
                    ]

                    outputs = image_processor.batch_decode(
                        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                    )

                    outputs= ''.join(map(str, outputs))
                    if outputs.endswith('</s>'):
                        outputs = outputs[:-len('</s>')]
                    
                    predicted_answer = outputs.strip()

                    embedding = None

                down_proj_features = [x[:,-1:,:].cpu() for x in down_proj_features]

                llm_head_feature1 = []
                for input in llm_head_features:
                    if(input.dim() == 3):
                        input = input.unsqueeze(0)
                    llm_head_feature1.append(input.cpu())
                llm_head_features = llm_head_feature1

                down_proj_handle.remove()
                lm_head_handle.remove()
                del llm_head_feature1

                embedding = embedding.cpu() if embedding is not None else None
                compute_acc = args.compute_accuracy_at_all_temps or (i == 0)
                if len(predicted_answer)==0:
                    continue
                if correct_answer and compute_acc:
                    acc = metric(predicted_answer, example, model)
                else:
                    acc = 0.0

                if i == 0:
                    if acc==1:
                        all_ans+=1
                    logging.info('Iteration ' + str(it) + ':  ' + 80*'#')
                    if args.use_context:
                        logging.info('context: '.ljust(15) + str(context))
                    logging.info('question: '.ljust(15) + question)
                    logging.info('low-t prediction: '.ljust(15) + predicted_answer)
                    logging.info('correct answer: '.ljust(15) + str(correct_answer))
                    logging.info('accuracy: '.ljust(15) + str(acc))

                    accuracies.append(acc)
                    most_likely_answer_dict = {
                        'response': predicted_answer,
                        'embedding': embedding,
                        'accuracy': acc,
                        'down_proj_inputs': down_proj_features,
                        'llm_head_inputs': llm_head_features,
                        }
                    generations[example['id']].update({
                        'most_likely_answer': most_likely_answer_dict,
                        'reference': utils.get_reference(example)})
                else:
                    logging.info('high-t prediction '.ljust(15) + str(i) + ' : ' + predicted_answer)
                    # Aggregate predictions over num_generations.
                    full_responses.append(
                        (predicted_answer, embedding, acc))


            # Append all predictions for this example to `generations`.
            generations[example['id']]['responses'] = full_responses

        # Save generations for that split.
        utils.save(generations, f'{dataset_split}_generations.pkl')

        # Log overall accuracy.
        accuracy = np.mean(accuracies)
        print(f"Overall {dataset_split} split accuracy: {accuracy}")
        wandb.log({f"{dataset_split}_accuracy": accuracy})

        if dataset_split == 'validation':
            utils.save(results_dict, 'uncertainty_measures.pkl')
    utils.save(experiment_details, 'experiment_details.pkl')
    logging.info('Run complete.')
    del model
    import gc
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == '__main__':
    parser = utils.get_parser()
    args, unknown = parser.parse_known_args()
    logging.info('Starting new run with args: %s', args)

    if unknown:
        raise ValueError(f'Unkown args: {unknown}')

    if args.compute_uncertainties:
        args.assign_new_wandb_id = False

    # First sample generations from LLM.
    logging.info('STARTING `generate_answers`!')
    main(args)
    logging.info('FINISHED `generate_answers`!')
    logging.info('run with args: %s', args)

    if args.compute_uncertainties:
        # Follow with uncertainty calculation script by default.
        args.assign_new_wandb_id = False
        gc.collect()
        torch.cuda.empty_cache()
        logging.info(50 * '#X')
        logging.info('STARTING `compute_uncertainty_measures`!')
        main_compute(args)
        logging.info('FINISHED `compute_uncertainty_measures`!')
        logging.info('run with args: %s', args)