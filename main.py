import gc
import pandas as pd
import torch
import unsloth
from datasets import load_dataset
from nl2query import MongoQuery
from peft import PeftModel
from rouge_score import rouge_scorer
import sacrebleu
from tqdm import tqdm
from transformers import AutoTokenizer
from unsloth import FastLanguageModel
import bert_score

from submodules.clean_table import get_header_row_index, infer_column_types, is_aggregate_field, is_pure_string, flatten_aggregated_table, find_matching_field_row
from submodules.subquestion_generation import get_subquestions_at_index, generate_subquestion_tokens_batched
from submodules.merge_answers import cache_table_tokens, answer_subquestions_batched, merge_subanswers


ds = load_dataset("DongfuJiang/FeTaQA")
train, val, test = ds["train"], ds['validation'], ds['test']

base_model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Llama-3.2-3B-Instruct-bnb-4bit",
    max_seq_length = 2048,
    load_in_4bit = True,
    load_in_8bit = False,
    full_finetuning = False,
    dtype=torch.bfloat16,
    device_map="auto"
)


subquestion_model = PeftModel.from_pretrained(base_model, './SubquestionModel')
subquestion_tokenizer = AutoTokenizer.from_pretrained('./SubquestionModel')

table_model, table_tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit",
    #max_seq_length = 2048,
    load_in_4bit = True,
    load_in_8bit = False,
    full_finetuning = False,
    dtype=torch.bfloat16,
    device_map="auto"
)


def answer_question_at_index(dataset, idx):
    with torch.inference_mode():
        df = pd.DataFrame(dataset[idx]['table_array'])
        header_idx, header = get_header_row_index(df, base_model, tokenizer)
        df = df[header_idx+1:].reset_index(drop=True)
        if df.shape[0] < 1:
            return ""
        else:
            df.columns = header
            
        
        col_types = dict(infer_column_types(df))
        str_mixed_indexes = [i for i in col_types.keys() if col_types[i] == 'string' or col_types[i] == 'mixed']
        name_to_index = {name: df.columns.get_loc(name) for name in str_mixed_indexes}
        possible_fields = [df[1:].iloc[:, name_to_index[name]].unique() for name in str_mixed_indexes]
        filtered_fields = [item for array in possible_fields for item in array if is_pure_string(item)]
        flattened_possible_fields = set(filtered_fields)
        
        aggregate_fields = []
        for field in flattened_possible_fields:
            if is_aggregate_field(field, base_model, tokenizer):
                aggregate_fields.append(field)
        
        filtered_dfs = []
        seen_rows = set()
        
        # Create fingerprints for the full df once
        row_fingerprints = df.astype(str).agg('|'.join, axis=1)
        
        for field in aggregate_fields:
            filtered_df = df[df.applymap(lambda x: field in str(x)).any(axis=1)].copy()
            
            filtered_fp = filtered_df.astype(str).agg('|'.join, axis=1)
            new_mask = ~filtered_fp.isin(seen_rows)
        
            new_rows = filtered_df[new_mask]
            seen_rows.update(filtered_fp[new_mask])
        
            filtered_dfs.append(new_rows)
        
        if filtered_dfs:
            merged_df = pd.concat(filtered_dfs).reset_index(drop=True)
        else:
            # Fallback: use an empty DataFrame with same columns as df
            merged_df = pd.DataFrame(columns=df.columns)
        
        # Reassign headers cleanly (assuming first row of original df is header)
        merged_df.columns = header
        
        # Compute df_remaining using same fast fingerprints
        full_df_fp = df.astype(str).agg('|'.join, axis=1)
        remaining_mask = ~full_df_fp.isin(seen_rows)
        df_remaining = df[remaining_mask].reset_index(drop=True)
        df_remaining.columns = header
        if 0 in df_remaining.index:
            df_remaining.drop(index=[0], inplace=True)
        merged_df = flatten_aggregated_table(merged_df, aggregate_fields)
        
        keys = list(df_remaining.columns)
        row_texts = merged_df.astype(str).agg(' '.join, axis=1)
        matched_fields = row_texts.apply(lambda text: find_matching_field_row(text, aggregate_fields))
        matched_fields = matched_fields.dropna().astype(str).tolist()
        keys.extend(matched_fields)
        keys = list(set(keys))
        
        subquestions = get_subquestions_at_index(dataset, idx, keys, subquestion_model, subquestion_tokenizer)
        if 'index' not in keys:
            keys = keys + ['index']
        keys = [k for k in keys if isinstance(k, str)]
        queryfier = MongoQuery('T5', collection_keys=keys, collection_name='exec')
        mongo_subquestions = [queryfier.generate_query(qn) for qn in subquestions]
        
        title = dataset[idx]['table_page_title'] + ' ' + dataset[idx]['table_section_title']
        cached_table_tokens = cache_table_tokens(
            df_remaining, merged_df, title,
            table_tokenizer, table_model.device
        )
        subq_inputs = generate_subquestion_tokens_batched(
            cached_table_tokens,
            subquestions,
            mongo_subquestions,
            table_tokenizer,
            table_model.device
        )
        
        answers = answer_subquestions_batched(subq_inputs, table_model, table_tokenizer)
        
        final_ans = merge_subanswers(dataset[idx]['question'], subquestions, answers, base_model, tokenizer)

    del df, df_remaining, merged_df, row_texts
    del cached_table_tokens, subq_inputs, answers, mongo_subquestions
    del queryfier, matched_fields, keys, subquestions
    gc.collect()
    torch.cuda.empty_cache()

    return final_ans


def evaluate(predictions, references):
    assert len(predictions) == len(references), "Mismatch between number of predictions and references"

    sacrebleu_score = sacrebleu.corpus_bleu(predictions, [references])
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    rouge1_list, rouge2_list, rougeL_list = [], [], []

    for pred, ref in zip(predictions, references):
        scores = scorer.score(ref, pred)
        rouge1_list.append(scores['rouge1'].fmeasure)
        rouge2_list.append(scores['rouge2'].fmeasure)
        rougeL_list.append(scores['rougeL'].fmeasure)

    avg_rouge1 = sum(rouge1_list) / len(rouge1_list)
    avg_rouge2 = sum(rouge2_list) / len(rouge2_list)
    avg_rougeL = sum(rougeL_list) / len(rougeL_list)

    P, R, F1 = bert_score.score(predictions, references, lang="en", rescale_with_baseline=True)
    avg_bertscore_f1 = F1.mean().item()

    results = {
        "sacrebleu": sacrebleu_score.score,
        "rouge1": avg_rouge1,
        "rouge2": avg_rouge2,
        "rougeL": avg_rougeL,
        "bertscore_f1": avg_bertscore_f1
    }

    return results


if __name__ == "__main__":
    with open('test.txt', 'a') as f:
        for i in tqdm(range(len(test))):
            try:
                answer = answer_question_at_index(test, i)
                if not answer or not isinstance(answer, str):
                    answer = ""
                
                f.write(answer.strip().replace('\n', ' ') + '\n')
            except Exception as e:
                print(f"[ERROR] Failed at index {i}: {e}")
                f.write('\n')