import torch

def cache_table_tokens(table_data, aggregate_data, table_title, tokenizer, device):
    table_block = f"""
### Table Title:
{table_title}

### Table:
{table_data}

### Aggregate data:
{aggregate_data}
"""

    full_context = (
        "You are a table question-answering assistant.\n"
        "You are given a table and will be asked subquestions.\n"
        "Your answers must be short (a single number, phrase, or name), factual, and based strictly on the table." 
        "Don't give any explanation or anything else apart from that. Don\'t write down anything else, it is important to just give the answer.\n"
        "If the answer is not found, return: \"\" (two quotes).\n\n"
        + table_block
    )

    tokenized = tokenizer(full_context, return_tensors="pt", truncation=True).to(device)
    return tokenized

def generate_subquestion_tokens_batched(cached_tokens, subquestions, mongo_subquestions, tokenizer, device):
    subq_blocks = [
        f"""
### Subquestion:
{subq}

### Structured Filter (MongoDB-style):
{mongo}

Answer:"""
        for subq, mongo in zip(subquestions, mongo_subquestions)
    ]

    subq_tokens = tokenizer(subq_blocks, return_tensors="pt", padding='longest', truncation=True).to(device)
    batch_size = len(subq_blocks)

    input_ids = torch.cat([cached_tokens["input_ids"].expand(batch_size, -1), subq_tokens["input_ids"]], dim=1)
    attention_mask = torch.cat([cached_tokens["attention_mask"].expand(batch_size, -1), subq_tokens["attention_mask"]], dim=1)

    return {"input_ids": input_ids, "attention_mask": attention_mask}

def answer_subquestions_batched(subq_inputs, model, tokenizer, max_new_tokens=20):
    with torch.inference_mode():
        outputs = model.generate(
            **subq_inputs,
            do_sample=False,
            num_beams=1,
            max_new_tokens=max_new_tokens,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )

    decoded_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    answers = []
    for out in decoded_outputs:
        if "Answer:" in out:
            answer = out.split("Answer:")[-1].strip()
        else:
            answer = out.strip().split("\n")[-1]
        answers.append(answer.strip())
    return answers

def merge_subanswers(question, subquestions, subanswers, model, tokenizer):
    input_prompt = f"""You are a helpful assistant.

You are given a main question, a list of subquestions, and the answers to those subquestions.
Your task is to generate a final complete answer to the main question.
Just write a single complete sentence.

- ONLY use the information provided in the subanswers.
- DO NOT invent facts, names, years, categories, or events.
- DO NOT add your own reasoning or explanations.
- If information is missing or partial, just use whatever subanswers provide without making up anything.
- If all answers are provided, synthesize them into one clear, factual sentence.
- If only partial information is given, still produce the best factual summary from what's available.
- Never say “I can't answer.” Instead, just omit what's missing and use what you have.
- Do not include disclaimers like “I don’t have enough info.”

Here are examples:

---

Example 1

Main Question:
What roles did Victoria Justice play in Zoey 101 and Victorious, and what years did those shows air?

Subquestions and Answers:
Q1: What role did Victoria Justice play in Zoey 101?
A1: Victoria Justice played Lola Martinez in Zoey 101.
Q2: What years did Zoey 101 air?
A2: Zoey 101 aired from 2005 to 2008.
Q3: What role did Victoria Justice play in Victorious?
A3: Victoria Justice played Tori Vega in Victorious.
Q4: What years did Victorious air?
A4: Victorious aired from 2010 to 2013.

Final Answer:
Victoria Justice played Lola Martinez in Zoey 101 (2005–2008) and Tori Vega in Victorious (2010–2013).

---

Example 2

Main Question:
How did Japan perform in Group H of the 2002 FIFA World Cup?

Subquestions and Answers:
Q1: How did Japan perform against Tunisia?
A1: Japan defeated Tunisia 2-0.
Q2: How did Japan perform against Russia?
A2: Japan defeated Russia 1-0.
Q3: How did Japan perform against Belgium?
A3: Japan tied with Belgium 2-2.

Final Answer:
In Group H of the 2002 FIFA World Cup, Japan defeated Tunisia 2-0, defeated Russia 1-0, and tied with Belgium 2-2.

---

Now for your real question:

Main Question:
{question}

Subquestions and Answers:
""" + "\n".join([f"Q{i+1}: {sq}\nA{i+1}: {sa}" for i, (sq, sa) in enumerate(zip(subquestions, subanswers))]) + "\n\nFinal Answer:"

    inputs = tokenizer(input_prompt, return_tensors="pt", truncation=True, padding='longest').to(model.device)
    output = model.generate(
        **inputs,
        max_new_tokens=128,
        num_beams=1,
        do_sample=False,
        early_stopping=True,
    )

    output_text = tokenizer.decode(output[0], skip_special_tokens=True).strip()
    generated_answer = output_text[len(input_prompt):].strip()

    if 'Final Answer:' in generated_answer:
        generated_answer = generated_answer.split('Final Answer:')[-1].strip()

    if 'Question:' in generated_answer:
        generated_answer = generated_answer.split('Question:')[0].strip()

    return generated_answer