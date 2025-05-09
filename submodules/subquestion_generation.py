import re

def table_to_markdown(table_array, title):
    markdown = f"### {title}\n\n"
    markdown += "| " + " | ".join(table_array[0]) + " |\n"
    markdown += "| " + " | ".join(["---"]*len(table_array[0])) + " |\n"
    for row in table_array[1:]:
        markdown += "| " + " | ".join(row) + " |\n"
    return markdown

def get_subquestions(question, table_data, table_title, model, tokenizer, num_questions=3):
    input_text = f"""\
{table_to_markdown(table_data, table_title)}

INSTRUCTION:
Do not answer the question (or any questions).
Decompose the following main question into exactly 3 non-overlapping, highly specific subquestions which, together, fully answer the main question.

Guidelines:
- Each subquestion must cover a *distinct* component or temporal segment.
- Focus on any time-related elements like "before", "after", "during", "change over time", etc.
- Split multi-step or comparative reasoning clearly.
- Avoid rephrasing or redundant overlap.
- Keep each subquestion specific, direct, and concise.
- Output only 3 numbered subquestions. No explanations.

Example:

Main Question:
How did the policies on renewable energy change before and after the Kyoto Protocol?

Subquestions:
1. What were the major renewable energy policies in place before the Kyoto Protocol?
2. What policy changes occurred as a direct result of the Kyoto Protocol?
3. How did global renewable energy strategies evolve after the Kyoto Protocol was implemented?

Example:

Main Question:
What TV shows was Shagun Sharma seen in 2019?

Subquestions:
1. Which TV shows featured Shagun Sharma in 2019?
2. What roles did Shagun Sharma play in 2019?
3. On which channels did Shagun Sharma air in 2019?


Now your turn.

Main Question:
{question}

Subquestions:
"""

    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, padding='longest').to(model.device)
    outputs = model.generate(
        input_ids=inputs.input_ids,
        max_new_tokens=256,
        num_beams=1,
        do_sample=False,
        early_stopping=True,
    )

    output_text = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

    generated_text = output_text[len(input_text):].strip()

    subquestions = []
    for line in generated_text.splitlines():
        if re.match(r'^\d+\.\s*', line):  # only process numbered lines
            match = re.match(r'^\d+\.\s*(.*)', line)
            if match:
                subq = match.group(1).strip()
                subquestions.append(subq)
    
    if len(subquestions) != 3:
        print(f"[WARN] Got {len(subquestions)} subquestions for: {question}")
        subquestions = subquestions[:3]  # Truncate if too many
        subquestions += [""] * (3 - len(subquestions))  # Pad if too few

    return subquestions

def get_subquestions_at_index(split, index, keys, subquestion_model, subquestion_tokenizer):
    return get_subquestions(split[index]['question'], 
                            split[index]['table_array'], 
                            split[index]['table_page_title'] + ' ' + split[index]['table_section_title'], 
                            subquestion_model, subquestion_tokenizer) + [split[index]['question']]