import re
import pandas as pd

def get_header_row_index(df, model, tokenizer):
    df = df.replace('-', '')
    df_numpy = df.to_numpy()
    
    header_rows = []
    prev = [''] * df.shape[1]
    # Currently we only consider the top 4 rows as being possibly header rows
    for i in range(min(df.shape[0], 3)):
        prev = [[prev[j], df_numpy[i][j]] for j in range(df.shape[1])]
        prev = [' '.join(dict.fromkeys(row)).strip() for row in prev]
        #if '' not in prev:
            #header_rows.append((i, prev))
        header_rows.append((i, prev))
    
    
    def select_header_row(header_rows, model, tokenizer):
        """
        Ask the LLM which of the given rows is the true header.
        header_rows: List[List[str]]  (each inner list is one candidate header)
        Returns: the integer index of the chosen header row.
        """
        # 1) Build the rows block
        rows_block = "\n".join(
            f"  {i}: {header_rows[i]!r}"
            for i in range(len(header_rows))
        )
    
        # 2) Strong JSON‐only prompt with one example
        prompt = f"""
    You are a table-row classifier. I will give you numbered candidate rows.
    Return exactly one value, which is
    is the integer index of the correct header row in JSON as header_index. 
    Ideally the header row is detailed and descriptive, 
    but it does not contain any actual data values 
    (like numbers, dates, or specific entries)—only the names and descriptors of each column.
    Header row is the topmost row of the table, which describes the exact content
    that every individual column contains. So for example, 'FA Cup Apps', 'FA Cup Goals' etc.
    
    Example:
    Rows:
      0: ['Party', 'Party', 'Candidate', 'Votes', '%', '±']
      1: ['Party', 'Party Republican', 'James R. Thompson (incumbent) Candidate', 'Votes 1,816,101', '% 49.44', '±']
      2: ['Party', 'Party Republican Democratic', 'Adlai Stevenson III James R. Thompson (incumbent) Candidate', '1,811,027 Votes 1,816,101', '49.30 % 49.44', '±']
    Answer:
    {{header_index: 0}}
    
    Example:
    Rows:
      0: ['', '', '', 'Regular season', 'Regular season', 'Regular season', 'Regular season', 'Regular season', 'Playoffs', 'Playoffs', 'Playoffs', 'Playoffs', 'Playoffs', '', '']
      1: ['Season', 'Team', 'League', 'Regular season', 'Regular season GP', 'Regular season G', 'A Regular season', 'Regular season Pts', 'Playoffs PIM', 'Playoffs', 'Playoffs GP', 'Playoffs G', 'A Playoffs', 'Pts', 'PIM']
      2: ['1990–91 Season', 'Laval Regents Team', 'QAHA League', 'Regular season', 'Regular season GP 30', 'Regular season G 25', 'A Regular season 20', 'Regular season Pts 45', '30 Playoffs PIM', 'Playoffs', '— Playoffs GP', '— Playoffs G', 'A Playoffs —', '— Pts', '— PIM']
    Answer:
    {{header_index: 1}}
    
    Example:
    Rows:
    0: ['Club', 'Season', 'League', 'League', 'League', 'FA Cup', 'FA Cup', 'Total', 'Total']
    1: ['Club', 'Season', 'League Division', 'League Apps', 'League Goals', 'FA Cup Apps', 'FA Cup Goals', 'Total Apps', 'Total Goals']
    
    Answer:
    {{header_index: 1}}
    
    Now your Rows:
    {rows_block}
    Answer:
    """
        # 3) Tokenize + generate
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(model.device)
        out = model.generate(
            **inputs,
            max_new_tokens=10,   # up to two digits + JSON syntax
            do_sample=False,
            num_beams=1,
            eos_token_id=tokenizer.eos_token_id,
        )
        text = tokenizer.decode(out[0], skip_special_tokens=True)[len(prompt):]
    
        # 4) Extract JSON from end
        #    Look for something like {"header_index": 2}
        m = re.search(r'\{ *header_index *: *(\d+) *\}', text)
        if not m:
            raise ValueError(f"Failed to parse header_index from model output:\n{text!r}")
        return int(m.group(1))
    
    
    best_idx = select_header_row(header_rows, model, tokenizer)
    return header_rows[best_idx][0], header_rows[best_idx][1]

def is_aggregate_field(field, model, tokenizer):
    """
    Ask the LLM if a given field name (e.g., 'Total Goals', 'Average', 'Career total')
    is an aggregate/statistical field.
    
    Returns: 1 if aggregate field, 0 otherwise.
    """

    prompt = f"""
You are a classifier for field names in tables.

If the field represents an **aggregate**, **summary**, **final result**, or **statistical measure** — output 1.
These include totals, averages, percentages, career totals, summary outcomes (like "Republican hold", "Turnout", "Swing", "Majority").

If the field is a **normal descriptor** such as a name, category, team, party, or location — output 0.
This includes standalone words like "Republican", "Democrat", "Club", "Season", "Division", "n-a", etc.

Be aware:
- Output 1 for outcome-oriented phrases like “Republican hold”, “Swing”, or “Turnout” that summarize results.
- Output 0 for standalone common words like “Republican”, “First Division”, or “California”.

Output exactly one digit: 1 or 0.

Examples:
Field: Total Goals → 1
Field: Average Points → 1
Field: Career total → 1
Field: Percentage → 1
Field: Republican hold → 1
Field: Turnout → 1
Field: Swing → 1
Field: Majority → 1
Field: Club → 0
Field: Republican → 0
Field: Season → 0
Field: First Division → 0
Field: California → 0

Now classify this:
Field: {field}
Answer:"""

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(model.device)
    out = model.generate(
        **inputs,
        max_new_tokens=5,
        do_sample=False,
        num_beams=1,
        eos_token_id=tokenizer.eos_token_id,
    )

    # Decode and extract final digit
    text = tokenizer.decode(out[0], skip_special_tokens=True)[len(prompt):].strip()
    if "1" in text:
        return 1
    elif "0" in text:
        return 0
    else:
        raise ValueError(f"Unexpected model output:\n{text}")

def infer_column_types(df, numeric_threshold=0.8, string_threshold=0.8):
    df.columns = df.columns.astype(str)  # Ensure column names are strings
    col_types = {}
    for col in df.columns:
        col_series = df[col]
        if not isinstance(col_series, pd.Series):
            continue  # skip if something weird
        values = col_series.astype(str).str.strip()
        total = len(values)
        if total == 0:
            col_types[col] = 'mixed'
            continue

        num_count = values.apply(lambda x: x.replace(',', '').replace('.', '', 1).isdigit()).sum()
        str_count = values.apply(lambda x: x.isalpha() or (x and not x.replace(',', '').replace('.', '', 1).isdigit())).sum()

        frac_num = num_count / total
        frac_str = str_count / total

        if frac_num >= numeric_threshold:
            col_types[col] = 'numeric'
        elif frac_str >= string_threshold:
            col_types[col] = 'string'
        else:
            col_types[col] = 'mixed'

    return col_types


def is_pure_string(value):
    """
    Returns True if value is a non-numeric, non-date, non-code string.
    """
    if not isinstance(value, str):
        return False
    value = value.strip()

    # Exclude if mostly numeric (years, numbers)
    if re.fullmatch(r'[\d,.\-–]+', value):
        return False

    # Exclude year-like
    if re.fullmatch(r'\d{4}(–\d{2,4})?', value):
        return False

    # Exclude empty/dashes
    if value == '' or value == '-':
        return False

    return True

def flatten_aggregated_table(merged_df, aggregate_fields):
    flat_rows = []

    for _, row in merged_df.iterrows():
        row_list = row.tolist()
        row_text = " ".join(str(x) for x in row_list)

        # Match aggregate field
        match_field = next((f for f in aggregate_fields if f in row_text), None)
        if not match_field:
            continue

        # Find field location
        match_indices = [i for i, val in enumerate(row_list) if match_field in str(val)]
        if not match_indices:
            continue  # no actual matching cell found
        first_idx, last_idx = match_indices[0], match_indices[-1]

        prefix = " ".join(str(x) for x in row_list[:first_idx]).strip()
        agg_name = f"{prefix} {match_field}".strip()

        # Build dict with stat fields
        stat_dict = {merged_df.columns[i]: row_list[i] for i in range(last_idx + 1, len(row_list))}
        stat_dict["aggregate_name"] = agg_name
        flat_rows.append(stat_dict)

    if not flat_rows:
        # Return empty DataFrame with just 'aggregate_name' column if nothing matched
        return pd.DataFrame(columns=["aggregate_name"])

    flat_df = pd.DataFrame(flat_rows)
    cols = ["aggregate_name"] + [col for col in flat_df.columns if col != "aggregate_name"]
    return flat_df[cols]

def find_matching_field(row, aggregate_fields):
    row_text = " ".join(str(x) for x in row)
    for field in aggregate_fields:
        if field in row_text:
            return field
    return None  # no match found

def find_matching_field_row(row_text, aggregate_fields):
    for field in aggregate_fields:
        if field in row_text:
            return field
    return None

def find_first_last_indices(lst, value):
    try:
        first = lst.index(value)
        last = len(lst) - 1 - lst[::-1].index(value)
        return first, last
    except ValueError:
        return None, None  # value not found