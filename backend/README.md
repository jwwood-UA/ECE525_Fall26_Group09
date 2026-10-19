# GPT-2 Logit Lens Backend

This backend runs pretrained GPT-2 small and shows next-token predictions from
each transformer block. It can append at most **one new tokenizer token** to a
prompt. A token can contain a space, part of a word, a whole word, or punctuation.

The input is a prompt and three optional settings. The output is a JSON object
with generated text, token records, and predictions for every position and block.
`logit_lens.py` provides a command-line interface and a Python API.

## 1. Set up the environment

Run these commands from this `backend` directory. This example creates a separate
Python 3.13 environment:

```bash
conda create -n ECE525 python=3.13 pip -y
conda activate ECE525
python -m pip install -r requirements.txt
```

The backend dependencies in this directory's `requirements.txt` are:

```text
torch>=2.6,<3
transformers==4.57.1
```

The parent directory's requirements are for the UI server. They do not replace
the backend requirements above. Install the backend requirements even if the UI
dependencies are already present.

### Model, device, and cache

The public model name `gpt2` loads `openai-community/gpt2` at the pinned revision
`607a30d783dfa663caf39e06633721c8d4cfcd7e`. The loader requests safetensors weights
and uses float32 with eager attention. It does not fall back to random weights.
On first use, it downloads the model configuration, weights, and tokenizer if
they are absent from the cache. This requires network access and local disk space.

| CLI setting | Behavior |
| --- | --- |
| `--device cpu` | Default. Run on the CPU. |
| `--device mps` | Run through PyTorch MPS on a supported Mac. Raise an error if MPS is unavailable. |
| `--device auto` | Select MPS when available; otherwise select the CPU. |
| `--local-files-only` | Load all model files from the cache. Fail if required files are missing. |

CUDA is not a supported device option in this backend.

By default, the loader uses the Hugging Face cache. For the CLI, set
`ECE525_MODEL_CACHE` to choose a different hub cache directory:

```bash
export ECE525_MODEL_CACHE="$HOME/.cache/ece525-hub"
python logit_lens.py --request request.json --output output.json
```

After the pinned model files are cached, run without downloads:

```bash
python logit_lens.py --request request.json --local-files-only --output output.json
```

For the Python API, pass the cache directory through `cache_dir` and the offline
option through `local_files_only`. The API constructor does not read
`ECE525_MODEL_CACHE` itself.

## 2. Run the backend

### Use a JSON request file

The included `request.json` contains:

```json
{
  "prompt": "The capital of France is",
  "max_new_tokens": 1,
  "model": "gpt2",
  "top_k": 5
}
```

Run it from this directory:

```bash
python logit_lens.py --request request.json --output output.json
```

`--output` writes UTF-8 JSON and replaces the file if it exists. Its parent
directory must exist. Without `--output`, the CLI prints the result JSON to
standard output.

### Pass a prompt directly

```bash
python logit_lens.py --prompt "The capital of France is" --output output.json
```

Choose exactly one input source: `--request` or `--prompt`.

### Override request settings

Explicit `--max-new-tokens`, `--model`, and `--top-k` flags override values from
the request file. Missing settings use the defaults in the input table below.
For example, this command disables generation and returns ten candidates per
position and block:

```bash
python logit_lens.py --request request.json --max-new-tokens 0 --top-k 10 --output analysis-only.json
```

The precedence is: **explicit CLI flag > JSON value > default**. A final
`max_new_tokens` value above 1 is rejected. Increasing `top_k` changes the number
of reported candidates; it does not increase the number of generated tokens.

### Call the Python API

Run Python from this directory so that `logit_lens` can be imported:

```python
from logit_lens import GPT2LogitLens

engine = GPT2LogitLens(
    model="gpt2",
    device="cpu",
    cache_dir=None,          # Or a local Hugging Face hub cache directory.
    local_files_only=False,
)
result = engine.analyze(
    prompt="The capital of France is",
    max_new_tokens=1,
    model="gpt2",
    top_k=5,
)
print(result["generated_text"])
```

Reuse `engine` across calls to avoid loading the model again. `analyze` returns a
JSON-safe Python dictionary; it does not write a file. Its defaults are also
`max_new_tokens=1`, `model="gpt2"`, and `top_k=5`.

## 3. Input fields and validation

A request file must contain a JSON object. Only the following fields are accepted;
unknown fields cause an error. `prompt` is required.

| Field | Type | Default | Meaning and limits |
| --- | --- | --- | --- |
| `prompt` | string | Required | Non-empty text, at most 100,000 Python characters. Spaces are preserved. It must encode to at least one token. |
| `max_new_tokens` | integer | `1` | Either `0` or `1`. `0` analyzes the prompt without appending a token. `1` allows one new tokenizer token. Larger values are rejected, not truncated. |
| `model` | string | `"gpt2"` | The only allowed model name. In the Python API, it must match the model loaded by the engine. |
| `top_k` | integer | `5` | Number of candidates per result row, from `1` to `50`. It must also fit within the model vocabulary. |

For JSON and Python calls, numeric settings must be integers. Booleans, floats,
and numeric strings are rejected. The CLI parses its numeric flags as integers.

The tokenizer encodes the prompt with `add_special_tokens=False`. The backend
requires:

```text
input_token_count + max_new_tokens <= context_window
```

The layer count and context window come from the loaded model configuration.
For this checkpoint, they are 12 blocks and 1,024 token positions. The character
limit does not replace the token limit. The backend rejects an input that is too
long; it does not truncate the prompt.

Invalid requests raise `RequestError` in Python. The CLI reports these errors
through its argument parser and exits with code 2. Model loading and runtime
failures can raise other exceptions. There is no JSON error-response schema.

## 4. How to read the output

### Start with the included example

The current `output.json` contains the prompt `"The capital of France is"` and
one generated token, `" the"` (token ID `262`). Its complete decoded text is
`"The capital of France is the"`.

The following table summarizes the file. It is not the full JSON output.

| Item | Value |
| --- | --- |
| Schema version | `"1.1"` |
| Input token count | `5` |
| Generated token count | `1` |
| Total token records | `6` |
| Blocks | `12`, indexed `0` through `11` |
| Result rows | `6 * 12 = 72` |
| Candidates per row | `5` |

Every result row at position `t` predicts the token at **`t + 1`**. It does not
predict the token at position `t`. For the example:

| Source position `t` | Token at `t` | Target position | Reference token | Result `generation_step` |
| --- | --- | --- | --- | --- |
| `0` | `"The"` | `1` | `" capital"` from the prompt | `null` |
| `4` | `" is"` | `5` | `" the"`, the first generated token | `0` |
| `5` | `" the"` | `null` | `null`: the next token was not appended | `null` |

Each source position has one row per block. The final position is the **terminal
position**. It still has next-token predictions, but the output sequence has no
next token to use as a reference. With generation enabled, the backend feeds the
generated token through the model to obtain these terminal rows. This forward
pass does not append another token. With `max_new_tokens=0`, the last prompt
position is the terminal position.

### Top-level fields

| Field | JSON type | Meaning |
| --- | --- | --- |
| `schema_version` | string | Output format version. Currently `"1.1"`. |
| `request_id` | string | A new UUID string for this successful analysis. |
| `status` | string | `"success"` for returned results. Errors are raised rather than returned in this object. |
| `model` | string | Loaded public model name, currently `"gpt2"`. |
| `method` | string | Analysis method, currently `"logit_lens"`. |
| `is_mock` | boolean | `false`. This backend uses pretrained model weights. |
| `input_text` | string | The original prompt, with its spaces preserved. |
| `max_new_tokens` | integer | The accepted generation limit, `0` or `1`. |
| `top_k` | integer | Number of candidates in each row's `predictions` array. |
| `n_layers` | integer | Number of transformer blocks from the model configuration; 12 for this checkpoint. |
| `context_window` | integer | Maximum token sequence length from the model configuration; 1,024 for this checkpoint. |
| `input_token_count` | integer | Number of tokenizer token IDs in the prompt. |
| `generated_token_count` | integer | Number of token IDs actually appended; no greater than `max_new_tokens`. An EOS token is included in this count. |
| `generated_text` | string | Decoded generated IDs only. Empty when no tokens are generated. Leading spaces are retained. |
| `output_text` | string | The decoded full sequence: input IDs followed by generated IDs. |
| `stop_reason` | string | `"max_new_tokens"` when the requested budget is reached, including a budget of 0; `"eos_token"` when the appended token is the tokenizer's end-of-sequence (EOS) token. |
| `tokens` | array of objects | Input and generated token records in sequence order. See below. |
| `results` | array of objects | One prediction row per sequence position and transformer block. Sorted by `token_position`, then `block_index`. |

`len(tokens) = input_token_count + generated_token_count`.
`len(results) = len(tokens) * n_layers`.

### `tokens[]`: the sequence itself

| Field | JSON type | Meaning |
| --- | --- | --- |
| `position` | integer | **0-based** index in the full input-plus-generated sequence. |
| `token_id` | integer | Tokenizer vocabulary ID. It is an identifier, not a sequence position or rank. |
| `text` | string | This token ID decoded on its own. Spaces and special-token text are retained. |
| `source` | string | `"input"` for prompt tokens; `"generated"` for appended tokens. |
| `generation_step` | integer or null | **0-based** generation index of this token itself: `position - input_token_count` for generated tokens. `null` for prompt tokens. |

Token strings are decoded separately. Use `output_text` and `generated_text` for
full-sequence decoding, especially when a character spans multiple token IDs.
Decoding uses `skip_special_tokens=False` and
`clean_up_tokenization_spaces=False`.

### `results[]`: predictions from each block

| Field | JSON type | Meaning |
| --- | --- | --- |
| `block_index` | integer | **0-based** transformer block index, from `0` to `n_layers - 1`. It refers to that block's output residual stream. |
| `token_position` | integer | **0-based** source position `t`. The row uses the residual stream at this position to predict the next token. |
| `target_token_position` | integer or null | `t + 1` when that token is present in the returned sequence. `null` at the terminal position. |
| `generation_step` | integer or null | **0-based** generation index of the row's known target: `target_token_position - input_token_count` when the target is generated. `null` when the target is a prompt token or no target is known. |
| `predictions` | array of objects | The `top_k` candidate next tokens, sorted from highest to lowest probability. Present even at the terminal position. |
| `reference_token` | object or null | The actual next token in the returned sequence, with its rank and probability under this block's prediction. `null` only when no next token is available at the terminal position. |

The two `generation_step` fields refer to different tokens. In the example,
`tokens[5].generation_step` is `0` because token 5 was generated first. A result
row at position 4 also has `generation_step=0` because it predicts token 5. A
result row at position 5 has `generation_step=null` because its next token was
not generated.

A prompt reference is the next supplied prompt token. A generated reference is
the token selected by greedy decoding. Neither is an external correctness label.

### `results[].predictions[]`: candidate fields

| Field | JSON type | Meaning |
| --- | --- | --- |
| `token_id` | integer | Candidate tokenizer vocabulary ID. |
| `token` | string | Candidate ID decoded on its own, including any leading space or special-token text. |
| `probability` | number | Probability after softmax over the **full vocabulary**. It lies between 0 and 1. |
| `rank` | integer | **1-based** position in the sorted candidate list: `1` through `top_k`. Equal scores still receive consecutive list ranks. |

The backend computes full-vocabulary softmax before selecting the top candidates.
It does **not** normalize the retained probabilities again. Their sum can be
less than 1 because other vocabulary tokens also have probability mass.

### `results[].reference_token`: known next-token fields

When the next token exists, this object is present even if that token is absent
from `predictions`. For example, the first row in `output.json` has reference
`" capital"` with rank `3157`, outside its five reported candidates.

| Field | JSON type | Meaning |
| --- | --- | --- |
| `token_id` | integer | Vocabulary ID of the known next token. |
| `token` | string | That ID decoded on its own. |
| `rank` | integer | **1-based**, full-vocabulary competition rank: `1 + number of vocabulary logits strictly greater than this token's logit`. Equal logits share the same rank. |
| `probability` | number | This token's full-vocabulary softmax probability under the current block and source position. |

Candidate list ranks and reference competition ranks use different tie rules.
Without ties, a reference in the top candidates has the same rank in both places.
`null` means there is no known next token; it does not mean rank 0 or probability 0.

## 5. What the Logit Lens measures

The backend captures each block's raw output residual stream, before the model's
final layer normalization. For each position, it applies the pretrained model's
final layer normalization (`ln_f`) and vocabulary projection (`lm_head`), then a
full-vocabulary softmax. It does not apply `ln_f` twice to the last block.

Greedy generation selects the largest native final-model logit. Intermediate
block predictions and `top_k` do not control generation. The Logit Lens exposes
how these intermediate states decode through the final projection; it does not
by itself prove that the model uses an intermediate prediction as a decision.
