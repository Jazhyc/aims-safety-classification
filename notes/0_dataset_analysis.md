# WildGuardMix (Train Subset)

## Important columns

- `prompt` (text): the input prompt we classify.
- `prompt_harm_label`: target label for whether the prompt is harmful (our primary target).
- `adversarial` (bool): whether the prompt was adversarially generated or modified.
- `subcategory`: fine-grained label for harmful prompts.

## Adversarial vs prompt label

Adversarial does not imply harmful. Keeping `adversarial` in the stratification key captures adversarial-but-unharmful as well as adversarial-and-harmful examples.

![Adversarial vs prompt label](../notes/imgs/adversarial_vs_prompt_label.png)

## Subcategory distribution

- 15 subcategories
- Benign is the majority class
- 13 dedicated harmful subcategories
- 1 catch-all harmful subcategory

![Subcategory distribution](../notes/imgs/subcategory_distribution.png)

The subcategory distribution motivates stratification — several subcategories are rare and would be underrepresented without stratified sampling.

## Stratification

We stratified on a combined key made from `prompt_harm_label`, `adversarial`, and `subcategory` to preserve joint proportions across splits. This ensures rare subcategories and intersections (for example, adversarial-but-unharmful) are represented in train, validation and test.

## Data Filtering

We filtered to keep only English prompts by examining the first 80 characters of each prompt using [fast-langdetect](https://github.com/LlmKira/fast-langdetect). This ensures our model focuses on English-language jailbreak attempts.

## Splits

The full dataset is split into two stages:

**Stage 1: Train vs Annotation set (10% / 90%)**
- **Annotation set:** 90% of the full dataset — reserved for human annotation to build labeled examples for analysis
- **Training set:** 10% of the full dataset — used for model training and evaluation

**Stage 2: Within the training set (80% train / 10% val / 10% test)**
- **Training:** 80% of the training set — used to train the model
- **Validation:** 10% of the training set — used for evaluation during training and limited hyperparameter tuning
- **Test:** 10% of the training set — held-out test set for evaluating final model performance

This approach preserves most of the data for annotation while using a smaller clean split for training/validation/testing.

## Prompt length & adversarial prompts

Prompt lengths (in words) differ by label and adversarial status: harmful prompts are generally longer, and adversarial prompts are substantially longer on average. This informed our max-length decision.

![Adversarial prompt length](../notes/imgs/adversarial_length.png)

- Observation: harmful prompts trend longer; adversarial prompts are much longer.
- With a context window of 2048 tokens, we can cover all prompts without truncation.

## Response-related fields

The dataset includes response-level columns (e.g., `response_refusal_label`, `response_harm_label`) but many samples lack response annotations. We preserve these columns, but they are not used for prompt-intent annotation.

## Reproducibility

- Random seed: 42
- Tools: scikit-learn 1.7.2, pandas, datasets
