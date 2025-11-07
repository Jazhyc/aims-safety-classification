# Data filtration and selection for annotation

We used the trained ModernBERT classifier to analyze the held-out test set and pick hard examples for manual intent annotation.

## Held-out test analysis
Classifier performance was lower on adversarial prompts (see plot) which seems logical since they are designed to be confusing.  
The plot below shows `test_adversarial_f1` (adversarial vs non-adversarial F1):

![Adversarial F1 (test)](../notes/imgs/test_adversarial_f1.png)

## Selecting borderline examples
We selected samples where the classifier's probability for the "harmful" class (i.e., harmful-probability) was near 0.5. Concretely, we exported all examples with harmful-probability in the range [0.35, 0.65] for manual annotation. The number of samples in each confidence bin is shown below:

![Filtered distribution around 0.5](../notes/imgs/filtered-distribution.png)

![Adversarial in filtered set](../notes/imgs/adversarial_status.png)

![Adversarial in filtered set](../notes/imgs/prompt_harm_status.png)

As seen in the above figure, adversarial prompts were overrepresented in these filtered sets, confirming they are harder for the classifier. The distribution of harmful vs non-harmful labels in these borderline examples is also roughly balanced, which is ideal for annotation.

## Annotation
The exported borderline examples were imported into Label Studio for manual intent annotation.

