# Annotation Protocol

## Annotation Guidelines

- Keep the annotation short (1 sentence) yet still nuanced (do not converge to classification).
- Do not make any assumption regarding true user intent; only the inferred intent from the prompt.
- This includes not adding any information about the type of jailbreak used.
- Get the main goal: what information is the user trying to get from the system?
- If the prompt is nonsensical or ambiguous, it will be labelled as “ambiguous” (indicating that more information is needed).

## Harm Classification

We will align ourselves with the harm classification proposed in the WildGuard paper.

| Category | Subcategory | #Adv. Train | #Vani. Train | #Adv. Test | #Vani. Test |
| :--- | :--- | ---:| ---:| ---:| ---:|
| **Privacy** | Sensitive Information (Organization) | 1,264 | 1,801 | 26 | 24 |
| | Private Information (Individual) | 1,527 | 1,008 | 24 | 57 |
| | Copyright Violations | 264 | 1,507 | 22 | 9 |
| **Misinformation** | False or Misleading Information | 1,437 | 2,647 | 22 | 22 |
| | Material Harm by Misinformation | 600 | 882 | 24 | 21 |
| **Harmful Language** | Social Stereotypes & Discrimination | 3,165 | 3,178 | 23 | 53 |
| | Violence and Physical Harm | 780 | 2,121 | 22 | 18 |
| | Toxic Language / Hate Speech | 1,056 | 1,964 | 22 | 19 |
| | Sexual Content | 444 | 1,712 | 27 | 20 |
| **Malicious Uses** | Cyberattacks | 450 | 1,149 | 31 | 17 |
| | Fraud & Assisting Illegal Activities | 1,116 | 1,164 | 31 | 29 |
| | Encouraging Unethical/Unsafe Actions | 762 | 1,658 | 24 | 22 |
| | Mental Health & Over-Reliance Crisis | 582 | 1,231 | 26 | 21 |
| **Other harms** | | 7,100 | 553 | 17 | 81 |
| **Benign** | | 20,389 | 11,849 | 455 | 490 |