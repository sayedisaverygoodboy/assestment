# RAG Evaluation Analysis

Generated: 2025-12-03T04:07:15.933498 UTC


## Aggregate comparison

| Chunk Size | HIT RATE | MRR | PRECISION@1 | PRECISION@5 | ROUGE L | BLEU | COSINE SIM | RELEVANCE | FAITHFULNESS |
|---|---|---|---|---|---|---|---|---|---|
| 200_300 | 0.880 | 0.757 | 0.680 | 0.464 | 0.241 | 0.057 | 0.521 | 0.596 | 0.579 |
| 500_600 | 0.840 | 0.738 | 0.680 | 0.592 | 0.251 | 0.072 | 0.505 | 0.587 | 0.581 |
| 800_1000 | 0.800 | 0.728 | 0.680 | 0.648 | 0.264 | 0.074 | 0.523 | 0.625 | 0.594 |

![hit_rate](hit_rate.png)


![mrr](mrr.png)


![precision@1](precision@1.png)


![precision@5](precision@5.png)


![rouge_l](rouge_l.png)


![bleu](bleu.png)


![cosine_sim](cosine_sim.png)


![relevance](relevance.png)


![faithfulness](faithfulness.png)


## Failure Mode Analysis


### Chunk: 200_300 — Worst 5 by ROUGE-L

- ID: 20 — Q: What is the relationship between education and liberation in Ambedkar's philosophy?...

  - ROUGE-L: 0.073, BLEU: 0.002, CosSim: 0.621

  - Retrieved: ['speech4.txt', 'speech6.txt', 'speech4.txt', 'speech6.txt', 'speech2.txt']

  - Answer (truncated): `In Ambedkar's philosophy, education is seen as a crucial factor in achieving liberation for the untouchables. It is considered the key to empowering them to fight against injustice and attain their rightful place in society. Through education, they c...`

- ID: 9 — Q: How does Ambedkar relate political power to social change in Documents 3 and 6?...

  - ROUGE-L: 0.114, BLEU: 0.004, CosSim: 0.333

  - Retrieved: ['speech3.txt', 'speech1.txt', 'speech1.txt', 'speech6.txt', 'speech2.txt']

  - Answer (truncated): `Ambedkar relates political power to social change by emphasizing that political democracy cannot last without social democracy, implying that true power lies in challenging the underlying social structures, such as the caste system and the authority ...`

- ID: 12 — Q: What is the Dhamma according to Document 2?...

  - ROUGE-L: 0.140, BLEU: 0.005, CosSim: 0.355

  - Retrieved: ['speech2.txt', 'speech2.txt', 'speech6.txt', 'speech3.txt', 'speech1.txt']

  - Answer (truncated): `The Dhamma, according to Document 2 ("The Buddha and His Dhamma"), is a social code of moral conduct for society, centered on man and the relationship between man and man, aiming to create a society of free and equal men.

SOURCES: The Buddha and His...`

- ID: 8 — Q: What is Ambedkar's concept of ideal society across Documents 1 and 3?...

  - ROUGE-L: 0.162, BLEU: 0.059, CosSim: 0.457

  - Retrieved: ['speech1.txt', 'speech1.txt', 'speech2.txt', 'speech6.txt', 'speech3.txt']

  - Answer (truncated): `Ambedkar's concept of an ideal society is based on the principles of liberty, equality, and fraternity, where individuals are free from the shackles of tradition and oppression, and are treated as equals with a sense of brotherhood. This ideal can be...`

- ID: 19 — Q: How does Ambedkar's view on religion differ between Documents 2 and 5?...

  - ROUGE-L: 0.175, BLEU: 0.044, CosSim: 0.811

  - Retrieved: ['speech2.txt', 'speech5.txt', 'speech2.txt', 'speech6.txt', 'speech5.txt']

  - Answer (truncated): `Ambedkar's view on religion in Document 2 ("Pakistan or the Partition of India") focuses on the differences between Hindus and Muslims, highlighting their distinct religions, cultures, and historical backgrounds. In contrast, Document 5 ("The Buddha ...`


### Chunk: 500_600 — Worst 5 by ROUGE-L

- ID: 20 — Q: What is the relationship between education and liberation in Ambedkar's philosophy?...

  - ROUGE-L: 0.067, BLEU: 0.002, CosSim: 0.605

  - Retrieved: ['speech4.txt', 'speech6.txt', 'speech4.txt', 'speech4.txt', 'speech4.txt']

  - Answer (truncated): `In Ambedkar's philosophy, education is seen as a crucial factor in achieving liberation for the untouchables. He believed that without education, they can never attain their rightful place in society, and that it gives them the strength to fight agai...`

- ID: 4 — Q: What personal experiences of untouchability does Ambedkar describe in Document 4?...

  - ROUGE-L: 0.114, BLEU: 0.005, CosSim: 0.329

  - Retrieved: ['speech6.txt', 'speech6.txt', 'speech6.txt', 'speech4.txt', 'speech4.txt']

  - Answer (truncated): `The passage provided does not describe Ambedkar's personal experiences of untouchability in Document 4, as the given passages are from "The Untouchables" and do not reference Document 4. The text discusses the history and plight of the untouchables, ...`

- ID: 9 — Q: How does Ambedkar relate political power to social change in Documents 3 and 6?...

  - ROUGE-L: 0.128, BLEU: 0.004, CosSim: 0.376

  - Retrieved: ['speech3.txt', 'speech1.txt', 'speech1.txt', 'speech3.txt', 'speech1.txt']

  - Answer (truncated): `Ambedkar relates political power to social change by emphasizing that political democracy cannot last without social democracy, which is based on liberty, equality, and fraternity. He argues that the existing social order, rooted in the shastras and ...`

- ID: 7 — Q: Compare Ambedkar's views on education in Documents 4 and 6...

  - ROUGE-L: 0.141, BLEU: 0.004, CosSim: 0.569

  - Retrieved: ['speech6.txt', 'speech4.txt', 'speech4.txt', 'speech4.txt', 'speech4.txt']

  - Answer (truncated): `Ambedkar's views on education in Documents 4 and 6 are not directly comparable as the passages provided are identical and do not have distinct document numbers. However, based on the given passage, Ambedkar valued education as a means to fight agains...`

- ID: 15 — Q: How did Ambedkar struggle for education according to Document 4?...

  - ROUGE-L: 0.169, BLEU: 0.013, CosSim: 0.288

  - Retrieved: ['speech6.txt', 'speech4.txt', 'speech4.txt', 'speech4.txt', 'speech2.txt']

  - Answer (truncated): `Ambedkar struggled for education by sitting outside the classroom to attend school and studying under street lights when he couldn't afford kerosene, showing his determination to get educated despite the odds.

SOURCES: Document 4 is not directly ava...`


### Chunk: 800_1000 — Worst 5 by ROUGE-L

- ID: 7 — Q: Compare Ambedkar's views on education in Documents 4 and 6...

  - ROUGE-L: 0.119, BLEU: 0.021, CosSim: 0.502

  - Retrieved: ['speech6.txt', 'speech4.txt', 'speech4.txt', 'speech4.txt', 'speech4.txt']

  - Answer (truncated): `Ambedkar's views on education are reflected in the personal anecdote of studying under street lights due to lack of money for kerosene, highlighting the importance he placed on education as a means to fight against injustice, as seen in Documents 4 a...`

- ID: 22 — Q: What specific restrictions did untouchables face according to Document 4?...

  - ROUGE-L: 0.128, BLEU: 0.009, CosSim: 0.550

  - Retrieved: ['speech6.txt', 'speech6.txt', 'speech6.txt', 'speech6.txt', 'speech6.txt']

  - Answer (truncated): `Untouchables faced specific restrictions, including being forced to live outside the village, being denied the right to education and property, and being condemned to do menial work.

SOURCES: The Untouchables`

- ID: 9 — Q: How does Ambedkar relate political power to social change in Documents 3 and 6?...

  - ROUGE-L: 0.138, BLEU: 0.007, CosSim: 0.605

  - Retrieved: ['speech3.txt', 'speech1.txt', 'speech3.txt', 'speech1.txt', 'speech3.txt']

  - Answer (truncated): `Ambedkar relates political power to social change by emphasizing that political democracy cannot last without social democracy, which is based on the principles of liberty, equality, and fraternity. He argues that the Constitution must provide for ec...`

- ID: 4 — Q: What personal experiences of untouchability does Ambedkar describe in Document 4?...

  - ROUGE-L: 0.141, BLEU: 0.007, CosSim: 0.432

  - Retrieved: ['speech6.txt', 'speech6.txt', 'speech6.txt', 'speech6.txt', 'speech4.txt']

  - Answer (truncated): `The passage "The Untouchables" does not describe Ambedkar's personal experiences of untouchability in Document 4, as the provided passages are likely excerpts from "The Untouchables" by Ambedkar, but the actual content of Document 4 is not available....`

- ID: 8 — Q: What is Ambedkar's concept of ideal society across Documents 1 and 3?...

  - ROUGE-L: 0.179, BLEU: 0.066, CosSim: 0.467

  - Retrieved: ['speech1.txt', 'speech1.txt', 'speech1.txt', 'speech2.txt', 'speech6.txt']

  - Answer (truncated): `Ambedkar's concept of an ideal society, as described across Documents 1 and 3, is based on the principles of liberty, equality, and fraternity. He emphasizes the need to abolish caste to achieve this ideal, highlighting the importance of being free f...`


## Recommendation
Based on a combined heuristic (relevance + faithfulness + hit_rate) the recommended chunk configuration is **200_300**.
