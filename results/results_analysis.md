# RAG Evaluation Analysis

Generated: 2025-12-03T04:53:02.126684 UTC


## Aggregate comparison

| Chunk Size | HIT RATE | MRR | PRECISION@1 | PRECISION@5 | ROUGE L | BLEU | COSINE SIM | RELEVANCE | FAITHFULNESS |
|---|---|---|---|---|---|---|---|---|---|
| 200_300 | 0.880 | 0.757 | 0.680 | 0.464 | 0.208 | 0.040 | 0.509 | 0.593 | 0.473 |
| 500_600 | 0.840 | 0.738 | 0.680 | 0.592 | 0.223 | 0.044 | 0.513 | 0.593 | 0.457 |
| 800_1000 | 0.800 | 0.728 | 0.680 | 0.648 | 0.223 | 0.045 | 0.517 | 0.596 | 0.456 |

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

- ID: 25 — Q: How does Ambedkar characterize the Hindu-Muslim problem in Document 5?...

  - ROUGE-L: 0.082, BLEU: 0.006, CosSim: 0.481

  - Retrieved: ['speech5.txt', 'speech5.txt', 'speech5.txt', 'speech1.txt', 'speech6.txt']

  - Answer (truncated): `Ambedkar characterizes the Hindu-Muslim problem in *Document 5* as a clash between two distinct nationalities rather than mere religious communities, emphasizing their divergent religions, cultures, languages, and historical backgrounds. He argues th...`

- ID: 9 — Q: How does Ambedkar relate political power to social change in Documents 3 and 6?...

  - ROUGE-L: 0.095, BLEU: 0.005, CosSim: 0.447

  - Retrieved: ['speech3.txt', 'speech1.txt', 'speech1.txt', 'speech6.txt', 'speech2.txt']

  - Answer (truncated): `In Documents 3 and 6, Ambedkar links political power to social change by arguing that political democracy is unsustainable without foundational social democracy, particularly the eradication of caste. He asserts that caste, rooted in the authority of...`

- ID: 12 — Q: What is the Dhamma according to Document 2?...

  - ROUGE-L: 0.110, BLEU: 0.007, CosSim: 0.350

  - Retrieved: ['speech2.txt', 'speech2.txt', 'speech6.txt', 'speech3.txt', 'speech1.txt']

  - Answer (truncated): `According to Document 2, the Dhamma, as taught by the Buddha, is a social and moral code centered on human relationships, aiming to create a society of free and equal individuals. It emphasizes ethical conduct, rejecting ritualism, sacrifices, the au...`

- ID: 8 — Q: What is Ambedkar's concept of ideal society across Documents 1 and 3?...

  - ROUGE-L: 0.131, BLEU: 0.047, CosSim: 0.421

  - Retrieved: ['speech1.txt', 'speech1.txt', 'speech2.txt', 'speech6.txt', 'speech3.txt']

  - Answer (truncated): `Ambedkar's concept of an ideal society, as reflected in Documents 1 and 3, is one founded on the principles of *liberty, equality, and fraternity*, free from the oppressive structures of caste and tradition. He envisions a society where individuals a...`

- ID: 4 — Q: What personal experiences of untouchability does Ambedkar describe in Document 4?...

  - ROUGE-L: 0.141, BLEU: 0.017, CosSim: 0.477

  - Retrieved: ['speech6.txt', 'speech6.txt', 'speech4.txt', 'speech4.txt', 'speech3.txt']

  - Answer (truncated): `In *Waiting for a Visa*, Ambedkar describes personal experiences of untouchability, including the humiliation of being barred from drinking water from public wells due to his caste. He recounts the stigma, insults, and systemic injustices he faced fr...`


### Chunk: 500_600 — Worst 5 by ROUGE-L

- ID: 9 — Q: How does Ambedkar relate political power to social change in Documents 3 and 6?...

  - ROUGE-L: 0.109, BLEU: 0.007, CosSim: 0.472

  - Retrieved: ['speech3.txt', 'speech1.txt', 'speech1.txt', 'speech3.txt', 'speech1.txt']

  - Answer (truncated): `In Documents 3 and 6, Ambedkar links political power to social change by arguing that political democracy alone is insufficient without social democracy. He asserts that true social transformation requires dismantling entrenched systems like caste, w...`

- ID: 4 — Q: What personal experiences of untouchability does Ambedkar describe in Document 4?...

  - ROUGE-L: 0.116, BLEU: 0.009, CosSim: 0.454

  - Retrieved: ['speech6.txt', 'speech6.txt', 'speech6.txt', 'speech4.txt', 'speech4.txt']

  - Answer (truncated): `In *Document 4*, Ambedkar describes personal experiences of untouchability such as being forced to live outside village boundaries, denied access to education, barred from owning property, and relegated to degrading menial labor. He also highlights s...`

- ID: 25 — Q: How does Ambedkar characterize the Hindu-Muslim problem in Document 5?...

  - ROUGE-L: 0.154, BLEU: 0.008, CosSim: 0.497

  - Retrieved: ['speech5.txt', 'speech5.txt', 'speech5.txt', 'speech5.txt', 'speech1.txt']

  - Answer (truncated): `Ambedkar characterizes the Hindu-Muslim problem in *Document 5* as a conflict between two distinct nationalities rather than mere religious communities. He argues that Hindus and Muslims possess fundamentally different religions, cultures, languages,...`

- ID: 8 — Q: What is Ambedkar's concept of ideal society across Documents 1 and 3?...

  - ROUGE-L: 0.157, BLEU: 0.050, CosSim: 0.490

  - Retrieved: ['speech1.txt', 'speech1.txt', 'speech1.txt', 'speech2.txt', 'speech6.txt']

  - Answer (truncated): `Ambedkar's concept of an ideal society, as outlined in Documents 1 and 3, is founded on the principles of **liberty, equality, and fraternity**, rejecting the constraints of tradition and caste-based oppression. He envisions a society where individua...`

- ID: 20 — Q: What is the relationship between education and liberation in Ambedkar's philosophy?...

  - ROUGE-L: 0.186, BLEU: 0.007, CosSim: 0.661

  - Retrieved: ['speech4.txt', 'speech6.txt', 'speech4.txt', 'speech4.txt', 'speech4.txt']

  - Answer (truncated): `In Ambedkar's philosophy, education is a fundamental tool for liberation, particularly for the oppressed and marginalized, such as the untouchables. He believed that education empowers individuals to challenge systemic injustice, assert their rights,...`


### Chunk: 800_1000 — Worst 5 by ROUGE-L

- ID: 4 — Q: What personal experiences of untouchability does Ambedkar describe in Document 4?...

  - ROUGE-L: 0.100, BLEU: 0.009, CosSim: 0.490

  - Retrieved: ['speech6.txt', 'speech6.txt', 'speech6.txt', 'speech6.txt', 'speech4.txt']

  - Answer (truncated): `In *Document 4*, Dr. B.R. Ambedkar describes personal experiences of untouchability such as being forced to live outside village boundaries, denied access to education, barred from owning property, and relegated to degrading, menial labor solely due ...`

- ID: 20 — Q: What is the relationship between education and liberation in Ambedkar's philosophy?...

  - ROUGE-L: 0.129, BLEU: 0.003, CosSim: 0.679

  - Retrieved: ['speech4.txt', 'speech4.txt', 'speech6.txt', 'speech4.txt', 'speech4.txt']

  - Answer (truncated): `In Ambedkar's philosophy, education is intrinsically linked to liberation, particularly for oppressed communities like the untouchables. He viewed education as the primary tool to dismantle systemic caste oppression, empowering individuals to challen...`

- ID: 25 — Q: How does Ambedkar characterize the Hindu-Muslim problem in Document 5?...

  - ROUGE-L: 0.129, BLEU: 0.007, CosSim: 0.492

  - Retrieved: ['speech5.txt', 'speech5.txt', 'speech5.txt', 'speech5.txt', 'speech5.txt']

  - Answer (truncated): `Ambedkar characterizes the Hindu-Muslim problem in *Document 5* as a clash of nationalities rather than mere communal differences, arguing that Hindus and Muslims constitute two distinct nations due to their divergent religions, cultures, languages, ...`

- ID: 8 — Q: What is Ambedkar's concept of ideal society across Documents 1 and 3?...

  - ROUGE-L: 0.154, BLEU: 0.055, CosSim: 0.431

  - Retrieved: ['speech1.txt', 'speech1.txt', 'speech1.txt', 'speech2.txt', 'speech6.txt']

  - Answer (truncated): `Ambedkar's concept of an ideal society, as outlined in Documents 1 and 3, is founded on the principles of **liberty, equality, and fraternity**, rejecting the constraints of tradition and caste-based oppression. He envisions a society where individua...`

- ID: 22 — Q: What specific restrictions did untouchables face according to Document 4?...

  - ROUGE-L: 0.160, BLEU: 0.015, CosSim: 0.551

  - Retrieved: ['speech6.txt', 'speech6.txt', 'speech6.txt', 'speech6.txt', 'speech6.txt']

  - Answer (truncated): `According to Document 4, untouchables faced severe restrictions including being forced to live outside villages, denied the right to education and property ownership, and condemned to perform only menial labor.

SOURCES: The Untouchables`


## Recommendation
Based on a combined heuristic (relevance + faithfulness + hit_rate) the recommended chunk configuration is **200_300**.
