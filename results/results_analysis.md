# RAG Evaluation Analysis

Generated: 2025-12-03T04:23:29.501399 UTC


## Aggregate comparison

| Chunk Size | HIT RATE | MRR | PRECISION@1 | PRECISION@5 | ROUGE L | BLEU | COSINE SIM | RELEVANCE | FAITHFULNESS |
|---|---|---|---|---|---|---|---|---|---|
| 200_300 | 0.880 | 0.757 | 0.680 | 0.464 | 0.220 | 0.038 | 0.514 | 0.589 | 0.468 |
| 500_600 | 0.880 | 0.748 | 0.680 | 0.576 | 0.220 | 0.043 | 0.510 | 0.595 | 0.461 |
| 800_1000 | 0.880 | 0.744 | 0.680 | 0.648 | 0.223 | 0.041 | 0.516 | 0.608 | 0.453 |

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

- ID: 9 — Q: How does Ambedkar relate political power to social change in Documents 3 and 6?...

  - ROUGE-L: 0.129, BLEU: 0.007, CosSim: 0.434

  - Retrieved: ['speech3.txt', 'speech1.txt', 'speech1.txt', 'speech6.txt', 'speech2.txt']

  - Answer (truncated): `In Documents 3 and 6, Ambedkar argues that political power alone is insufficient for meaningful social change without addressing underlying social structures. He asserts that political democracy cannot endure without social democracy, emphasizing tha...`

- ID: 8 — Q: What is Ambedkar's concept of ideal society across Documents 1 and 3?...

  - ROUGE-L: 0.131, BLEU: 0.047, CosSim: 0.421

  - Retrieved: ['speech1.txt', 'speech1.txt', 'speech2.txt', 'speech6.txt', 'speech3.txt']

  - Answer (truncated): `Ambedkar's concept of an ideal society, as reflected in Documents 1 and 3, is one founded on the principles of *liberty, equality, and fraternity*, free from the oppressive structures of caste and tradition. He envisions a society where individuals a...`

- ID: 4 — Q: What personal experiences of untouchability does Ambedkar describe in Document 4?...

  - ROUGE-L: 0.141, BLEU: 0.017, CosSim: 0.477

  - Retrieved: ['speech6.txt', 'speech6.txt', 'speech4.txt', 'speech4.txt', 'speech3.txt']

  - Answer (truncated): `In *Waiting for a Visa*, Ambedkar describes personal experiences of untouchability, including the humiliation of being barred from drinking water from public wells due to his caste. He recounts the stigma, insults, and systemic injustices he faced fr...`

- ID: 25 — Q: How does Ambedkar characterize the Hindu-Muslim problem in Document 5?...

  - ROUGE-L: 0.141, BLEU: 0.006, CosSim: 0.556

  - Retrieved: ['speech5.txt', 'speech5.txt', 'speech5.txt', 'speech1.txt', 'speech6.txt']

  - Answer (truncated): `Ambedkar characterizes the Hindu-Muslim problem in *Document 5* as a clash of nationalities rather than mere communal differences, arguing that Hindus and Muslims constitute two distinct nations due to their divergent religions, cultures, languages, ...`

- ID: 7 — Q: Compare Ambedkar's views on education in Documents 4 and 6...

  - ROUGE-L: 0.154, BLEU: 0.042, CosSim: 0.644

  - Retrieved: ['speech6.txt', 'speech4.txt', 'speech4.txt', 'speech2.txt', 'speech6.txt']

  - Answer (truncated): `Dr. B.R. Ambedkar’s views on education in Documents 4 and 6 emphasize its transformative power as a tool for liberation and social justice. In Document 4, he underscores education as essential for untouchables to fight for their rights, organize, and...`


### Chunk: 500_600 — Worst 5 by ROUGE-L

- ID: 9 — Q: How does Ambedkar relate political power to social change in Documents 3 and 6?...

  - ROUGE-L: 0.111, BLEU: 0.007, CosSim: 0.390

  - Retrieved: ['speech3.txt', 'speech1.txt', 'speech1.txt', 'speech3.txt', 'speech1.txt']

  - Answer (truncated): `In Documents 3 and 6, Ambedkar links political power to social change by arguing that political democracy is unsustainable without foundational social democracy, which requires dismantling oppressive structures like caste and the authority of the *sh...`

- ID: 8 — Q: What is Ambedkar's concept of ideal society across Documents 1 and 3?...

  - ROUGE-L: 0.147, BLEU: 0.053, CosSim: 0.427

  - Retrieved: ['speech1.txt', 'speech1.txt', 'speech1.txt', 'speech2.txt', 'speech6.txt']

  - Answer (truncated): `Ambedkar's concept of an ideal society, as outlined in Documents 1 and 3, is founded on the principles of **liberty, equality, and fraternity**, rejecting the constraints of tradition and caste-based oppression. He envisions a society where individua...`

- ID: 4 — Q: What personal experiences of untouchability does Ambedkar describe in Document 4?...

  - ROUGE-L: 0.148, BLEU: 0.041, CosSim: 0.449

  - Retrieved: ['speech6.txt', 'speech6.txt', 'speech6.txt', 'speech4.txt', 'speech4.txt']

  - Answer (truncated): `In *Document 4*, Dr. B.R. Ambedkar describes personal experiences of untouchability, including being forced to sit outside classrooms, denied access to water from public wells, and subjected to humiliation and segregation in schools and society due t...`

- ID: 25 — Q: How does Ambedkar characterize the Hindu-Muslim problem in Document 5?...

  - ROUGE-L: 0.154, BLEU: 0.008, CosSim: 0.497

  - Retrieved: ['speech5.txt', 'speech5.txt', 'speech5.txt', 'speech5.txt', 'speech1.txt']

  - Answer (truncated): `Ambedkar characterizes the Hindu-Muslim problem in *Document 5* as a conflict between two distinct nationalities rather than mere religious communities. He argues that Hindus and Muslims possess fundamentally different religions, cultures, languages,...`

- ID: 20 — Q: What is the relationship between education and liberation in Ambedkar's philosophy?...

  - ROUGE-L: 0.154, BLEU: 0.007, CosSim: 0.702

  - Retrieved: ['speech4.txt', 'speech6.txt', 'speech4.txt', 'speech4.txt', 'speech4.txt']

  - Answer (truncated): `In Ambedkar's philosophy, education is intrinsically linked to liberation as it serves as the primary tool for emancipation from caste oppression and social injustice. He believed that without education, marginalized communities like the untouchables...`


### Chunk: 800_1000 — Worst 5 by ROUGE-L

- ID: 4 — Q: What personal experiences of untouchability does Ambedkar describe in Document 4?...

  - ROUGE-L: 0.100, BLEU: 0.009, CosSim: 0.492

  - Retrieved: ['speech6.txt', 'speech6.txt', 'speech6.txt', 'speech6.txt', 'speech4.txt']

  - Answer (truncated): `In *Document 4*, Dr. B.R. Ambedkar describes personal experiences of untouchability such as being forced to live outside village boundaries, denied access to education, barred from owning property, and relegated to degrading menial labor solely due t...`

- ID: 20 — Q: What is the relationship between education and liberation in Ambedkar's philosophy?...

  - ROUGE-L: 0.129, BLEU: 0.004, CosSim: 0.683

  - Retrieved: ['speech4.txt', 'speech4.txt', 'speech6.txt', 'speech4.txt', 'speech4.txt']

  - Answer (truncated): `In Ambedkar's philosophy, education is intrinsically linked to liberation, particularly for oppressed communities like the untouchables. He viewed education as the essential tool to dismantle systemic caste oppression, empowering individuals to chall...`

- ID: 25 — Q: How does Ambedkar characterize the Hindu-Muslim problem in Document 5?...

  - ROUGE-L: 0.129, BLEU: 0.007, CosSim: 0.492

  - Retrieved: ['speech5.txt', 'speech5.txt', 'speech5.txt', 'speech5.txt', 'speech5.txt']

  - Answer (truncated): `Ambedkar characterizes the Hindu-Muslim problem in *Document 5* as a clash of nationalities rather than mere communal differences, arguing that Hindus and Muslims constitute two distinct nations due to their divergent religions, cultures, languages, ...`

- ID: 9 — Q: How does Ambedkar relate political power to social change in Documents 3 and 6?...

  - ROUGE-L: 0.141, BLEU: 0.006, CosSim: 0.536

  - Retrieved: ['speech3.txt', 'speech1.txt', 'speech3.txt', 'speech1.txt', 'speech3.txt']

  - Answer (truncated): `Ambedkar argues in Documents 3 and 6 that political power alone is insufficient for lasting change without underlying social transformation. He asserts that political democracy cannot endure unless it is built on *social democracy*—a society rooted i...`

- ID: 8 — Q: What is Ambedkar's concept of ideal society across Documents 1 and 3?...

  - ROUGE-L: 0.157, BLEU: 0.050, CosSim: 0.490

  - Retrieved: ['speech1.txt', 'speech1.txt', 'speech1.txt', 'speech2.txt', 'speech6.txt']

  - Answer (truncated): `Ambedkar's concept of an ideal society, as outlined in Documents 1 and 3, is founded on the principles of **liberty, equality, and fraternity**, rejecting the constraints of tradition and caste-based oppression. He envisions a society where individua...`


## Recommendation
Based on a combined heuristic (relevance + faithfulness + hit_rate) the recommended chunk configuration is **800_1000**.
