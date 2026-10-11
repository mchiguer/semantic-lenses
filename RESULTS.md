# Results — Rules (offline)

TF-IDF baseline threshold (tuned on DEV only): **0.805**.
The lenses have no threshold tuned on this data (fixed thresholds in `lenses/base.py`).

## DEV (used to write the rules) — 30 pairs

| Family | n | TF-IDF baseline | Lenses |
|---|---|---|---|
| Near-identical, different meaning (should not merge) | 17 | 17/17 | 17/17 |
| Paraphrases, different words (should merge) | 11 | 0/11 | 11/11 |
| Out of the lenses' reach (should not merge) | 2 | 2/2 | 0/2 |
| **Total** | 30 | **19/30** | **28/30** |

Reference: always giving the same answer would score 19/30. AUC of the TF-IDF similarity for "same meaning": 0.11 (0.5 = chance; below that, similarity ranks the pairs backwards).

Explanation: of 17 near-identical pairs flagged as different by the lenses, the right dimension is found exactly in 17 cases (included in 17).

<details><summary>Error details</summary>

| id | a | b | gold | baseline (sim) | lenses |
|---|---|---|---|---|---|
| P01 | The number of malaria cases is rising in the district. | Malaria cases are on the rise in the district. | same meaning | different (0.38) | same |
| P02 | No patient was received yesterday. | The clinic received nobody yesterday. | same meaning | different (0.18) | same |
| P03 | Flights will resume on Monday. | The resumption of flights will take place on Monday. | same meaning | different (0.39) | same |
| P04 | Why does the server crash? | What is the cause of the server crashes? | same meaning | different (0.15) | same |
| P05 | I think the model is reliable. | In my opinion, the model is reliable. | same meaning | different (0.46) | same |
| P06 | The budget was voted in December. | Lawmakers voted on the budget in December. | same meaning | different (0.49) | same |
| P07 | The camp no longer has drinking water. | Refugees no longer have access to drinking water. | same meaning | different (0.40) | same |
| P08 | Air traffic is collapsing. | Air traffic is plunging. | same meaning | different (0.61) | same |
| P09 | You must send the report before Friday. | Please send the report before Friday. | same meaning | different (0.67) | same |
| P10 | The team leaves on a mission tomorrow. | Tomorrow, the team will be on its way for the mission. | same meaning | different (0.26) | same |
| P11 | The patient was operated on yesterday. | The patient's operation took place yesterday. | same meaning | different (0.18) | same |
| L01 | The doctor examined the patient. | The patient examined the doctor. | different meaning | different (0.80) | same |
| L02 | The clinic received 200 patients. | The clinic received 20 patients. | different meaning | different (0.50) | same |

</details>

## TEST (not used to tune anything) — 20 pairs

| Family | n | TF-IDF baseline | Lenses |
|---|---|---|---|
| Near-identical, different meaning (should not merge) | 12 | 12/12 | 12/12 |
| Paraphrases, different words (should merge) | 6 | 0/6 | 6/6 |
| Out of the lenses' reach (should not merge) | 2 | 1/2 | 0/2 |
| **Total** | 20 | **13/20** | **18/20** |

Reference: always giving the same answer would score 14/20. AUC of the TF-IDF similarity for "same meaning": 0.19 (0.5 = chance; below that, similarity ranks the pairs backwards).

Explanation: of 12 near-identical pairs flagged as different by the lenses, the right dimension is found exactly in 12 cases (included in 12).

<details><summary>Error details</summary>

| id | a | b | gold | baseline (sim) | lenses |
|---|---|---|---|---|---|
| TP1 | No team arrived on site. | Nobody came to the area. | same meaning | different (0.00) | same |
| TP2 | The number of patients is climbing fast. | Patients are becoming more and more numerous. | same meaning | different (0.05) | same |
| TP3 | In my view, the project is too risky. | To me, the project is too risky. | same meaning | different (0.58) | same |
| TP4 | When will the bridge be repaired? | On what date will the bridge be repaired? | same meaning | different (0.62) | same |
| TP5 | The funds will be paid next month. | The payment of the funds will take place next month. | same meaning | different (0.45) | same |
| TP6 | The virus was detected on the server. | Malware was detected on the server. | same meaning | different (0.68) | same |
| TL1 | The NGO sent food to the hospital. | The hospital sent food to the NGO. | different meaning | same (0.88) | same |
| TL2 | The price rose by 5%. | The price rose by 50%. | different meaning | different (0.68) | same |

</details>

## Honest reading

- The corpora are small and hand-written: read differences as numbers of pairs, not percentages.
- The rules were written with DEV in view; TEST was not used to tune anything but was written by the same author at the same time, so it measures generalisation, not independence. The real test is pairs you write yourself.
- The baseline is lexical (TF-IDF), not neural: it is a reference point, not an upper bound.
- The "out of reach" families (swapped roles, changed quantities) show what the lenses cannot see.
