# POC 4 — RAG Q&A Bot: Evaluation Table

**Instructions:** Run each question below in the Streamlit app. Fill in the actual generated answer, the top retrieval distance (shown in "Show retrieved chunks"), and mark Pass/Fail based on whether the behavior matches what's expected.

---

| # | Question | Expected Behavior | Actual Answer | Top Distance | Pass/Fail | Notes |
|---|----------|--------------------|----------------|---------------|-----------|-------|
| 1 | How many leave days do employees get per year? | Correct answer with citation to leave_policy.txt | | | | |
| 2 | Can I work from home every day? | Should clarify it's limited to 2 days/week, not every day | | | | |
| 3 | What are the skills mentioned in the resume? | Lists skills from sample_resume.txt with citation | | | | |
| 4 | What is the capital of France? | "I don't know" — completely irrelevant to documents | | | | |
| 5 | How is employee data protected? | Answer grounded in data_privacy_policy.pdf | | | | |
| 6 | What certifications does the candidate have? | Lists certifications from resume with citation | | | | |
| 7 | Is training mandatory? | Answer grounded in training_development_policy.pdf | | | | |
| 8 | What's the process to request work from home? | Correct procedural detail (e.g. submit 1 day in advance) | | | | |
| 9 | What is the employee's salary? | "I don't know" — info not present in any document (hallucination check) | | | | |
| 10 | Summarize the leave and remote work policies together | Coherent answer combining both documents with citations | | | | |

---

## Summary

- **Total questions tested:** 10
- **Passed:** ___ / 10
- **Failed:** ___ / 10
- **Key observations:**
  - (e.g. "Threshold of 1.5 correctly separated relevant vs irrelevant questions in most cases")
  - (e.g. "Statement-phrased questions needed prompt adjustment to avoid false 'I don't know' responses")
  - (e.g. "Multi-document synthesis question worked well / had issues combining sources")

## Failure modes observed (if any)

- [ ] Wrong chunks retrieved
- [ ] Model ignored provided context
- [ ] Incomplete answer (chunk cut off relevant info)
- [ ] Vocabulary mismatch between question and document wording
- [ ] Overconfident wrong answer (hallucination instead of "I don't know")
