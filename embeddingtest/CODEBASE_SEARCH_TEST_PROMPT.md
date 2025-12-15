# Codebase Semantic Search Evaluation Test

## Instructions for AI Model

You are testing a semantic search tool (`query_codebase.py`) that searches an embedded codebase. Your task is to:

1. Run each query below using the terminal command provided
2. Evaluate the quality of results for each query
3. Document your findings in a structured report
4. Save your report to `evaluation_report_[MODEL_NAME].md`

**IMPORTANT**: Use `--code-only` flag for all queries to filter to source code only.

---

## Test Queries to Run

Run each of these commands and evaluate the results. For each query, note:
- Top 3 similarity scores
- Whether results are relevant to the query
- Whether you could use this to complete a development task
- Any issues or observations

### Category 1: Architecture & Setup Understanding

```bash
# Q1: How is the FastAPI application initialized?
python query_codebase.py "FastAPI app initialization lifespan startup" --code-only -n 5

# Q2: What database is used and how is it connected?
python query_codebase.py "database connection MongoDB initialization" --code-only -n 5

# Q3: How is authentication configured?
python query_codebase.py "Firebase authentication token verification" --code-only -n 5
```

### Category 2: Data Models & Schemas

```bash
# Q4: What is the User model structure?
python query_codebase.py "User model class definition with fields" --code-only -n 5

# Q5: What is the Application/Opportunity model?
python query_codebase.py "Opportunity model beanie document" --code-only -n 5

# Q6: Find all Pydantic request/response models
python query_codebase.py "pydantic BaseModel request response schema" --code-only -n 5
```

### Category 3: API Endpoints

```bash
# Q7: Find user profile endpoints
python query_codebase.py "user profile GET PUT endpoints" --code-only -n 5

# Q8: Find application/job tracking endpoints
python query_codebase.py "application create update delete endpoints" --code-only -n 5

# Q9: Find authentication endpoints
python query_codebase.py "login signup OAuth Google authentication endpoint" --code-only -n 5
```

### Category 4: Business Logic & Services

```bash
# Q10: How does the CV review feature work?
python query_codebase.py "CV resume review analysis service" --code-only -n 5

# Q11: How do notifications work?
python query_codebase.py "notification service send push email" --code-only -n 5

# Q12: How does the mock interview feature work?
python query_codebase.py "mock interview questions AI feedback" --code-only -n 5
```

### Category 5: Feature Implementation Tasks

Imagine you need to implement these features. Can the search help you find where to add code?

```bash
# Q13: Where would I add a new API endpoint for user settings?
python query_codebase.py "user settings preferences update endpoint" --code-only -n 5

# Q14: Where is email sending handled?
python query_codebase.py "send email Resend SMTP notification" --code-only -n 5

# Q15: Where would I add analytics tracking?
python query_codebase.py "analytics tracking PostHog events" --code-only -n 5
```

### Category 6: Edge Cases & Specific Lookups

```bash
# Q16: Find error handling patterns
python query_codebase.py "HTTPException error handling try except" --code-only -n 5

# Q17: Find file upload handling
python query_codebase.py "file upload S3 storage document" --code-only -n 5

# Q18: Find rate limiting or security middleware
python query_codebase.py "rate limit security middleware protection" --code-only -n 5
```

---

## Evaluation Criteria

For each query result, rate on a scale of 1-5:

| Score | Meaning |
|-------|---------|
| 5 | Perfect - Exactly what I needed, could complete task immediately |
| 4 | Good - Relevant results, minor additional searching needed |
| 3 | Okay - Partially relevant, would need more queries |
| 2 | Poor - Mostly irrelevant results |
| 1 | Failed - No useful results |

---

## Report Template

Create your report with this structure:

```markdown
# Codebase Search Evaluation Report

**Model**: [Your Model Name]
**Date**: [Current Date]
**Total Queries**: 18

## Summary Statistics

- Average Similarity Score: X.XX
- Queries rated 4-5: X/18
- Queries rated 1-2: X/18

## Detailed Results

### Q1: FastAPI app initialization
- **Top Similarity**: X.XXX
- **Rating**: X/5
- **Relevant?**: Yes/No
- **Files Found**: [list key files]
- **Could complete task?**: Yes/No
- **Notes**: [observations]

[Repeat for all 18 queries]

## Overall Assessment

### Strengths
- [List what worked well]

### Weaknesses  
- [List what didn't work]

### Recommendation
[Would you recommend this tool as "Cursor on a budget"? Why/why not?]
```

---

## How to Run This Test

1. Make sure you're in the `embeddingtest` directory
2. Activate the virtual environment: `.\venv\Scripts\activate`
3. Run each query command above
4. Document results in your report
5. Save report as: `evaluation_report_[MODEL_NAME].md`

Example: `evaluation_report_claude_opus_4.md` or `evaluation_report_gpt4o.md`

---

## After Completing the Test

Tell the user:
1. Your overall impression
2. The average quality score
3. Whether this would be useful for development tasks
4. Key differences you noticed (if comparing to another model's results)



