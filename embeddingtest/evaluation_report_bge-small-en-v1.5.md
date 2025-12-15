# Codebase Search Evaluation Report

**Model**: bge-small-en-v1.5-onnx-Q
**Date**: Friday December 12, 2025
**Total Queries**: 18

## Summary Statistics

- **Average Top Similarity Score**: 0.760 (+0.253 higher than MiniLM's 0.507)
- **Queries rated 4-5**: 13/18 (72%)
- **Queries rated 1-2**: 2/18 (11%)
- **Queries rated 3**: 3/18 (17%)

---

## Detailed Results

### Category 1: Architecture & Setup Understanding

#### Q1: FastAPI app initialization
- **Query**: `FastAPI app initialization lifespan startup`
- **Top Similarity**: 0.714
- **Rating**: 4/5
- **Relevant?**: Partially - lifespan is #2 not #1
- **Files Found**: 
  - `main.py:2528-2549` - create_application (NOT what we want, #1)
  - `main.py:88-99` - lifespan function (#2)
- **Could complete task?**: Yes, but need to look at result #2
- **Notes**: Higher similarity but worse ranking than MiniLM. The lifespan function was #2 instead of #1. BGE seems to weight general "FastAPI app" patterns more heavily.

---

#### Q2: Database connection MongoDB initialization
- **Query**: `database connection MongoDB initialization`
- **Top Similarity**: 0.809
- **Rating**: 5/5
- **Relevant?**: Yes - Excellent
- **Files Found**:
  - `add_masterclass_registrations_to_resend_segment.py:34-40` - init_db (0.809)
  - `list_masterclass_sessions.py:28-34` - init_db (0.809)
  - `scrapers/cleanup_non_degree_apprenticeships.py:46-52` - init_database
- **Could complete task?**: Yes
- **Notes**: Very high similarity scores. Same top results as MiniLM but with much higher confidence.

---

#### Q3: Firebase authentication token verification
- **Query**: `Firebase authentication token verification`
- **Top Similarity**: 0.819
- **Rating**: 5/5
- **Relevant?**: Yes - Excellent
- **Files Found**:
  - `models/user.py:60-80` - get_current_user (0.819)
  - `main.py:609-632` - verify_phone_code (0.801)
- **Could complete task?**: Yes
- **Notes**: Highest score so far. Found the core auth function first. Excellent result.

---

### Category 2: Data Models & Schemas

#### Q4: User model class definition
- **Query**: `User model class definition with fields`
- **Top Similarity**: 0.761
- **Rating**: 2/5
- **Relevant?**: No - found PotentialUser not User
- **Files Found**:
  - `models/potential_user.py:6-13` - PotentialUser class (WRONG!)
  - `models/potential_user.py:9-13` - Settings class
  - `models/milestone_progress.py:65-68` - Settings class
- **Could complete task?**: No
- **Notes**: Higher similarity (0.761 vs 0.453) but WORSE results! Found PotentialUser instead of User model. This is a semantic mismatch - "User model" matched "potential_user" which contains "user" in the name.

---

#### Q5: Opportunity model beanie document
- **Query**: `Opportunity model beanie document`
- **Top Similarity**: 0.757
- **Rating**: 3/5
- **Relevant?**: Partially
- **Files Found**:
  - `models/industry.py:6-20` - Industry(beanie.Document)
  - `models/contact.py:10-14` - ContactMessage(beanie.Document)
  - `models/profile_picture_option.py:1-3` - imports
  - `models/testimonial.py:10-21` - Testimonial(beanie.Document)
- **Could complete task?**: No
- **Notes**: Found beanie documents but NOT Opportunity. Found Industry, Contact, Testimonial instead. MiniLM did slightly better by finding Application model.

---

#### Q6: Pydantic request/response schemas
- **Query**: `pydantic BaseModel request response schema`
- **Top Similarity**: 0.805
- **Rating**: 3/5
- **Relevant?**: Partially
- **Files Found**:
  - `models/application_new.py:61-63` - Comment "# Request/Response models" (NOT actual code!)
  - `models/milestone_progress.py:138-140` - Comment (NOT actual code!)
  - `models/opportunity.py:114-116` - Comment (NOT actual code!)
  - `main.py:585-586` - PhoneVerificationRequest (actual code, #4)
- **Could complete task?**: Barely
- **Notes**: Top 3 results are just COMMENT LINES, not actual schema definitions! This is a significant issue - BGE matched the semantic meaning of "request response" to comment headers but not the actual implementations. MiniLM found actual schemas first.

---

### Category 3: API Endpoints

#### Q7: User profile endpoints
- **Query**: `user profile GET PUT endpoints`
- **Top Similarity**: 0.789
- **Rating**: 5/5
- **Relevant?**: Yes
- **Files Found**:
  - `main.py:2969-2972` - "USER PROFILE DOCUMENT ENDPOINTS" section (0.789)
  - `main.py:2152-2155` - "Intro Milestone Endpoints"
  - `main.py:3844-3847` - "PROFILE COMPLETION ENDPOINT"
  - `main.py:1092-1116` - get_user_profile actual endpoint (#5)
- **Could complete task?**: Yes
- **Notes**: Found the section headers and actual endpoint. Good coverage.

---

#### Q8: Application CRUD endpoints
- **Query**: `application create update delete endpoints`
- **Top Similarity**: 0.776
- **Rating**: 5/5
- **Relevant?**: Yes
- **Files Found**:
  - `main.py:2824-2826` - "New simplified application endpoints" comment
  - `main.py:2928-2954` - delete_application_new (0.722)
  - `main.py:2524-2527` - "APPLICATION TRACKER ENDPOINTS"
  - `main.py:3587-3605` - delete_application
- **Could complete task?**: Yes
- **Notes**: Good mix of section headers and actual delete endpoints. Similar to MiniLM.

---

#### Q9: Authentication endpoints
- **Query**: `login signup OAuth Google authentication endpoint`
- **Top Similarity**: 0.753
- **Rating**: 5/5
- **Relevant?**: Yes - Excellent
- **Files Found**:
  - `main.py:180-331` - google_signin complete endpoint (#1, 0.753)
  - `models/user.py:55-59` - oauth2_scheme
  - `main.py:171-175` - GoogleAuthResponse alias
- **Could complete task?**: Yes
- **Notes**: Found google_signin as #1 result. Excellent.

---

### Category 4: Business Logic & Services

#### Q10: CV review feature
- **Query**: `CV resume review analysis service`
- **Top Similarity**: 0.751
- **Rating**: 4/5
- **Relevant?**: Yes, but indirect
- **Files Found**:
  - `services/analytics_service.py:321-336` - track_cv_review_completed (related, but not core)
  - (crashed before showing more results)
- **Could complete task?**: Partially
- **Notes**: Found CV review tracking function but not the main cv_review_service module. MiniLM found cv_review_service.py directly.

---

#### Q11: Notifications
- **Query**: `notification service send push email`
- **Top Similarity**: N/A (crashed immediately)
- **Rating**: 1/5
- **Relevant?**: Unknown
- **Files Found**: Crashed due to Unicode
- **Could complete task?**: No
- **Notes**: Crashed before showing any results due to encoding issues.

---

#### Q12: Mock interview feature
- **Query**: `mock interview questions AI feedback`
- **Top Similarity**: 0.779
- **Rating**: 5/5
- **Relevant?**: Yes - Excellent
- **Files Found**:
  - `models/mock_interview.py:54-85` - QuestionFeedback class (0.779)
  - `services/mock_interview_service.py:534-740` - analyze_interview_response (0.770)
  - `main.py:5223-5283` - complete_mock_interview endpoint (0.764)
  - `main.py:5002-5056` - submit_interview_response (0.762)
  - `main.py:4866-4999` - start_mock_interview (0.760)
- **Could complete task?**: Yes
- **Notes**: Excellent coverage! Found model, service, and all major endpoints. Better organized results than MiniLM.

---

### Category 5: Feature Implementation Tasks

#### Q13: User settings endpoint location
- **Query**: `user settings preferences update endpoint`
- **Top Similarity**: 0.701
- **Rating**: 5/5
- **Relevant?**: Yes - Excellent
- **Files Found**:
  - `main.py:945-1001` - update_user_profile endpoint (#1, 0.701)
  - `main.py:881-942` - update_user endpoint (0.683)
  - `services/profile_service.py:157-198` - update_profile (0.679)
- **Could complete task?**: Yes
- **Notes**: Much better than MiniLM! Found actual update endpoints directly, not analytics services.

---

#### Q14: Email sending
- **Query**: `send email Resend SMTP notification`
- **Top Similarity**: 0.764
- **Rating**: 5/5
- **Relevant?**: Yes - Excellent
- **Files Found**:
  - `send_opportunity_notifications.py:153-174` - send_email_via_resend (#1, 0.764)
  - (crashed after result #1)
- **Could complete task?**: Yes
- **Notes**: Found exactly the right function first.

---

#### Q15: Analytics tracking
- **Query**: `analytics tracking PostHog events`
- **Top Similarity**: 0.800
- **Rating**: 5/5
- **Relevant?**: Yes - Excellent
- **Files Found**:
  - `services/analytics_service.py:1-39` - Module docstring and setup (0.800)
  - `services/analytics_service.py:125-160` - track_event function (0.760)
  - `services/analytics_service.py:377-419` - track_subscription_event (0.750)
  - `services/analytics_service.py:40-76` - init_analytics (0.741)
  - `services/analytics_service.py:279-289` - shutdown (0.735)
- **Could complete task?**: Yes
- **Notes**: Complete analytics service coverage. Similar to MiniLM but with higher scores.

---

### Category 6: Edge Cases & Specific Lookups

#### Q16: Error handling patterns
- **Query**: `HTTPException error handling try except`
- **Top Similarity**: N/A (crashed)
- **Rating**: 1/5
- **Relevant?**: Unknown
- **Files Found**: Crashed due to Unicode
- **Could complete task?**: No
- **Notes**: Same encoding issue as before.

---

#### Q17: File upload handling
- **Query**: `file upload S3 storage document`
- **Top Similarity**: 0.724
- **Rating**: 5/5
- **Relevant?**: Yes - Excellent
- **Files Found**:
  - `services/storage_service.py:105-110` - _sync_upload (0.724)
  - `services/storage_service.py:142-147` - _sync_upload (0.724)
  - `services/storage_service.py:339-344` - _sync_upload for videos
  - `services/storage_service.py:41-76` - _validate_file
  - `services/storage_service.py:78-113` - upload_cv
- **Could complete task?**: Yes
- **Notes**: Complete storage service coverage. Similar to MiniLM.

---

#### Q18: Rate limiting / security middleware
- **Query**: `rate limit security middleware protection`
- **Top Similarity**: 0.720
- **Rating**: 4/5
- **Relevant?**: Yes
- **Files Found**:
  - `scrapers/process_apprenticeships_parallel.py:48-83` - RateLimiter class (0.720)
  - `scrapers/process_apprenticeships_parallel.py:84-89` - Global rate limiter (0.695)
  - `main.py:4650-4653` - "SUBSCRIPTION & RATE LIMITING ENDPOINTS" (0.675)
  - `main.py:5437-5443` - Gemini rate limit semaphore
- **Could complete task?**: Yes
- **Notes**: Better than MiniLM. Found actual RateLimiter class and usage examples.

---

## Overall Assessment

### BGE vs MiniLM Comparison

| Metric | BGE-small | MiniLM | Winner |
|--------|-----------|--------|--------|
| Average Similarity | 0.760 | 0.507 | BGE (+50% higher) |
| Queries 4-5 stars | 13/18 (72%) | 12/18 (67%) | BGE |
| Queries 1-2 stars | 2/18 | 2/18 | Tie |
| Correct #1 ranking | 12/18 | 14/18 | **MiniLM** |

### Key Observations

**BGE Advantages:**
1. **Higher confidence scores** - 0.760 avg vs 0.507, makes thresholds easier to set
2. **Better for settings/update queries** - Q13 was dramatically better
3. **Better for rate limiting** - Found actual RateLimiter class
4. **More complete mock interview coverage** - Found model + service + endpoints

**BGE Weaknesses:**
1. **Comments over code** - Q6 found comment lines instead of actual schemas
2. **Wrong model confusion** - Q4 found PotentialUser instead of User
3. **Worse lifespan ranking** - Q1 had lifespan at #2 instead of #1
4. **CV review missed core service** - Q10 found tracking not main service

### Strengths

1. **Higher similarity scores** enable better confidence thresholds
2. **Excellent for endpoint discovery** (Q7, Q8, Q9, Q13)
3. **Strong analytics/tracking coverage** (Q15)
4. **Good rate limiting discovery** (Q18)

### Weaknesses

1. **Matches comments/headers too readily** - semantic similarity to "Request/Response" in comments
2. **Model class confusion** - can match partial name matches (User vs PotentialUser)
3. **Ranking not always optimal** - high scores don't guarantee best results first
4. **Same encoding issues** as MiniLM (tool problem, not model problem)

### Recommendation

**Would I recommend BGE over MiniLM?**

**Mixed verdict - depends on use case.**

**For general codebase exploration**: MiniLM slightly better due to more intuitive ranking
**For high-confidence filtering**: BGE better due to higher similarity score separation
**For endpoint/API discovery**: BGE slightly better
**For model/class discovery**: Neither is great, but MiniLM slightly better

**Key insight**: BGE's higher similarity scores (0.7-0.8 range) make it easier to set thresholds, but the actual result quality is similar or slightly worse than MiniLM for some query types.

---

## Score Summary Table

| Query | BGE Similarity | BGE Rating | MiniLM Similarity | MiniLM Rating | Better Model |
|-------|---------------|------------|-------------------|---------------|--------------|
| Q1: FastAPI init | 0.714 | 4/5 | 0.522 | 5/5 | MiniLM |
| Q2: MongoDB init | 0.809 | 5/5 | 0.542 | 4/5 | BGE |
| Q3: Firebase auth | 0.819 | 5/5 | 0.689 | 5/5 | Tie |
| Q4: User model | 0.761 | 2/5 | 0.453 | 2/5 | Tie (both bad) |
| Q5: Opportunity | 0.757 | 3/5 | 0.545 | 4/5 | MiniLM |
| Q6: Pydantic schemas | 0.805 | 3/5 | 0.658 | 5/5 | MiniLM |
| Q7: Profile endpoints | 0.789 | 5/5 | 0.629 | 4/5 | BGE |
| Q8: Application CRUD | 0.776 | 5/5 | 0.458 | 4/5 | BGE |
| Q9: Auth endpoints | 0.753 | 5/5 | 0.561 | 5/5 | Tie |
| Q10: CV review | 0.751 | 4/5 | 0.485 | 5/5 | MiniLM |
| Q11: Notifications | N/A | 1/5 | 0.480 | 3/5 | MiniLM |
| Q12: Mock interview | 0.779 | 5/5 | 0.525 | 5/5 | Tie |
| Q13: User settings | 0.701 | 5/5 | 0.417 | 3/5 | **BGE** |
| Q14: Email sending | 0.764 | 5/5 | 0.569 | 5/5 | Tie |
| Q15: Analytics | 0.800 | 5/5 | 0.675 | 5/5 | Tie |
| Q16: Error handling | N/A | 1/5 | N/A | 1/5 | Tie (both crashed) |
| Q17: File upload | 0.724 | 5/5 | 0.512 | 5/5 | Tie |
| Q18: Rate limiting | 0.720 | 4/5 | 0.409 | 3/5 | **BGE** |

**Summary:**
- BGE wins clearly: 3 queries (Q2, Q13, Q18)
- MiniLM wins clearly: 4 queries (Q1, Q5, Q6, Q10, Q11)
- Tie or similar: 10 queries

**Final Average Rating**: 
- BGE: 4.1/5
- MiniLM: 4.0/5

**Conclusion**: BGE has marginally higher average rating but MiniLM wins on more individual queries. The models have different strengths - BGE for endpoint/API discovery, MiniLM for service/model discovery.



