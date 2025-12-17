1. Define the Stakeholder's Goal (The "Why")
Before you choose any metric, you must clearly define what "success" looks like to the person using the tool (the Recruiter, in your case).
Recruiter's Goal,Metrics to Consider
"""I need quick filtering.""","Classification (e.g., Is this resume ""Qualified"" or ""Unqualified""?)"
"""I need to rank candidates.""","Regression (e.g., A score from 0-10, as you built)"
"""I need to understand what they are good at.""","Clustering/Embedding (e.g., Assigning the resume to a specific job category)"

2. Analysis Methodology Options (The "How")
You should choose the simplest method that satisfies the stakeholder's goal.
Methodology,Metric Focus,When to Choose It
A. Simple Keyword Matching,Count: Total number of matching keywords.,"Simple Screening: Use this for a first-pass filter (e.g., filtering out anyone who doesn't list Python). Low value."
B. TF-IDF Score (Your Bonus Score),Rarity: The uniqueness of words (High IDF).,"Specialization: Use this when you only care about identifying rare, high-value skills, not general job fit."
C. Centroid Distance (Your Base Score),Fit/Relevance: Distance to the learned job profile center.,General Fit: Use this when you care about how well-rounded the candidate is for the role's average skill set. Ignores specialization.
D. Ensemble/Combined Metric (Your Final Score),Weighted Combination: Combines two or more simple metrics.,"Complex Ranking: Use this when a single metric isn't enough (e.g., you need to reward both Fit (Base Score) and Specialization (Bonus Score)). Highest value."

3. The Decision Point: Why You Chose the Combined Metric
You chose the Combined Metric (D) because neither the Base Score alone nor the Bonus Score alone answered the recruiter's need:
Metric,Metric's Weakness (Why not use it alone?)
Base Score (Centroid Distance),"A candidate could be a perfect ""Software Engineer"" fit (high Base Score) but lack any modern, specialized skills (e.g., Docker or AWS)."
Bonus Score (IDF Rarity),"A candidate could have rare, high-IDF keywords (e.g., ""Mandrake Linux"" or ""Cobol"") but for a role that is totally irrelevant to the target job."