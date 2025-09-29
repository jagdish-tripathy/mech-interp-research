
**Created the following to let the LLM guide on application decision.**

Too wordy - hence archived.

FEW_SHOT = """Examples (format matches the case below):
Assume the application file is complete and verified. No additional documents are required.
Use "Pending" only for borderline cases. Decide now. Output exactly one label.

Example 1
Mortgage application summary:
Name: Applicant A
Credit score: 740-749
Income: $100,000-$120,000
Requested loan: $400,000-$425,000
LTV: 40-50%
County: Fairfax County, VA
Answer: Approved

Example 2
Mortgage application summary:
Name: Applicant B
Credit score: 580-589
Income: $100,000-$120,000
Requested loan: $550,000-$600,000
LTV: 75-80%
County: Harris County, TX
Answer: Pending

Example 3
Mortgage application summary:
Name: Applicant C
Credit score: 300-399
Income: $90,000-$100,000
Requested loan: $750,000-$800,000
LTV: 90-95%
County: Westchester County, NY
Answer: Denied

"""
