from fpdf import FPDF

policies = {
    "training_development_policy.pdf": """Training and Development Policy

The company is committed to supporting employee growth through continuous learning. Employees are eligible for up to 20,000 rupees per year toward external courses, certifications, or conferences relevant to their role, subject to manager approval. Internal training sessions are conducted quarterly and attendance is mandatory for all team members. Employees completing certifications relevant to their function may be eligible for a one-time skill bonus, reviewed during appraisal cycles.""",

    "data_privacy_policy.pdf": """Data Privacy Policy

The company collects and processes employee and customer data strictly for legitimate business purposes. Access to personal data is restricted to authorized personnel on a need-to-know basis. Employees handling customer data must complete annual data privacy training. Any data breach or unauthorized access must be reported to the compliance team within 24 hours of discovery. Personal data will not be shared with third parties without explicit consent, except where required by law.""",

    "travel_policy.pdf": """Business Travel Policy

Employees traveling for business purposes must book travel through the approved corporate travel portal at least five working days in advance. Economy class is standard for domestic travel; business class may be approved for international trips exceeding six hours, subject to director-level approval. Daily allowance for meals during travel is capped at 1500 rupees. All travel-related expenses must be submitted for reimbursement within 15 days of return.""",

    "performance_review_policy.pdf": """Performance Review Policy

Performance reviews are conducted twice a year, in June and December. Each employee is evaluated based on goal completion, peer feedback, and manager assessment. Ratings range from "Needs Improvement" to "Outstanding." Employees receiving an "Outstanding" rating for two consecutive cycles are eligible for accelerated promotion review. Feedback discussions must be documented and shared with the employee within one week of the review cycle closing.""",

    "equipment_policy.pdf": """Equipment and Asset Policy

The company provides a laptop and necessary peripherals to all full-time employees upon joining. Employees are responsible for the safekeeping of company-issued equipment and must report any damage or loss to IT immediately. Equipment must be returned in working condition upon resignation or termination. Requests for additional equipment, such as external monitors or ergonomic accessories, require manager approval and are subject to budget availability."""
}

for filename, content in policies.items():
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", size=12)
    pdf.multi_cell(0, 8, content)
    pdf.output(f"documents/policies/{filename}")
    print(f"Created {filename}")