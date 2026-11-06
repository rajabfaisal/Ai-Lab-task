def check_eligibility(score, docs_complete, attendance, income):
    reasons = []
    if score < 70:
        reasons.append("Score is below 70")
    if not docs_complete:
        reasons.append("Documents are incomplete")
    if attendance < 75:
        reasons.append("Attendance is below 75%")
    if income > 50000:
        reasons.append("Family income is above 50000")
    if len(reasons) > 0:
        return False, ", ".join(reasons)
    return True, "All conditions are satisfied"

cases = [
    ["Case 1: Normal pass", 85, True, 90, 30000, True],
    ["Case 2: Fails attendance condition", 85, True, 60, 30000, False],
    ["Case 3: Fails family income condition", 85, True, 90, 80000, False],
    ["Case 4: Threshold boundary (score = 70)", 70, True, 75, 50000, True],
]

for case in cases:
    name, score, docs, attendance, income, expected = case
    actual, reason = check_eligibility(score, docs, attendance, income)
    print(name)
    print("Input:", score, docs, attendance, income)
    print("Expected:", "Eligible" if expected else "Not eligible")
    print("Actual:", "Eligible" if actual else "Not eligible")
    print("Reason:", reason)
    if actual == expected:
        print("Result: PASS")
    else:
        print("Result: FAIL")
    print()
