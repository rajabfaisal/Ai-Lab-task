def check_result(score, documents_complete, attendance, assignment_submitted):
    if score < 70:
        return "Fail - Score is below 70"
    if documents_complete == False:
        return "Fail - Documents are incomplete"
    if attendance < 75:
        return "Fail - Attendance is below 75%"
    if assignment_submitted == False:
        return "Fail - Assignment is not submitted"
    return "Pass"

test_cases = [
    {
        "name": "Case 1 - Pass",
        "score": 80,
        "documents": True,
        "attendance": 85,
        "assignment": True,
        "expected": "Pass"
    },
    {
        "name": "Case 2 - Attendance Failure",
        "score": 80,
        "documents": True,
        "attendance": 60,
        "assignment": True,
        "expected": "Fail - Attendance is below 75%"
    },
    {
        "name": "Case 3 - Assignment Failure",
        "score": 80,
        "documents": True,
        "attendance": 85,
        "assignment": False,
        "expected": "Fail - Assignment is not submitted"
    },
    {
        "name": "Case 4 - Threshold Boundary",
        "score": 70,
        "documents": True,
        "attendance": 75,
        "assignment": True,
        "expected": "Pass"
    }
]

for case in test_cases:
    actual = check_result(
        case["score"],
        case["documents"],
        case["attendance"],
        case["assignment"]
    )
    print("\n", case["name"])
    print("Input:", case["score"], case["documents"],
          case["attendance"], case["assignment"])
    print("Expected Result:", case["expected"])
    print("Actual Result:", actual)
    if actual == case["expected"]:
        print("Reason: The result matches the expected result.")
    else:
        print("Reason: The result does not match the expected result.")
