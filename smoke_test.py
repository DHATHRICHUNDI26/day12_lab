
from rag import index_policy, retrieve, answer_question

test_cases = [
    {
        "id": "Q1",
        "question": "How many casual leave days are provided annually?",
        "expected": [["12"]]
    },
    {
        "id": "Q2",
        "question": "How many sick leave days are provided annually?",
        "expected": [["8"]]
    },
    {
        "id": "Q3",
        "question": "When are casual and sick leaves credited?",
        "expected": [["January 1", "1 January"]]
    },
    {
        "id": "Q4",
        "question": "How early should employees apply for casual leave?",
        "expected": [["2 working days"]]
    },
    {
        "id": "Q5",
        "question": "When can employees apply for sick leave?",
        "expected": [["same day"], ["2 working days"]]
    },
    {
        "id": "Q6",
        "question": "Is half-day casual leave allowed?",
        "expected": [["half-day", "half day"], ["allowed"]]
    },
    {
        "id": "Q7",
        "question": "Is half-day sick leave allowed?",
        "expected": [["half-day", "half day"], ["allowed"]]
    },
    {
        "id": "Q8",
        "question": "Who approves leave requests?",
        "expected": [["reporting manager"]]
    },
    {
        "id": "Q9",
        "question": "How many unused casual leave days can be carried forward?",
        "expected": [["5"]]
    },
    {
        "id": "Q10",
        "question": "When must carried-forward casual leave be used?",
        "expected": [["March 31", "31 March"]]
    },
    {
        "id": "Q11",
        "question": "Can sick leave be carried forward?",
        "expected": [["cannot", "not allowed", "no"]]
    },
    {
        "id": "Q12",
        "question": "What happens to unused carried-forward casual leave after March 31?",
        "expected": [["lapse", "expire"]]
    },
    {
        "id": "Q13",
        "question": "How many days of maternity leave are provided?",
        "expected": [["not specified", "not available"]]
    },
    {
        "id": "Q14",
        "question": "What percentage of salary is deducted for Loss of Pay?",
        "expected": [["not specified", "not available"]]
    },
    {
        "id": "Q15",
        "question": "Can employees work from home while on sick leave?",
        "expected": [["not specified", "not available"]]
    }
]


def check_answer(answer, expected_groups):
    answer = answer.lower()

    for group in expected_groups:
        if not any(word.lower() in answer for word in group):
            return False

    return True


def run_smoke_tests():

    print("\n========== HR LEAVE POLICY SMOKE TEST ==========")

    index_policy()

    results = []
    log_lines = []

    for test in test_cases:

        print(f"\nRunning {test['id']}: {test['question']}")

        try:
            chunks = retrieve(test["question"])

            answer = answer_question(test["question"])

            passed = check_answer(
                answer,
                test["expected"]
            )

            status = "PASS" if passed else "FAIL"

            sources = list(set(
                chunk["source"] + " - " + chunk["section"]
                for chunk in chunks
            ))

            print("Retrieved chunks:", len(chunks))
            print("Sources:", sources)
            print("Test Status:", status)

            log_lines.append(
                f"{test['id']} | {test['question']}\n"
                f"Expected: {test['expected']}\n"
                f"Actual: {answer}\n"
                f"Retrieved Sources: {sources}\n"
                f"Status: {status}\n"
                f"{'-' * 60}\n"
            )

            results.append(status)

        except Exception as e:

            print("Test Status: ERROR")
            print("Error:", e)

            log_lines.append(
                f"{test['id']} | ERROR: {e}\n"
                f"Status: ERROR\n"
                f"{'-' * 60}\n"
            )

            results.append("ERROR")

    passed = results.count("PASS")
    failed = results.count("FAIL")
    errors = results.count("ERROR")

    summary = f"""
========== SMOKE TEST SUMMARY ==========

Total Tests : {len(test_cases)}
Passed      : {passed}
Failed      : {failed}
Errors      : {errors}

Overall Result: {"PASS" if passed == len(test_cases) else "FAIL"}
"""

    print(summary)

    with open("test_log.txt", "w", encoding="utf-8") as file:
        file.writelines(log_lines)
        file.write(summary)

    print("Test results saved to test_log.txt")


if __name__ == "__main__":
    run_smoke_tests()
