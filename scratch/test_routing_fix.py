import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.getcwd(), 'supportmind', 'src'))

from ensemble_router import EnsembleRouter

def test_routing():
    router = EnsembleRouter()
    
    test_cases = [
        ("My invoice is wrong, please fix the billing error.", "billing"),
        ("The server is down and I am getting 500 errors.", "technical_support"),
        ("How do I add a new user to my account?", "account_management"),
        ("I want to cancel my subscription immediately.", "churn_risk"),
    ]
    
    print(f"{'Text':<50} | {'Expected':<20} | {'Got':<20} | {'Status'}")
    print("-" * 110)
    
    for text, expected in test_cases:
        res = router.route(text)
        got = res['top_category']
        status = "PASS" if got == expected else "FAIL"
        print(f"{text[:49]:<50} | {expected:<20} | {got:<20} | {status}")

if __name__ == "__main__":
    test_routing()
