import requests, json

tests = [
    "welcome to my channel",
    "hi",
    "My server is down and we are losing $10000 per hour URGENT",
    "I was charged twice for my subscription last month",
    "subscribe and like for more content",
    "Can you help me reset my password?",
]

for text in tests:
    r = requests.post("http://localhost:7860/route", json={"text": text})
    d = r.json()
    print(f"--- '{text}' ---")
    print(f"  Action:     {d.get('action')}")
    print(f"  SLA Risk:   {d.get('sla_breach_probability')}")
    print(f"  Confidence: {d.get('confidence')}")
    print(f"  Entropy:    {d.get('entropy')}")
    print(f"  Top:        {d.get('top_category')}")
    if d.get('error_type'):
        print(f"  Error Type: {d.get('error_type')}")
    print()
