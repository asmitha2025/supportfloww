# SupportMind — Data Preprocessing Pipeline
# Loads raw data, cleans, splits, and saves processed datasets
# Run: python data/preprocess.py

import os
import csv
import random
import hashlib

random.seed(42)

CATEGORIES = [
    'billing', 'technical_support', 'account_management', 'feature_request',
    'compliance_legal', 'onboarding', 'general_inquiry', 'churn_risk'
]
CATEGORY_MAP = {cat: i for i, cat in enumerate(CATEGORIES)}

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(BASE_DIR, 'raw')
PROC_DIR = os.path.join(BASE_DIR, 'processed')


def load_raw_tickets():
    """Load tickets from raw CSV."""
    tickets = []
    csv_path = os.path.join(RAW_DIR, 'support_tickets.csv')
    if not os.path.exists(csv_path):
        print(f"⚠️  Raw data not found at {csv_path}")
        print("   Generating synthetic training data instead...")
        return generate_synthetic_data()

    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            tickets.append({
                'text': row['text'],
                'label': CATEGORY_MAP.get(row['category'], 6),
                'category': row['category'],
            })
    return tickets


def generate_synthetic_data():
    """Generate synthetic support tickets for training."""
    templates = {
        'billing': [
            "My invoice shows {amount} but my plan should be {plan_amount}.",
            "I was double-charged for {feature} this month.",
            "Please process a refund for the overcharge on my last invoice.",
            "How do I update my payment method to a different credit card?",
            "The subscription renewal charge is incorrect.",
            "I need an itemized breakdown of charges for Q{quarter}.",
            "Tax was applied incorrectly on our tax-exempt account.",
            "We need to change our billing cycle from monthly to annual.",
            "The promo code {code} didn't apply the discount correctly.",
            "Can you send a duplicate invoice for our records?",
        ],
        'technical_support': [
            "The {feature} endpoint returns a {error_code} error.",
            "Dashboard has been loading slowly for the past {days} days.",
            "The export function hangs at {percent}% and never completes.",
            "API timeout errors when processing batches larger than {size}.",
            "The {feature} integration stopped working after the latest update.",
            "Getting CORS errors when making API calls from our frontend.",
            "The search function returns irrelevant results since last deploy.",
            "Webhook deliveries are failing with SSL certificate errors.",
            "Mobile app crashes when viewing {section} on {device}.",
            "Data sync between the web app and mobile is inconsistent.",
        ],
        'account_management': [
            "How do I transfer organization ownership to another admin?",
            "I need to add {count} new users to our team account.",
            "Please change the primary email address on our account.",
            "How do I set up role-based access for different team members?",
            "We need to merge two separate organization accounts.",
            "Remove user {email} from our organization immediately.",
            "How do I configure single sign-on for our team?",
            "We need to update our company name in the system.",
            "Can I restrict API access to specific team members?",
            "How do I revoke access for a former employee?",
        ],
        'feature_request': [
            "Would it be possible to add {feature} to the dashboard?",
            "We need an API endpoint for bulk {operation}.",
            "Can you add support for {format} file exports?",
            "It would be great to have {feature} as a built-in option.",
            "We'd like to see integration support for {tool}.",
            "Can you add two-factor authentication via {method}?",
            "We need custom fields for {entity} categorization.",
            "Please add a dark mode option to the interface.",
            "Can you implement scheduled {action} functionality?",
            "We'd like to have real-time collaboration features.",
        ],
        'compliance_legal': [
            "We need your SOC 2 compliance report for our annual audit.",
            "Please provide documentation on your GDPR data handling.",
            "We need a signed data processing agreement before proceeding.",
            "What is your data retention policy for deleted accounts?",
            "We have a CCPA deletion request for user {user_id}.",
            "Can you provide evidence of encryption at rest and in transit?",
            "We need to verify your platform meets HIPAA requirements.",
            "Our security team requires your penetration test results.",
            "What certifications does your infrastructure provider hold?",
            "We need to audit all API access logs for the past {months} months.",
        ],
        'onboarding': [
            "We just signed up and need help setting up {feature}.",
            "First day using the platform. How do I configure {setting}?",
            "Can you walk us through the initial data import process?",
            "We're new to the platform and the setup wizard seems broken.",
            "How do we connect our existing {tool} to your platform?",
            "Our team of {count} people needs onboarding training.",
            "The getting started guide doesn't cover {topic}.",
            "We need help migrating data from our previous platform.",
            "How long does the typical onboarding process take?",
            "Can we schedule a kickoff call with your onboarding team?",
        ],
        'general_inquiry': [
            "What are the differences between the {plan1} and {plan2} plans?",
            "How does your usage-based pricing work for API calls?",
            "Is there a free trial available for the Enterprise plan?",
            "Where can I find the API documentation?",
            "What is your typical support response time?",
            "Do you offer discounts for nonprofits or educational institutions?",
            "Can you provide a demo of the {feature} module?",
            "How does your product compare to {competitor}?",
            "Is there a community forum for users?",
            "What's on your product roadmap for {quarter}?",
        ],
        'churn_risk': [
            "We've been customers for {years} years but the quality has declined. Considering alternatives.",
            "This is the {ordinal} time reporting this issue. Still not fixed. Looking at competitors.",
            "Very frustrated with the constant {problem}. This is unacceptable.",
            "We're moving to {competitor} at the end of this month. Please export our data.",
            "The recent price increase is making us reconsider our subscription.",
            "If this isn't resolved by {deadline}, we'll need to cancel.",
            "Our team is fed up with the lack of {feature}. Evaluating other options.",
            "I want to cancel our subscription effective immediately.",
            "The service has not met our expectations. Please process our cancellation.",
            "We're downgrading because the premium features aren't worth the cost.",
        ],
    }

    tickets = []
    for category, tmpls in templates.items():
        for i in range(50):  # 50 tickets per category = 400 total
            tmpl = random.choice(tmpls)
            # Fill in template placeholders with realistic values
            text = tmpl.format(
                amount="$" + str(random.randint(99, 999)),
                plan_amount="$" + str(random.randint(49, 499)),
                feature=random.choice(["analytics", "export", "import", "webhook", "SSO", "dashboard", "reporting"]),
                error_code=random.choice(["500", "502", "503", "429", "403", "401"]),
                days=random.randint(2, 14),
                percent=random.choice(["50", "75", "90"]),
                size=random.choice(["500", "1000", "5000"]),
                section=random.choice(["reports", "analytics", "settings"]),
                device=random.choice(["iPad", "iPhone", "Android tablet"]),
                count=random.randint(5, 200),
                email=f"user{random.randint(1,99)}@company.com",
                format=random.choice(["PDF", "Excel", "XML", "Parquet"]),
                tool=random.choice(["Salesforce", "Slack", "Jira", "HubSpot", "Zapier"]),
                method=random.choice(["hardware keys", "TOTP", "SMS"]),
                operation=random.choice(["user import", "data export", "ticket update"]),
                entity=random.choice(["ticket", "user", "organization"]),
                action=random.choice(["report generation", "data backup", "cleanup"]),
                user_id=f"USR-{random.randint(10000, 99999)}",
                months=random.choice(["3", "6", "12"]),
                setting=random.choice(["API keys", "notifications", "permissions", "integrations"]),
                topic=random.choice(["API setup", "SSO config", "data import", "team management"]),
                plan1=random.choice(["Starter", "Pro"]),
                plan2=random.choice(["Pro", "Enterprise"]),
                competitor=random.choice(["Zendesk", "Freshdesk", "ServiceNow"]),
                quarter=random.choice(["Q1 2026", "Q2 2026", "Q3 2026"]),
                years=random.randint(1, 5),
                ordinal=random.choice(["third", "fourth", "fifth"]),
                problem=random.choice(["downtime", "bugs", "slow performance", "data loss"]),
                deadline=random.choice(["Friday", "end of month", "next week"]),
                code=random.choice(["SAVE20", "LAUNCH50", "PARTNER15"]),
            )
            tickets.append({
                'text': text,
                'label': CATEGORY_MAP[category],
                'category': category,
            })
    random.shuffle(tickets)
    return tickets


def split_data(tickets, train_ratio=0.7, val_ratio=0.15):
    """Split into train/val/test sets."""
    random.shuffle(tickets)
    n = len(tickets)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    return tickets[:train_end], tickets[train_end:val_end], tickets[val_end:]


def save_csv(tickets, filepath):
    """Save tickets to CSV."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['text', 'label', 'category'])
        writer.writeheader()
        writer.writerows(tickets)
    print(f"  ✅ Saved {len(tickets)} rows → {filepath}")


def generate_sla_data():
    """Generate SLA breach training data."""
    import numpy as np
    np.random.seed(42)
    n = 15000
    rows = []
    for i in range(n):
        tc = round(np.random.uniform(3.0, 18.0), 2)
        qd = int(np.random.poisson(12))
        ct = int(np.random.choice([1, 2, 3, 4], p=[0.15, 0.30, 0.35, 0.20]))
        hr = int(np.random.randint(0, 24))
        dw = int(np.random.randint(0, 7))
        ah = round(float(np.random.exponential(4.0).clip(0.5, 48.0)), 2)
        se = round(np.random.uniform(-1.0, 1.0), 3)
        ri = int(np.random.binomial(1, 0.25))
        eb = int(np.random.binomial(1, 0.15))
        logit = (0.08*tc + 0.12*qd - 0.3*(ct-2.5) + 0.05*abs(hr-12)
                 + 0.15*(1 if dw >= 5 else 0) + 0.10*ah - 0.5*se
                 + 0.8*ri + 0.6*eb + np.random.normal(0, 1.0))
        breach = 1 if (1.0 / (1.0 + np.exp(-logit + 2.0))) > 0.5 else 0
        rows.append({
            'text_complexity_score': tc, 'agent_queue_depth': qd,
            'customer_tier': ct, 'hour_of_day': hr, 'day_of_week': dw,
            'similar_ticket_avg_hrs': ah, 'sentiment_score': se,
            'repeat_issue': ri, 'escalated_before': eb, 'sla_breached': breach,
        })
    filepath = os.path.join(PROC_DIR, 'sla_train.csv')
    with open(filepath, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    breach_count = sum(r['sla_breached'] for r in rows)
    print(f"  ✅ SLA data: {n} rows ({breach_count} breaches, {n - breach_count} OK) → {filepath}")


def main():
    print("=" * 60)
    print("SupportMind — Data Preprocessing Pipeline")
    print("=" * 60)

    # Load and process tickets
    print("\n📂 Loading raw ticket data...")
    tickets = load_raw_tickets()
    print(f"   Loaded {len(tickets)} tickets across {len(set(t['category'] for t in tickets))} categories")

    # Category distribution
    print("\n📊 Category distribution:")
    for cat in CATEGORIES:
        count = sum(1 for t in tickets if t['category'] == cat)
        print(f"   {cat:25s} {count:4d} tickets")

    # Split
    print("\n✂️  Splitting data (70/15/15)...")
    train, val, test = split_data(tickets)

    save_csv(train, os.path.join(PROC_DIR, 'train.csv'))
    save_csv(val, os.path.join(PROC_DIR, 'val.csv'))
    save_csv(test, os.path.join(PROC_DIR, 'test.csv'))

    # SLA data
    print("\n🚨 Generating SLA breach training data...")
    try:
        generate_sla_data()
    except ImportError:
        print("  ⚠️  numpy not installed — skipping SLA data generation")

    print("\n" + "=" * 60)
    print("✅ Preprocessing complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
