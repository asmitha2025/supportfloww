# data/generate_ood_test.py
# Generates a genuinely out-of-distribution (OOD) test set for SupportMind.
#
# WHY THIS EXISTS:
# The standard val/test split in preprocess.py is drawn from the same
# template distribution as training data. Evaluating on it produces inflated
# accuracy (~100%) because the model has seen structurally identical sentences.
#
# This script generates hand-crafted tickets that deliberately avoid all
# training templates — different vocabulary, informal phrasing, multi-sentence
# context, typos, ambiguous cases, and real-world edge cases.
# Evaluating on this set yields honest, lower out-of-distribution accuracy.
#
# Usage:
#   python data/generate_ood_test.py
# Output:
#   data/processed/ood_test.csv  (labeled OOD test set)
#
# SupportMind — Asmitha

import os
import csv
import random

random.seed(99)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROC_DIR = os.path.join(BASE_DIR, 'processed')

# ── Hand-crafted OOD tickets per category ─────────────────────────────────────
# Rules for every sentence written here:
#   1. Must NOT use any phrase from preprocess.py templates
#   2. Should reflect how real B2B users actually write tickets
#   3. Mix of: informal, multi-sentence, typos, jargon, abbreviations
#   4. Some tickets are deliberately ambiguous (hard cases)

OOD_TICKETS = {

    'billing': [
        # Informal, real-world phrasing
        "hey our last receipt had a weird extra line item under 'platform fee' - never seen that before, what is it",
        "we got charged twice last tuesday, both transactions visible in our bank statement",
        "finance team says the amount debited doesn't match the quote we accepted in march",
        "i need VAT removed from our invoices asap, we're registered as VAT-exempt in ireland",
        "our accountant is asking for a credit note for the overpayment in february",
        "the system won't let me download our Q4 invoices, getting a blank page",
        "we switched from annual to monthly plan two weeks ago but still got charged the annual rate",
        "can someone explain why our bill went up 23% this cycle with no plan changes",
        "the autopay failed last night and now the account shows past due, but card is valid",
        "we need invoices re-issued with our new entity name after the company rebrand",
        "just noticed we've been on the wrong tier for 3 months paying for enterprise when we're on growth",
        "our CFO needs a statement of account going back 18 months for the audit",
    ],

    'technical_support': [
        # Technical but conversational, real errors
        "reports tab just shows a spinner forever, been like this since yesterday morning for all our users",
        "getting a 'session expired' popup every 20 minutes even though we haven't changed any timeout settings",
        "the CSV download for our user list is coming back completely empty, no headers even",
        "two factor auth codes stopped arriving via email, tried multiple accounts same problem",
        "our zapier integration triggers fired but nothing updated on your end, checked the zap logs they show 200 ok",
        "whenever someone tries to @ mention a teammate in a comment it crashes the whole tab",
        "the search bar stopped returning results for anything older than 30 days overnight",
        "push notifications on android are delayed by like 4-5 hours, ios users are fine",
        "we have a data discrepancy: dashboard shows 4,201 active users but the API returns 3,988",
        "after last night's deployment our SSO stopped working, redirects to a blank page",
        "rate limiting hitting us at 180 req/min when our tier should allow 500",
        "bulk import keeps failing at exactly row 847 regardless of what data we put in the file",
    ],

    'account_management': [
        # Org admin requests in real language
        "our CTO just left the company and he was the only owner on the account, how do we take control",
        "is there a way to see a log of who accessed what data over the last 90 days",
        "we need to split our account into two separate orgs for our EU and US business units",
        "can we set it so only admin-level users can export data, regular agents shouldn't have this",
        "our company was acquired and we need to migrate everything to the parent company's account",
        "we have 3 employees who left - need their sessions terminated and access revoked right now",
        "how do I stop certain team members from seeing the billing section specifically",
        "we need guest access for external consultants that expires after 30 days automatically",
        "our SAML cert just rotated, how do we update it without locking everyone out",
        "can you add a second domain to our account, we have a subsidiary that uses a different email domain",
    ],

    'feature_request': [
        # Product feedback, wishlist items
        "would really love a way to pin certain tickets to the top of the queue for my whole team",
        "any plans to add a kanban view option, the list view doesn't work well for our workflow",
        "we need the ability to create custom SLA policies per customer tier not just one global setting",
        "it would be incredibly useful if the API returned the agent assignment history not just current",
        "please add a way to bulk-archive tickets older than X days, doing it one by one is painful",
        "is there a public roadmap we can subscribe to for updates",
        "we'd love conditional logic in your forms so irrelevant fields hide based on category",
        "can you add keyboard shortcuts to your UI, our agents live in this tool all day",
        "we need the reports to be schedulable so they auto-send to stakeholders every monday",
        "any chance of a read-only link sharing option for specific reports without requiring a login",
    ],

    'compliance_legal': [
        # Legal/security/compliance, formal tone
        "our legal team requires a subprocessor list before we can sign the DPA",
        "can you confirm whether your infrastructure is hosted entirely within the EU for GDPR purposes",
        "we've received a court order requesting all communications from user account X, how do we proceed",
        "need to know your breach notification timeline and who gets notified under your incident response plan",
        "our ISO 27001 auditor is asking for your risk register or equivalent documentation",
        "is there a way to ensure certain data never leaves a specific geographic region",
        "we need to perform a right-to-erasure under GDPR for a former customer, what's the process",
        "our security team flagged that API tokens don't expire - is there a way to enforce rotation",
        "can you provide a copy of your most recent vulnerability assessment report",
        "we need a BAA signed before we can process any data through your platform",
    ],

    'onboarding': [
        # New customer confusion, getting-started language
        "just got access this morning, where do i even start - the dashboard is overwhelming",
        "we were told to set up a webhook during onboarding but the setup page doesn't match the docs",
        "our success manager said we'd get a kickoff call scheduled but we haven't heard anything after signing",
        "migrating from zendesk, is there an automated way to move our ticket history over",
        "how long does it usually take before we can go live? we have a deadline in 3 weeks",
        "the setup checklist says 'configure routing rules' but I can't find where that setting lives",
        "we have 80 agents who need accounts, is there a bulk invite feature or do we do it one by one",
        "our team uses desk phones for support too, does your system integrate with any VOIP providers",
        "the trial data we imported doesn't show up in production after upgrading, where did it go",
        "we bought the enterprise tier but some features listed on the pricing page aren't showing up yet",
    ],

    'general_inquiry': [
        # Pre-sales, casual questions, no clear urgency
        "quick question - if we go over our monthly ticket limit do you auto-charge or just cap it",
        "do you have a status page where we can check if there are ongoing incidents",
        "we're evaluating you alongside helpscout and intercom, what makes you different",
        "our team is fully remote across 6 timezones, does your support cover that or only business hours",
        "is there a sandbox/staging environment we can use to test things without affecting production",
        "roughly how much data storage does a team of 50 agents typically use per year",
        "we're a startup of 8 people, is the starter plan enough or will we hit limits fast",
        "what happens to our data if we decide to cancel, can we export everything",
        "do you have native integrations with linear or notion or do we need zapier for that",
        "are there any uptime SLA guarantees in the contract and what's the compensation if missed",
    ],

    'churn_risk': [
        # Frustration, implicit/explicit cancellation signals
        "honestly at this point we've had 6 open tickets for over a month with no resolution",
        "i've been a customer for 3 years and the service has gotten noticeably worse since your redesign",
        "our team is actively demoing competitors right now because of the reliability issues",
        "sent 4 follow up emails to our account manager with no response, extremely unprofessional",
        "the product has gone downhill since the acquisition and the pricing just went up again",
        "we're going to have a hard conversation internally about renewing if this isn't fixed this week",
        "i need our full data export immediately - zip file of everything - we've made a decision",
        "your SLA says 4 hour response but we've been waiting 3 days on a critical issue",
        "leadership is losing patience and so am i, this is the last time i'm asking nicely",
        "please confirm our cancellation date is end of month as I requested last thursday",
        "we switched half our workflow to a competitor last week as a backup, tell me why we shouldn't move fully",
        "the contract is up for renewal in 6 weeks and right now i would not recommend renewing",
    ],
}

# ── Ambiguous edge cases (deliberately hard, labeled with best-fit category) ──
AMBIGUOUS_TICKETS = [
    # billing vs account_management
    ("we need to add a new entity to our account that should be billed separately", "billing"),
    ("our subsidiary needs its own contract and invoice, how does that work", "billing"),
    # technical_support vs feature_request
    ("the analytics charts don't show data older than 6 months, is this a limit or a bug", "technical_support"),
    ("can we get more granular filters on the reports page, the current ones aren't enough", "feature_request"),
    # onboarding vs technical_support
    ("we just started and the API key we got doesn't work in production, returns 401", "technical_support"),
    ("new account setup, the webhook we configured isn't firing at all during testing", "onboarding"),
    # churn_risk vs general_inquiry
    ("what would it cost to move down to the basic tier, weighing our options", "general_inquiry"),
    ("thinking about whether the enterprise features are worth what we're paying", "churn_risk"),
    # compliance_legal vs account_management
    ("we need to know exactly which of your staff can see our customer data", "compliance_legal"),
    ("can we restrict which of our admins have access to audit logs", "account_management"),
]


def build_ood_dataset():
    """Assemble and shuffle the full OOD test dataset."""
    from preprocess import CATEGORY_MAP  # reuse the label mapping

    tickets = []

    # Main hand-crafted tickets
    for category, texts in OOD_TICKETS.items():
        label = CATEGORY_MAP[category]
        for text in texts:
            tickets.append({'text': text.strip(), 'label': label, 'category': category, 'ood_type': 'hand_crafted'})

    # Ambiguous tickets
    for text, category in AMBIGUOUS_TICKETS:
        label = CATEGORY_MAP[category]
        tickets.append({'text': text.strip(), 'label': label, 'category': category, 'ood_type': 'ambiguous'})

    random.shuffle(tickets)
    return tickets


def save_ood_csv(tickets, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    fieldnames = ['text', 'label', 'category', 'ood_type']
    with open(filepath, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(tickets)
    print(f"  Saved {len(tickets)} OOD test samples -> {filepath}")


def main():
    print("=" * 60)
    print("SupportMind — OOD Test Set Generator")
    print("=" * 60)
    print("\nThis dataset is deliberately out-of-distribution:")
    print("  - Hand-crafted sentences (no template reuse from training)")
    print("  - Informal phrasing, typos, multi-sentence context")
    print("  - Includes ambiguous edge-case tickets")
    print()

    tickets = build_ood_dataset()

    # Stats
    from collections import Counter
    cat_counts = Counter(t['category'] for t in tickets)
    type_counts = Counter(t['ood_type'] for t in tickets)

    print("Category distribution:")
    for cat, count in sorted(cat_counts.items()):
        print(f"  {cat:25s}  {count:3d} samples")

    print(f"\nOOD types:")
    for ood_type, count in type_counts.items():
        print(f"  {ood_type:20s}  {count:3d} samples")

    out_path = os.path.join(PROC_DIR, 'ood_test.csv')
    save_ood_csv(tickets, out_path)

    print(f"\nTotal: {len(tickets)} OOD test samples")
    print("\nNext step: run  python src/evaluate_ood.py  to get honest metrics")
    print("=" * 60)


if __name__ == '__main__':
    main()
