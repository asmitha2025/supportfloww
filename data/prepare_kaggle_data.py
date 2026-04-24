# data/prepare_kaggle_data.py
# Combines Kaggle Customer Support Tickets with Synthetic Categories
import os
import csv
import random

random.seed(42)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
KAGGLE_CSV = os.path.join(ROOT_DIR, '..', 'customer_support_tickets.csv')
PROC_DIR = os.path.join(BASE_DIR, 'processed')

CATEGORIES = [
    'billing', 'technical_support', 'account_management', 'feature_request',
    'compliance_legal', 'onboarding', 'general_inquiry', 'churn_risk'
]
CATEGORY_MAP = {cat: i for i, cat in enumerate(CATEGORIES)}

# Mapping from Kaggle 'Ticket Type' to our categories
KAGGLE_MAPPING = {
    'Billing inquiry': 'billing',
    'Refund request': 'billing',
    'Technical issue': 'technical_support',
    'Cancellation request': 'churn_risk',
    'Product inquiry': 'general_inquiry'
}

def load_kaggle_data():
    tickets = []
    if not os.path.exists(KAGGLE_CSV):
        print(f"Kaggle CSV not found at {KAGGLE_CSV}")
        return tickets
        
    with open(KAGGLE_CSV, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            ttype = row.get('Ticket Type', '')
            desc = row.get('Ticket Description', '').strip()
            
            if not desc or len(desc) < 10:
                continue
                
            if ttype in KAGGLE_MAPPING:
                cat = KAGGLE_MAPPING[ttype]
                tickets.append({
                    'text': desc,
                    'label': CATEGORY_MAP[cat],
                    'category': cat,
                    'source': 'kaggle'
                })
    return tickets

def get_synthetic_data_for_missing():
    # Import from the existing preprocess.py to generate the missing classes
    import sys
    sys.path.append(BASE_DIR)
    try:
        from preprocess import generate_synthetic_data
        synth_all = generate_synthetic_data()
        
        # Only keep classes that Kaggle dataset doesn't cover well
        missing_cats = ['account_management', 'feature_request', 'compliance_legal', 'onboarding']
        filtered = [t for t in synth_all if t['category'] in missing_cats]
        for f in filtered:
            f['source'] = 'synthetic'
        return filtered
    except ImportError:
        print("Could not import preprocess.py")
        return []

def extract_sla_kaggle_data():
    """Extract rows for SLA predictor training from Kaggle dataset."""
    sla_data = []
    if not os.path.exists(KAGGLE_CSV):
        return sla_data
        
    with open(KAGGLE_CSV, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                # Text complexity score (heuristic: length / 20)
                desc = row.get('Ticket Description', '')
                tc = min(len(desc) / 20.0, 18.0)
                if tc < 3.0: tc = 3.0
                
                # Extract priorities and satisfaction
                priority = row.get('Ticket Priority', 'Medium')
                qd = 10
                if priority == 'Critical': qd = 20
                elif priority == 'High': qd = 15
                elif priority == 'Low': qd = 5
                
                # Tier based on age or arbitrary for demo
                age = int(row.get('Customer Age', 30))
                ct = 4 if age > 50 else (3 if age > 35 else (2 if age > 25 else 1))
                
                # Sentiment (if low satisfaction, negative sentiment)
                sat_rating = row.get('Customer Satisfaction Rating', '')
                if sat_rating:
                    sat = float(sat_rating)
                    se = (sat - 3.0) / 2.0  # Maps 1-5 to -1.0 to 1.0
                else:
                    se = 0.0
                    
                # Breach heuristic (if priority is Critical or High, higher chance)
                status = row.get('Ticket Status', '')
                breach = 1 if priority in ['Critical', 'High'] and status == 'Pending Customer Response' else 0
                
                sla_data.append({
                    'text_complexity_score': round(tc, 2),
                    'agent_queue_depth': qd,
                    'customer_tier': ct,
                    'hour_of_day': random.randint(8, 18),
                    'day_of_week': random.randint(0, 4),
                    'similar_ticket_avg_hrs': round(random.uniform(1.0, 24.0), 2),
                    'sentiment_score': round(se, 2),
                    'repeat_issue': random.randint(0, 1),
                    'escalated_before': 1 if priority == 'Critical' else 0,
                    'sla_breached': breach
                })
            except ValueError:
                continue
    return sla_data

def main():
    os.makedirs(PROC_DIR, exist_ok=True)
    
    print("Loading Kaggle data...")
    kaggle_tickets = load_kaggle_data()
    print(f"Loaded {len(kaggle_tickets)} valid Kaggle tickets.")
    
    print("Loading Synthetic data for missing categories...")
    synth_tickets = get_synthetic_data_for_missing()
    print(f"Loaded {len(synth_tickets)} synthetic tickets.")
    
    all_tickets = kaggle_tickets + synth_tickets
    random.shuffle(all_tickets)
    
    # Subsample to keep training fast but effective (~5000 tickets)
    if len(all_tickets) > 5000:
        all_tickets = all_tickets[:5000]
        print(f"Subsampled down to {len(all_tickets)} tickets for efficient training.")
        
    print("\nCategory distribution for training:")
    counts = {c: 0 for c in CATEGORIES}
    for t in all_tickets:
        counts[t['category']] += 1
    for k, v in counts.items():
        print(f"  {k:20s}: {v}")
        
    # Split 80/20
    n = len(all_tickets)
    train_end = int(n * 0.8)
    train = all_tickets[:train_end]
    val = all_tickets[train_end:]
    
    def save_csv(data, filename):
        path = os.path.join(PROC_DIR, filename)
        with open(path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['text', 'label', 'category', 'source'])
            writer.writeheader()
            writer.writerows(data)
        print(f"Saved {len(data)} rows to {filename}")
        
    save_csv(train, 'train.csv')
    save_csv(val, 'val.csv')
    
    # Process SLA Data
    print("\nProcessing Kaggle SLA data...")
    sla_data = extract_sla_kaggle_data()
    if len(sla_data) > 10000:
        sla_data = random.sample(sla_data, 10000)
    
    if sla_data:
        sla_path = os.path.join(PROC_DIR, 'sla_train.csv')
        with open(sla_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=list(sla_data[0].keys()))
            writer.writeheader()
            writer.writerows(sla_data)
        print(f"Saved {len(sla_data)} SLA rows to sla_train.csv")

if __name__ == "__main__":
    main()
