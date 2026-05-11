
import sys
import os

# Aggressive memory and backend management
os.environ['USE_TF'] = '0'
os.environ['USE_JAX'] = '0'
os.environ['USE_TORCH'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

sys.path.insert(0, os.path.join(os.getcwd(), 'src'))

from ensemble_router import EnsembleRouter

def test_billing_misrouting():
    router = EnsembleRouter()
    
    # This ticket should ideally be billing, but if it's misrouted to technical_support,
    # we want to see if our safety check catches it.
    ticket = "I have a question about my last invoice, the charge seems too high."
    
    result = router.route(ticket)
    
    print(f"Ticket: {ticket}")
    print(f"Top Category: {result['top_category']}")
    print(f"Action: {result['action']}")
    print(f"Reason: {result['reason']}")
    print(f"Ranking: {result['category_ranking'][:3]}")

if __name__ == "__main__":
    test_billing_misrouting()
