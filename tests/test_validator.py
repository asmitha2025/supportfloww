# tests/test_validator.py
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from ticket_validator import TicketValidator

validator = TicketValidator()

def test_empty_string():
    result = validator.validate("")
    assert not result['valid']
    assert result['error_type'] == 'empty'

def test_whitespace_only():
    result = validator.validate("   ")
    assert not result['valid']
    assert result['error_type'] == 'empty'

def test_greeting_hi():
    result = validator.validate("hi")
    assert not result['valid']
    assert result['error_type'] == 'greeting'

def test_greeting_hello():
    result = validator.validate("hello!")
    assert not result['valid']
    assert result['error_type'] == 'greeting'

def test_too_short():
    result = validator.validate("help me")
    assert not result['valid']
    assert result['error_type'] in ('too_short', 'greeting')

def test_valid_ticket():
    result = validator.validate(
        "My invoice from last month shows wrong amount please help"
    )
    assert result['valid']
    assert result['error_type'] is None
    assert result['cleaned_text'] is not None

def test_too_long():
    long_text = "The API is broken " * 300
    result = validator.validate(long_text)
    assert result['valid']  # Still valid — just truncated
    assert result['warning'] is True
    assert len(result['cleaned_text'].split()) <= 500

def test_tamil_text():
    result = validator.validate("எனது கணக்கில் சிக்கல் உள்ளது")
    assert not result['valid']
    assert result['error_type'] == 'non_english'

def test_hindi_text():
    result = validator.validate("मेरे खाते में समस्या है")
    assert not result['valid']
    assert result['error_type'] == 'non_english'

def test_gibberish():
    result = validator.validate("asdfghjkl qwerty zxcvbnm poiuytrewq lkjhgfd")
    assert not result['valid']
    assert result['error_type'] == 'gibberish'

def test_resolved():
    result = validator.validate("nevermind problem is solved thanks")
    assert not result['valid']
    assert result['error_type'] == 'resolved'

def test_numbers_only():
    result = validator.validate("500 404 200 301")
    assert not result['valid']
    assert result['error_type'] == 'no_text'

def test_whitespace_cleaned():
    result = validator.validate(
        "  My   invoice   is   wrong  "
    )
    assert result['valid']
    assert "  " not in result['cleaned_text']

if __name__ == '__main__':
    tests = [
        test_empty_string,
        test_whitespace_only,
        test_greeting_hi,
        test_greeting_hello,
        test_too_short,
        test_valid_ticket,
        test_too_long,
        test_tamil_text,
        test_hindi_text,
        test_gibberish,
        test_resolved,
        test_numbers_only,
        test_whitespace_cleaned,
    ]
    
    print("=" * 50)
    print("TICKET VALIDATOR TESTS")
    print("=" * 50)
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            print(f"[PASS] {test.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"[FAIL] {test.__name__} — {e}")
            failed += 1
    
    print(f"\n{passed} passed, {failed} failed")
    if failed == 0:
        print("All tests passed! [OK]")

