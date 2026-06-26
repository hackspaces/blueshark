"""The Indian-context task suite. Each task is execution-graded: the model's
submitted function is run against hidden tests that encode the *actual Indian rule*
(UPI spec, GSTIN mod-36 checksum, Aadhaar Verhoeff checksum), so a plausible-looking
format-only answer fails where it counts."""
from indeval.schema import Task

# ---------------------------------------------------------------------------
# Shared test-program footer: runs a list of (name, thunk) checks, emits JSON.
# ---------------------------------------------------------------------------
_FOOTER = '''
import json as _json
_results = {"passed": 0, "total": 0, "failures": []}
for _name, _check in _CHECKS:
    _results["total"] += 1
    try:
        _check()
        _results["passed"] += 1
    except Exception as _e:
        _results["failures"].append(f"{_name}: {type(_e).__name__}: {_e}")
print("__RESULT__" + _json.dumps(_results))
'''

# ===========================================================================
# TASK 1 — UPI merchant (P2M) deep-link builder
# ===========================================================================
UPI = Task(
    task_id="upi_p2m_link",
    domain="upi",
    title="Build an NPCI-compliant UPI merchant payment deep link",
    entry_point="build_upi_link",
    prompt=(
        "Write a Python function `build_upi_link(payee_vpa, payee_name, amount, txn_ref, "
        "merchant_code)` that returns an NPCI-compliant UPI *merchant* (P2M) payment deep link.\n\n"
        "Requirements (per the NPCI UPI Linking Specification):\n"
        "- The link must use the scheme/host `upi://pay?`.\n"
        "- Include payee address `pa`, payee name `pn`, amount `am`, currency `cu`.\n"
        "- `am` MUST be a decimal string with exactly two decimal places (e.g. 10.00).\n"
        "- `cu` MUST be `INR` (the only currency UPI supports).\n"
        "- Because this is a *merchant* (P2M) transaction, you MUST include a unique transaction "
        "reference `tr` and the merchant code `mc`.\n"
        "- All query values must be correctly URL-encoded (spaces in `pn` must not break the URL).\n\n"
        "Return only the URL string. `amount` is a float (rupees); `merchant_code` and `txn_ref` are strings."
    ),
    reference_solution='''
from urllib.parse import urlencode, quote
def build_upi_link(payee_vpa, payee_name, amount, txn_ref, merchant_code):
    params = {
        "pa": payee_vpa,
        "pn": payee_name,
        "am": f"{float(amount):.2f}",
        "cu": "INR",
        "tr": txn_ref,
        "mc": merchant_code,
    }
    return "upi://pay?" + urlencode(params, quote_via=quote)
''',
    # Plausible but wrong: forgets cu=INR, no 2-decimal formatting, raw f-string concat
    # (breaks on spaces), omits the P2M-mandatory tr/mc handling rigor.
    naive_solution='''
def build_upi_link(payee_vpa, payee_name, amount, txn_ref, merchant_code):
    return f"upi://pay?pa={payee_vpa}&pn={payee_name}&am={amount}"
''',
    test_program='''
from urllib.parse import urlparse, parse_qs
def _parsed():
    u = build_upi_link("merchant@okhdfcbank", "Chai Point Store", 10, "ORD12345", "5411")
    p = urlparse(u)
    return u, p, parse_qs(p.query)
def t_scheme():
    u, p, q = _parsed(); assert p.scheme == "upi" and (p.netloc == "pay" or p.path.startswith("pay")), u
def t_pa():
    _,_,q = _parsed(); assert q.get("pa") == ["merchant@okhdfcbank"], q.get("pa")
def t_pn():
    _,_,q = _parsed(); assert q.get("pn") == ["Chai Point Store"], q.get("pn")
def t_amount_two_decimals():
    _,_,q = _parsed(); assert q.get("am") == ["10.00"], ("am must be 2dp decimal, got " + str(q.get("am")))
def t_currency_inr():
    _,_,q = _parsed(); assert q.get("cu") == ["INR"], ("cu must be INR, got " + str(q.get("cu")))
def t_merchant_ref():
    _,_,q = _parsed(); assert q.get("tr") == ["ORD12345"], "P2M requires tr"
def t_merchant_code():
    _,_,q = _parsed(); assert q.get("mc") == ["5411"], "P2M requires mc"
def t_space_encoded():
    u,_,_ = _parsed(); assert " " not in u, "spaces in pn must be URL-encoded, URL contains a raw space"
_CHECKS = [
    ("scheme is upi://pay", t_scheme),
    ("payee address pa", t_pa),
    ("payee name pn", t_pn),
    ("amount formatted 2dp", t_amount_two_decimals),
    ("currency = INR", t_currency_inr),
    ("merchant tr present", t_merchant_ref),
    ("merchant mc present", t_merchant_code),
    ("spaces URL-encoded", t_space_encoded),
]
''' + _FOOTER,
)

# ===========================================================================
# TASK 2 — GSTIN validator (Luhn mod-36 check digit)
# ===========================================================================
GST = Task(
    task_id="gstin_validate",
    domain="gst",
    title="Validate a GSTIN including the mod-36 check digit",
    entry_point="is_valid_gstin",
    prompt=(
        "Write a Python function `is_valid_gstin(gstin)` returning True/False for whether a string "
        "is a structurally valid Indian GSTIN.\n\n"
        "A GSTIN is 15 characters: 2-digit state code (01-37), then a 10-char PAN "
        "(5 letters, 4 digits, 1 letter), then a 1-char entity number (1-9 or A-Z), then the "
        "literal 'Z', then a check digit.\n\n"
        "Crucially, the 15th character is NOT free-form: it is a checksum over the first 14 "
        "characters using the Luhn mod-36 algorithm (digits map 0-9, letters A-Z map 10-35; "
        "alternate weights of 1 and 2 from the right; sum the base-36 digits of each product; "
        "the check char makes the total a multiple of 36). A correct validator must verify this "
        "check digit, not merely the shape."
    ),
    reference_solution='''
import re
_CP = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
def _check_digit(first14):
    factor, total, n = 2, 0, 36
    for ch in reversed(first14):
        a = factor * _CP.index(ch)
        factor = 1 if factor == 2 else 2
        total += a // n + a % n
    return _CP[(n - total % n) % n]
def is_valid_gstin(gstin):
    if not isinstance(gstin, str) or len(gstin) != 15:
        return False
    if not re.fullmatch(r"[0-3][0-9][A-Z]{5}[0-9]{4}[A-Z][0-9A-Z]Z[0-9A-Z]", gstin):
        return False
    if not (1 <= int(gstin[:2]) <= 37):
        return False
    return _check_digit(gstin[:14]) == gstin[14]
''',
    # Plausible but wrong: validates the SHAPE with a regex but never checks the mod-36 digit.
    naive_solution='''
import re
def is_valid_gstin(gstin):
    return bool(re.fullmatch(r"[0-9]{2}[A-Z]{5}[0-9]{4}[A-Z][0-9A-Z]Z[0-9A-Z]", gstin or ""))
''',
    test_program='''
def t_valid_real():
    for g in ["27AAPFU0939F1ZV", "09AAACH7409R1ZZ", "07AAACG0527D1Z8"]:
        assert is_valid_gstin(g) is True, ("should be valid: " + g)
def t_bad_checkdigit():
    # same string, last char wrong -> a checksum-aware validator MUST reject
    assert is_valid_gstin("27AAPFU0939F1ZX") is False, "accepted a GSTIN with a wrong check digit"
def t_bad_checkdigit2():
    assert is_valid_gstin("09AAACH7409R1ZA") is False, "accepted a GSTIN with a wrong check digit"
def t_wrong_length():
    assert is_valid_gstin("27AAPFU0939F1Z") is False
def t_bad_state():
    assert is_valid_gstin("99AAPFU0939F1ZV") is False, "accepted invalid state code 99"
def t_bad_pan_shape():
    assert is_valid_gstin("271APFU0939F1ZV") is False
_CHECKS = [
    ("accepts real GSTINs", t_valid_real),
    ("rejects wrong check digit", t_bad_checkdigit),
    ("rejects wrong check digit (2)", t_bad_checkdigit2),
    ("rejects wrong length", t_wrong_length),
    ("rejects bad state code", t_bad_state),
    ("rejects malformed PAN", t_bad_pan_shape),
]
''' + _FOOTER,
)

# ===========================================================================
# TASK 3 — Aadhaar validator (Verhoeff check digit)
# ===========================================================================
AADHAAR = Task(
    task_id="aadhaar_validate",
    domain="aadhaar",
    title="Validate an Aadhaar number using the Verhoeff checksum",
    entry_point="is_valid_aadhaar",
    prompt=(
        "Write a Python function `is_valid_aadhaar(num)` returning True/False for whether a 12-digit "
        "string is a structurally valid Aadhaar number.\n\n"
        "Rules: exactly 12 digits; the first digit cannot be 0 or 1 (UIDAI rule); and the 12th digit "
        "is a checksum over the first 11 digits computed with the Verhoeff algorithm. A correct "
        "validator must implement the Verhoeff check, not just check that the input is 12 digits.\n\n"
        "(Use only synthetic inputs; do not hardcode any real person's Aadhaar.)"
    ),
    reference_solution='''
_d=[[0,1,2,3,4,5,6,7,8,9],[1,2,3,4,0,6,7,8,9,5],[2,3,4,0,1,7,8,9,5,6],
[3,4,0,1,2,8,9,5,6,7],[4,0,1,2,3,9,5,6,7,8],[5,9,8,7,6,0,4,3,2,1],
[6,5,9,8,7,1,0,4,3,2],[7,6,5,9,8,2,1,0,4,3],[8,7,6,5,9,3,2,1,0,4],
[9,8,7,6,5,4,3,2,1,0]]
_p=[[0,1,2,3,4,5,6,7,8,9],[1,5,7,6,2,8,3,0,9,4],[5,8,0,3,7,9,6,1,4,2],
[8,9,1,6,0,4,3,5,2,7],[9,4,5,3,1,2,6,8,7,0],[4,2,8,6,5,7,3,9,0,1],
[2,7,9,3,8,0,6,4,1,5],[7,0,4,6,9,1,3,2,5,8]]
def is_valid_aadhaar(num):
    if not isinstance(num, str) or len(num) != 12 or not num.isdigit():
        return False
    if num[0] in "01":
        return False
    c = 0
    for i, ch in enumerate(reversed(num)):
        c = _d[c][_p[i % 8][int(ch)]]
    return c == 0
''',
    # Plausible but wrong: checks 12 digits + first-digit rule but skips Verhoeff entirely.
    naive_solution='''
def is_valid_aadhaar(num):
    return isinstance(num, str) and len(num) == 12 and num.isdigit() and num[0] not in "01"
''',
    test_program='''
# synthetic helper to MINT valid test vectors (not real Aadhaar numbers)
_d=[[0,1,2,3,4,5,6,7,8,9],[1,2,3,4,0,6,7,8,9,5],[2,3,4,0,1,7,8,9,5,6],
[3,4,0,1,2,8,9,5,6,7],[4,0,1,2,3,9,5,6,7,8],[5,9,8,7,6,0,4,3,2,1],
[6,5,9,8,7,1,0,4,3,2],[7,6,5,9,8,2,1,0,4,3],[8,7,6,5,9,3,2,1,0,4],
[9,8,7,6,5,4,3,2,1,0]]
_p=[[0,1,2,3,4,5,6,7,8,9],[1,5,7,6,2,8,3,0,9,4],[5,8,0,3,7,9,6,1,4,2],
[8,9,1,6,0,4,3,5,2,7],[9,4,5,3,1,2,6,8,7,0],[4,2,8,6,5,7,3,9,0,1],
[2,7,9,3,8,0,6,4,1,5],[7,0,4,6,9,1,3,2,5,8]]
_inv=[0,4,3,2,1,5,6,7,8,9]
def _mint(base11):
    c = 0
    for i, ch in enumerate(reversed(base11)):
        c = _d[c][_p[(i+1) % 8][int(ch)]]
    return base11 + str(_inv[c])
_GOOD = _mint("23412341234")   # synthetic, first digit 2
_GOOD2 = _mint("98765432109")  # synthetic, first digit 9
def t_valid_synthetic():
    assert is_valid_aadhaar(_GOOD) is True, ("should accept valid Verhoeff number " + _GOOD)
    assert is_valid_aadhaar(_GOOD2) is True
def t_bad_checksum():
    bad = _GOOD[:-1] + str((int(_GOOD[-1]) + 1) % 10)
    assert is_valid_aadhaar(bad) is False, "accepted a number that fails the Verhoeff checksum"
def t_first_digit_rule():
    assert is_valid_aadhaar(_mint("03412341234")) is False, "first digit 0 must be rejected"
def t_length():
    assert is_valid_aadhaar("23412341234") is False  # 11 digits
def t_nondigit():
    assert is_valid_aadhaar("2341234123X4") is False
_CHECKS = [
    ("accepts valid Verhoeff numbers", t_valid_synthetic),
    ("rejects wrong checksum", t_bad_checksum),
    ("enforces first-digit rule", t_first_digit_rule),
    ("rejects wrong length", t_length),
    ("rejects non-digits", t_nondigit),
]
''' + _FOOTER,
)

# ===========================================================================
# TASK 4 — IFSC code validator (5th character reserved '0')
# ===========================================================================
IFSC = Task(
    task_id="ifsc_validate",
    domain="ifsc",
    title="Validate an Indian bank IFSC code",
    entry_point="is_valid_ifsc",
    prompt=(
        "Write a Python function `is_valid_ifsc(ifsc)` returning True/False for whether a string "
        "is a valid Indian IFSC (Indian Financial System Code).\n\n"
        "An IFSC is exactly 11 characters: the first 4 are uppercase letters (the bank code), the "
        "5th character is ALWAYS '0' (reserved by RBI for future use), and the last 6 characters are "
        "alphanumeric (the branch code). A correct validator must enforce the reserved '0', not just "
        "the overall shape."
    ),
    reference_solution='''
import re
def is_valid_ifsc(ifsc):
    if not isinstance(ifsc, str) or len(ifsc) != 11:
        return False
    return bool(re.fullmatch(r"[A-Z]{4}0[A-Z0-9]{6}", ifsc))
''',
    # Plausible but wrong: allows any character in the 5th slot.
    naive_solution='''
import re
def is_valid_ifsc(ifsc):
    return bool(re.fullmatch(r"[A-Z]{4}[A-Z0-9]{7}", ifsc or ""))
''',
    test_program='''
def t_valid():
    for s in ["HDFC0001234", "SBIN0000456", "ICIC0000123"]:
        assert is_valid_ifsc(s) is True, ("should be valid: " + s)
def t_fifth_is_zero():
    assert is_valid_ifsc("HDFC1001234") is False, "5th char must be the reserved 0"
def t_length():
    assert is_valid_ifsc("HDFC000123") is False
def t_lowercase():
    assert is_valid_ifsc("hdfc0001234") is False
def t_bank_letters():
    assert is_valid_ifsc("HD1C0001234") is False
_CHECKS = [
    ("accepts valid IFSC", t_valid),
    ("5th char must be 0", t_fifth_is_zero),
    ("rejects wrong length", t_length),
    ("rejects lowercase", t_lowercase),
    ("bank code must be letters", t_bank_letters),
]
''' + _FOOTER,
)

# ===========================================================================
# TASK 5 — PAN validator (4th character is the holder-type code)
# ===========================================================================
PAN = Task(
    task_id="pan_validate",
    domain="pan",
    title="Validate a PAN including the holder-type character",
    entry_point="is_valid_pan",
    prompt=(
        "Write a Python function `is_valid_pan(pan)` returning True/False for whether a string is a "
        "valid Indian PAN (Permanent Account Number).\n\n"
        "A PAN is 10 characters: 5 uppercase letters, then 4 digits, then 1 uppercase letter. "
        "Crucially, the 4th character encodes the type of holder and must be one of "
        "P, C, H, A, B, G, J, L, F, T (e.g. P for individual, C for company, F for firm, T for trust). "
        "A correct validator must check that 4th character, not merely the letter/digit shape."
    ),
    reference_solution='''
import re
_PAN_TYPES = set("PCHABGJLFT")
def is_valid_pan(pan):
    if not isinstance(pan, str) or len(pan) != 10:
        return False
    if not re.fullmatch(r"[A-Z]{5}[0-9]{4}[A-Z]", pan):
        return False
    return pan[3] in _PAN_TYPES
''',
    # Plausible but wrong: checks the shape but not the holder-type character.
    naive_solution='''
import re
def is_valid_pan(pan):
    return bool(re.fullmatch(r"[A-Z]{5}[0-9]{4}[A-Z]", pan or ""))
''',
    test_program='''
def t_valid():
    for s in ["ABCPK1234L", "XYZCA9876B", "AAAFA1234H"]:
        assert is_valid_pan(s) is True, ("should be valid: " + s)
def t_holder_type():
    assert is_valid_pan("ABCXK1234L") is False, "4th char X is not a valid holder type"
def t_length():
    assert is_valid_pan("ABCPK1234") is False
def t_shape():
    assert is_valid_pan("ABC1K1234L") is False
def t_lowercase():
    assert is_valid_pan("abcpk1234l") is False
_CHECKS = [
    ("accepts valid PAN", t_valid),
    ("rejects invalid holder type", t_holder_type),
    ("rejects wrong length", t_length),
    ("rejects malformed shape", t_shape),
    ("rejects lowercase", t_lowercase),
]
''' + _FOOTER,
)

# ===========================================================================
# TASK 6 — Indian mobile number validator (prefix normalization + start digit)
# ===========================================================================
MOBILE = Task(
    task_id="mobile_validate",
    domain="mobile",
    title="Validate an Indian mobile number after normalizing prefixes",
    entry_point="is_valid_mobile",
    prompt=(
        "Write a Python function `is_valid_mobile(num)` returning True/False for whether a string is a "
        "valid Indian mobile number.\n\n"
        "Rules: the number may carry a country/trunk prefix you must strip first: a leading '+91', "
        "'0091', or a single leading '0'. After stripping, what remains must be exactly 10 digits, and "
        "the first of those 10 digits must be 6, 7, 8, or 9 (Indian mobile series). A correct validator "
        "must normalize the prefix AND enforce the start digit, not just check for 10 digits."
    ),
    reference_solution='''
def is_valid_mobile(num):
    if not isinstance(num, str):
        return False
    s = num.strip().replace(" ", "").replace("-", "")
    if s.startswith("+91"):
        s = s[3:]
    elif s.startswith("0091"):
        s = s[4:]
    elif s.startswith("0") and len(s) == 11:
        s = s[1:]
    if not (s.isdigit() and len(s) == 10):
        return False
    return s[0] in "6789"
''',
    # Plausible but wrong: just checks ten digits, no prefix handling, no start-digit rule.
    naive_solution='''
import re
def is_valid_mobile(num):
    return bool(re.fullmatch(r"[0-9]{10}", num or ""))
''',
    test_program='''
def t_valid_plain():
    for s in ["9876543210", "6012345678", "7000000000"]:
        assert is_valid_mobile(s) is True, ("should be valid: " + s)
def t_valid_prefixed():
    assert is_valid_mobile("+919876543210") is True, "must accept and strip +91"
    assert is_valid_mobile("09876543210") is True, "must accept a leading trunk 0"
def t_start_digit():
    assert is_valid_mobile("5876543210") is False, "Indian mobile numbers start 6-9"
def t_length():
    assert is_valid_mobile("98765") is False
def t_nondigit():
    assert is_valid_mobile("98765abcde") is False
_CHECKS = [
    ("accepts valid 10-digit", t_valid_plain),
    ("normalizes +91 / 0 prefix", t_valid_prefixed),
    ("enforces start digit 6-9", t_start_digit),
    ("rejects wrong length", t_length),
    ("rejects non-digits", t_nondigit),
]
''' + _FOOTER,
)

TASKS = [UPI, GST, AADHAAR, IFSC, PAN, MOBILE]

# ---------------------------------------------------------------------------
# Additional tasks (authored and verified in a batch) live in tasks_batch.json
# so the code strings escape cleanly. Their test_program omits the shared footer;
# we append _FOOTER here, exactly as the inline tasks above do.
# ---------------------------------------------------------------------------
import json as _json
import os as _os

_BATCH = _os.path.join(_os.path.dirname(__file__), "tasks_batch.json")
if _os.path.exists(_BATCH):
    with open(_BATCH, encoding="utf-8") as _f:
        for _t in _json.load(_f):
            TASKS.append(Task(
                task_id=_t["task_id"],
                domain=_t["domain"],
                title=_t["title"],
                entry_point=_t["entry_point"],
                prompt=_t["prompt"],
                reference_solution=_t["reference_solution"],
                naive_solution=_t["naive_solution"],
                test_program=_t["test_program"] + _FOOTER,
            ))
