"""Training-task generators + verifiers.

IMPORTANT (contamination control): these training instances are deliberately DISTINCT
from the indeval eval tasks. They share the same underlying Indian *rules* (so the model
learns the rule) but use different prompts, signatures, and freshly-randomised test
vectors. The eval set stays held out and never appears here.
"""
import random
import string

_CP = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"

def _gstin_check_digit(first14):
    factor, total, n = 2, 0, 36
    for ch in reversed(first14):
        a = factor * _CP.index(ch); factor = 1 if factor == 2 else 2
        total += a // n + a % n
    return _CP[(n - total % n) % n]

def _random_pan(rng):
    L = lambda k: "".join(rng.choice(string.ascii_uppercase) for _ in range(k))
    D = lambda k: "".join(rng.choice(string.digits) for _ in range(k))
    return L(5) + D(4) + L(1)

def random_valid_gstin(rng):
    state = f"{rng.randint(1,37):02d}"
    body = state + _random_pan(rng) + rng.choice(_CP[1:]) + "Z"
    return body + _gstin_check_digit(body)

# Verhoeff (Aadhaar)
_d=[[0,1,2,3,4,5,6,7,8,9],[1,2,3,4,0,6,7,8,9,5],[2,3,4,0,1,7,8,9,5,6],
[3,4,0,1,2,8,9,5,6,7],[4,0,1,2,3,9,5,6,7,8],[5,9,8,7,6,0,4,3,2,1],
[6,5,9,8,7,1,0,4,3,2],[7,6,5,9,8,2,1,0,4,3],[8,7,6,5,9,3,2,1,0,4],
[9,8,7,6,5,4,3,2,1,0]]
_p=[[0,1,2,3,4,5,6,7,8,9],[1,5,7,6,2,8,3,0,9,4],[5,8,0,3,7,9,6,1,4,2],
[8,9,1,6,0,4,3,5,2,7],[9,4,5,3,1,2,6,8,7,0],[4,2,8,6,5,7,3,9,0,1],
[2,7,9,3,8,0,6,4,1,5],[7,0,4,6,9,1,3,2,5,8]]
_inv=[0,4,3,2,1,5,6,7,8,9]

def random_valid_aadhaar(rng):
    base = str(rng.randint(2,9)) + "".join(rng.choice(string.digits) for _ in range(10))
    c = 0
    for i, ch in enumerate(reversed(base)):
        c = _d[c][_p[(i+1) % 8][int(ch)]]
    return base + str(_inv[c])

# ---------------------------------------------------------------------------
# A generator produces (prompt, entry_point, test_program) per domain.
# The test_program (used by indeval.grader) bakes in FRESH random vectors,
# so every generated training item is verified against the real rule.
# ---------------------------------------------------------------------------
_FOOTER = '''
import json as _json
_r={"passed":0,"total":0,"failures":[]}
for _n,_c in _CHECKS:
    _r["total"]+=1
    try: _c(); _r["passed"]+=1
    except Exception as _e: _r["failures"].append(f"{_n}: {_e}")
print("__RESULT__"+_json.dumps(_r))
'''

def gen_upi(rng):
    vpa = f"merchant{rng.randint(1,999)}@ok{rng.choice(['hdfcbank','axis','sbi'])}"
    name = rng.choice(["Surat Chai Co", "Patel Electronics", "Anand Sweets House"])
    amt = round(rng.uniform(1, 9999), 2)
    ref = "ORD" + "".join(rng.choice(string.digits) for _ in range(6))
    mc = str(rng.choice([5411, 5812, 5732]))
    prompt = (f"Write `build_link(vpa, name, amount, ref, mcc)` returning an NPCI-compliant "
              f"UPI *merchant* (P2M) deep link. Mandatory: scheme upi://pay?, payee address pa, "
              f"payee name pn, amount am as a 2-decimal string, currency cu=INR, and for P2M the "
              f"transaction ref tr and merchant code mc. URL-encode all values.")
    tp = f'''
from urllib.parse import urlparse, parse_qs
u = build_link({vpa!r}, {name!r}, {amt!r}, {ref!r}, {mc!r})
p = urlparse(u); q = parse_qs(p.query)
def t_scheme(): assert p.scheme=="upi"
def t_inr(): assert q.get("cu")==["INR"], q.get("cu")
def t_am(): assert q.get("am")==[f"{amt:.2f}"], q.get("am")
def t_tr(): assert q.get("tr")==[{ref!r}]
def t_mc(): assert q.get("mc")==[{mc!r}]
def t_enc(): assert " " not in u
_CHECKS=[("scheme",t_scheme),("cu INR",t_inr),("am 2dp",t_am),("tr",t_tr),("mc",t_mc),("encoded",t_enc)]
''' + _FOOTER
    return prompt, "build_link", tp

def gen_gstin(rng):
    good = [random_valid_gstin(rng) for _ in range(3)]
    bad = [g[:14] + ("X" if g[14] != "X" else "Y") for g in good]   # wrong check digit
    prompt = ("Write `check_gstin(s)` returning True/False for a structurally valid Indian "
              "GSTIN. It must verify the 15th-character Luhn-mod-36 CHECK DIGIT over the first "
              "14 characters, not merely the regex shape.")
    tp = f'''
good={good!r}; bad={bad!r}
def t_good():
    for g in good: assert check_gstin(g) is True, g
def t_bad():
    for b in bad: assert check_gstin(b) is False, ("accepted wrong check digit "+b)
def t_len(): assert check_gstin(good[0][:14]) is False
_CHECKS=[("accepts valid",t_good),("rejects wrong checkdigit",t_bad),("rejects short",t_len)]
''' + _FOOTER
    return prompt, "check_gstin", tp

def gen_aadhaar(rng):
    good = [random_valid_aadhaar(rng) for _ in range(3)]
    bad = [g[:-1] + str((int(g[-1]) + 1) % 10) for g in good]
    prompt = ("Write `check_aadhaar(s)` returning True/False for a valid 12-digit Aadhaar: "
              "12 digits, first digit not 0 or 1, and the 12th digit a valid VERHOEFF checksum "
              "over the first 11 digits. Use only synthetic inputs.")
    tp = f'''
good={good!r}; bad={bad!r}
def t_good():
    for g in good: assert check_aadhaar(g) is True, g
def t_bad():
    for b in bad: assert check_aadhaar(b) is False, ("accepted bad verhoeff "+b)
def t_len(): assert check_aadhaar(good[0][:11]) is False
_CHECKS=[("accepts valid",t_good),("rejects bad checksum",t_bad),("rejects short",t_len)]
''' + _FOOTER
    return prompt, "check_aadhaar", tp

def gen_ifsc(rng):
    L = lambda k: "".join(rng.choice(string.ascii_uppercase) for _ in range(k))
    AN = lambda k: "".join(rng.choice(string.ascii_uppercase + string.digits) for _ in range(k))
    good = [L(4) + "0" + AN(6) for _ in range(3)]
    prompt = ("Write `is_valid_ifsc(s)` returning True/False for a valid Indian IFSC: exactly 11 "
              "chars, first 4 uppercase letters (bank), the 5th character ALWAYS the reserved '0', "
              "last 6 alphanumeric. Enforce the reserved 0, not just the shape.")
    tp = f'''
good={good!r}
def t_good():
    for g in good: assert is_valid_ifsc(g) is True, g
def t_fifth(): assert is_valid_ifsc(good[0][:4]+"1"+good[0][5:]) is False, "5th must be 0"
def t_len(): assert is_valid_ifsc(good[0][:10]) is False
def t_lower(): assert is_valid_ifsc(good[0].lower()) is False
_CHECKS=[("accepts valid",t_good),("5th is 0",t_fifth),("rejects short",t_len),("rejects lowercase",t_lower)]
''' + _FOOTER
    return prompt, "is_valid_ifsc", tp

def gen_pan(rng):
    L = lambda k: "".join(rng.choice(string.ascii_uppercase) for _ in range(k))
    D = lambda k: "".join(rng.choice(string.digits) for _ in range(k))
    types = "PCHABGJLFT"
    good = [L(3) + rng.choice(types) + L(1) + D(4) + L(1) for _ in range(3)]
    badc = [g[:3] + (set("XYZ") - set(g[3])).pop() + g[4:] for g in good]  # invalid 4th char
    prompt = ("Write `is_valid_pan(s)` returning True/False for a valid Indian PAN: 5 uppercase "
              "letters, 4 digits, 1 uppercase letter; the 4th character must be a valid holder-type "
              "code (one of P,C,H,A,B,G,J,L,F,T). Check the 4th char, not just the shape.")
    tp = f'''
good={good!r}; badc={badc!r}
def t_good():
    for g in good: assert is_valid_pan(g) is True, g
def t_type():
    for b in badc: assert is_valid_pan(b) is False, ("accepted bad holder type "+b)
def t_len(): assert is_valid_pan(good[0][:9]) is False
_CHECKS=[("accepts valid",t_good),("rejects bad holder type",t_type),("rejects short",t_len)]
''' + _FOOTER
    return prompt, "is_valid_pan", tp

def gen_mobile(rng):
    base = [str(rng.choice([6, 7, 8, 9])) + "".join(rng.choice(string.digits) for _ in range(9)) for _ in range(3)]
    prompt = ("Write `is_valid_mobile(s)` returning True/False for a valid Indian mobile number. "
              "Strip a leading '+91', '0091', or single leading '0' first; the remaining must be exactly "
              "10 digits starting with 6, 7, 8, or 9. Normalize the prefix AND enforce the start digit.")
    tp = f'''
base={base!r}
def t_plain():
    for s in base: assert is_valid_mobile(s) is True, s
def t_prefixed():
    assert is_valid_mobile("+91"+base[0]) is True
    assert is_valid_mobile("0"+base[0]) is True
def t_start(): assert is_valid_mobile("5"+base[0][1:]) is False, "must start 6-9"
def t_len(): assert is_valid_mobile(base[0][:5]) is False
_CHECKS=[("accepts valid",t_plain),("normalizes prefix",t_prefixed),("start digit 6-9",t_start),("rejects short",t_len)]
''' + _FOOTER
    return prompt, "is_valid_mobile", tp

def gen_inr(rng):
    import re as _re
    def grp(n):
        s = str(n)
        if len(s) <= 3:
            return s
        h, r = s[:-3], s[-3:]
        h = _re.sub(r"(\d)(?=(\d\d)+$)", r"\1,", h)
        return h + "," + r
    vals = [rng.randint(1000, 9_99_99_999) for _ in range(4)]
    expect = {v: grp(v) for v in vals}
    prompt = ("Write `format_inr(n)` formatting a non-negative integer with the Indian lakh-crore "
              "grouping: the last 3 digits, then groups of 2 (e.g. 1234567 -> '12,34,567', "
              "100000 -> '1,00,000'). NOT the Western thousands grouping.")
    tp = f'''
expect={expect!r}
def t_grp():
    for n,e in expect.items(): assert format_inr(n)==e, (str(n)+" -> "+str(format_inr(n))+" want "+e)
def t_small(): assert format_inr(500)=="500"
_CHECKS=[("lakh-crore grouping",t_grp),("small unchanged",t_small)]
''' + _FOOTER
    return prompt, "format_inr", tp

def gen_fy(rng):
    cases = []
    for _ in range(4):
        y = rng.randint(2018, 2025); m = rng.randint(1, 12); d = rng.randint(1, 28)
        fy = y if m >= 4 else y - 1
        cases.append((f"{y:04d}-{m:02d}-{d:02d}", f"{fy}-{str(fy+1)[-2:]}"))
    prompt = ("Write `indian_fy(date_str)` taking an ISO date 'YYYY-MM-DD' and returning the Indian "
              "financial year as 'YYYY-YY' (the FY runs April 1 to March 31; e.g. 2023-05-10 -> "
              "'2023-24', 2023-02-10 -> '2022-23').")
    tp = f'''
cases={cases!r}
def t_fy():
    for ds,e in cases: assert indian_fy(ds)==e, (ds+" -> "+str(indian_fy(ds))+" want "+e)
_CHECKS=[("apr-mar financial year",t_fy)]
''' + _FOOTER
    return prompt, "indian_fy", tp

GENERATORS = {"upi": gen_upi, "gst": gen_gstin, "aadhaar": gen_aadhaar,
              "ifsc": gen_ifsc, "pan": gen_pan, "mobile": gen_mobile,
              "inr": gen_inr, "fy": gen_fy}
