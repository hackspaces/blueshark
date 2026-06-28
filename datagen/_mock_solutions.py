"""Canned teacher solutions for the offline (mock) data-engine demo.

For each domain: a "good" solution that actually enforces the India-specific rule
(passes the indeval grader) and a "naive" one that only matches the surface format
(fails the rule check). The mock teacher returns these so `build_dataset.py
--teacher mock` proves the verify-filter bites: good -> 100% kept, naive -> 0%.
"""

_GST_GOOD = '''
def check_gstin(s):
    cp = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    if not isinstance(s, str) or len(s) != 15 or any(c not in cp for c in s):
        return False
    factor, total, n = 2, 0, 36
    for ch in reversed(s[:14]):
        a = factor * cp.index(ch); factor = 1 if factor == 2 else 2
        total += a // n + a % n
    return cp[(n - total % n) % n] == s[14]
'''

_GST_NAIVE = '''
import re
def check_gstin(s):
    return bool(re.fullmatch(r"[0-9A-Z]{15}", s or ""))   # shape only, ignores check digit
'''

_AADHAAR_GOOD = '''
def check_aadhaar(s):
    if not isinstance(s, str) or len(s) != 12 or not s.isdigit() or s[0] in "01":
        return False
    d = [[0,1,2,3,4,5,6,7,8,9],[1,2,3,4,0,6,7,8,9,5],[2,3,4,0,1,7,8,9,5,6],
         [3,4,0,1,2,8,9,5,6,7],[4,0,1,2,3,9,5,6,7,8],[5,9,8,7,6,0,4,3,2,1],
         [6,5,9,8,7,1,0,4,3,2],[7,6,5,9,8,2,1,0,4,3],[8,7,6,5,9,3,2,1,0,4],
         [9,8,7,6,5,4,3,2,1,0]]
    p = [[0,1,2,3,4,5,6,7,8,9],[1,5,7,6,2,8,3,0,9,4],[5,8,0,3,7,9,6,1,4,2],
         [8,9,1,6,0,4,3,5,2,7],[9,4,5,3,1,2,6,8,7,0],[4,2,8,6,5,7,3,9,0,1],
         [2,7,9,3,8,0,6,4,1,5],[7,0,4,6,9,1,3,2,5,8]]
    c = 0
    for i, ch in enumerate(reversed(s)):
        c = d[c][p[i % 8][int(ch)]]
    return c == 0
'''

_AADHAAR_NAIVE = '''
def check_aadhaar(s):
    return isinstance(s, str) and len(s) == 12 and s.isdigit() and s[0] not in "01"  # no checksum
'''

_UPI_GOOD = '''
from urllib.parse import quote
def build_link(vpa, name, amount, ref, mcc):
    params = {"pa": vpa, "pn": name, "am": f"{float(amount):.2f}",
              "cu": "INR", "tr": ref, "mc": mcc}
    return "upi://pay?" + "&".join(f"{k}={quote(str(v), safe='')}" for k, v in params.items())
'''

_UPI_NAIVE = '''
def build_link(vpa, name, amount, ref, mcc):
    return f"upi://pay?pa={vpa}&pn={name}&am={amount}&cu=INR"   # no tr/mc, unencoded, am not 2dp
'''

SOLUTIONS = {
    "gst":     {"good": _GST_GOOD,     "naive": _GST_NAIVE},
    "aadhaar": {"good": _AADHAAR_GOOD, "naive": _AADHAAR_NAIVE},
    "upi":     {"good": _UPI_GOOD,     "naive": _UPI_NAIVE},
}
