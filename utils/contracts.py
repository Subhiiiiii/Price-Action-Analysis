# market_analysis/utils/contracts.py
"""
Contract generation: front-N spreads and jumbos for any product.
Also converts instrument strings to/from API format.
"""

import re
import datetime
from typing import Optional

from ..config import PRODUCTS, STRUCTURES, HORIZON_BARS

# CME month codes
MONTH_CODES: dict[int, str] = {
    1:"F", 2:"G", 3:"H", 4:"J", 5:"K", 6:"M",
    7:"N", 8:"Q", 9:"U", 10:"V", 11:"X", 12:"Z",
}


def parse_contract(s: str) -> tuple[str, str, str]:
    """'SMZ26' -> ('SM', 'Z', '26')"""
    m = re.match(r"^([A-Za-z]+?)([A-Z])(\d+)$", s)
    if not m:
        raise ValueError(f"Cannot parse contract: {s!r}")
    return m.group(1), m.group(2), m.group(3)


def fmt_contract(product: str, year: int, month: int) -> str:
    """('SM', 2026, 12) -> 'SMZ26'"""
    return f"{product}{MONTH_CODES[month]}{str(year)[2:]}"


def instrument_to_api_str(instrument: str) -> str:
    """
    Convert an instrument string to API short form.

    New format (already short):  'SMZ26-H27'   -> 'SMZ26-H27'  (no-op)
    Legacy long format:          'SMZ26-SMH27' -> 'SMZ26-H27'
    """
    front, back = instrument.split("-", 1)
    try:
        _, mc, yr = parse_contract(back)
        return f"{front}-{mc}{yr}"
    except ValueError:
        # back is already in short form (e.g. 'H27') -- return as-is
        return instrument


def api_str_to_instrument(api_str: str, product: str) -> str:
    """
    Convert API short-form back leg to the long internal form (legacy).

    'SMZ26-F27' -> 'SMZ26-SMF27'

    Note: get_front_instruments() now emits the short form directly,
    so this helper is only needed for legacy stored strings.
    """
    front, back_short = api_str.split("-", 1)
    return f"{front}-{product}{back_short}"


def next_n_contract_months(
    traded_months: list[int],
    reference_date: datetime.date,
    n: int,
) -> list[tuple[int, int]]:
    """Return next n (year, month) tuples strictly after reference_date."""
    y, m    = reference_date.year, reference_date.month
    results = []
    while len(results) < n:
        m += 1
        if m > 12:
            m = 1; y += 1
        if m in traded_months:
            results.append((y, m))
    return results


def _fmt_back_leg(year: int, month: int) -> str:
    """Format the back leg of a spread/jumbo: month_code + 2-digit year only."""
    return f"{MONTH_CODES[month]}{str(year)[2:]}"


def get_front_instruments(
    product_code: str,
    structure_name: str,
    reference_date: Optional[datetime.date] = None,
    front_n: int = 5,
) -> list[str]:
    """
    Return front_n instrument strings for a product + structure.

    The back leg is always in short form (month_code + 2-digit year, no prefix).

    For spread (back_offsets=[1]):
      get_front_instruments("SM", "spread") ->
        ["SMV26-Z26", "SMZ26-F27", ...]

    For jumbo (back_offsets=[2, 3]):
      get_front_instruments("SM", "jumbo") ->
        ["SMV26-F27", "SMV26-H27", "SMZ26-H27", "SMZ26-K27", "SMF27-K27"]

      i.e. for each front month i, generate pairs i→i+2 and i→i+3 in order,
      then take the first front_n across all i.
    """
    if reference_date is None:
        reference_date = datetime.date.today()

    product      = PRODUCTS[product_code]
    front_off, back_offsets = STRUCTURES[structure_name]   # (0, [1]) or (0, [2,3])
    back_offsets = sorted(back_offsets)
    max_bo       = max(back_offsets)

    # Fetch enough legs so that the last pair we might need is valid.
    # Worst case: i iterates up to front_n-1 (for a single-offset structure),
    # but for multi-offset we stop as soon as we collect front_n pairs, so
    # front_n + max_bo is always a safe upper bound.
    max_legs = front_n + max_bo + front_off + 1

    legs   = next_n_contract_months(
        product["traded_months"], reference_date, n=max_legs
    )
    prefix = product["api_prefix"]

    pairs: list[tuple[int, int]] = []
    for i in range(len(legs)):
        for bo in back_offsets:
            bi = front_off + i + bo
            if bi >= len(legs):
                continue
            pairs.append((front_off + i, bi))
            if len(pairs) == front_n:
                break
        if len(pairs) == front_n:
            break

    # Front leg: full product prefix (e.g. "SMV26").
    # Back leg:  short form — month_code + 2-digit year only (e.g. "F27").
    # This matches the API format and avoids the spurious double-prefix
    # (e.g. "SMZ26-SMH27" → correct form is "SMZ26-H27").
    return [
        f"{fmt_contract(prefix, *legs[fi])}-{_fmt_back_leg(*legs[bi])}"
        for fi, bi in pairs
    ]



def get_all_front_instruments(
    product_code: str,
    reference_date: Optional[datetime.date] = None,
    front_n: int = 5,
) -> dict[str, list[str]]:
    """
    Return {structure_name: [instrument_strings]} for all structures.

    Back leg is always in short form (no product prefix).

    get_all_front_instruments("SM") ->
    {
      "spread": ["SMV26-Z26", ...],
      "jumbo" : ["SMV26-F27", "SMV26-H27", "SMZ26-H27", "SMZ26-K27", "SMF27-K27"],
    }
    """
    return {
        struct: get_front_instruments(product_code, struct, reference_date, front_n)
        for struct in STRUCTURES
    }