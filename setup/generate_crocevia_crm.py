import os
import random
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from faker import Faker


RANDOM_SEED = 42
NUM_ROWS = 1_000_000
ORIG_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "summit_sports_crm.parquet"))
OUT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "crocevia_crm.parquet"))


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def _is_email(val: str) -> bool:
    if not isinstance(val, str):
        return False
    if "@" not in val or val.count("@") != 1:
        return False
    if any(ch.isspace() for ch in val):
        return False
    local, _, domain = val.partition("@")
    return bool(local) and "." in domain


def _is_fr_phone_like(val: str) -> bool:
    if not isinstance(val, str):
        return False
    s = val.replace(" ", "").replace("-", "").replace(".", "")
    if s.startswith("+33"):
        s = s[3:]
    if s.startswith("(0)"):
        s = s[3:]
    if s.startswith("0"):
        s = s[1:]
    return s.isdigit() and 8 <= len(s) <= 10


def _is_name_like(val: str) -> bool:
    if not isinstance(val, str):
        return False
    if len(val) == 0:
        return False
    if any(ch.isdigit() for ch in val):
        return False
    tokens = val.replace("-", " ").split()
    if len(tokens) > 2:
        return False
    # Capitalization heuristic
    return all(tok[0].isalpha() and tok[0].upper() == tok[0] for tok in tokens)


def _is_fr_postal(val: str) -> bool:
    if not isinstance(val, str):
        return False
    s = val.strip()
    return len(s) == 5 and s.isdigit()


def infer_columns(df: pd.DataFrame) -> Dict[str, Optional[str]]:
    sample = df.head(5000)
    scores: Dict[str, Dict[str, float]] = {col: {} for col in df.columns}
    for col in df.columns:
        ser = sample[col].astype(str)
        values = ser.values
        n = max(1, len(values))
        email_score = float(np.mean([_is_email(v) for v in values]))
        phone_score = float(np.mean([_is_fr_phone_like(v) for v in values]))
        name_score = float(np.mean([_is_name_like(v) for v in values]))
        postal_score = float(np.mean([_is_fr_postal(v) for v in values]))
        uniq_ratio = ser.nunique(dropna=True) / max(1, len(ser))
        avg_len = ser.str.len().mean()
        scores[col] = {
            "email": email_score,
            "phone": phone_score,
            "name": name_score,
            "postal": postal_score,
            "uniq": float(uniq_ratio),
            "avg_len": float(avg_len if pd.notnull(avg_len) else 0.0),
        }

    # Pick columns
    email_col = max(df.columns, key=lambda c: scores[c]["email"]) if df.shape[1] > 0 else None
    if scores[email_col]["email"] < 0.4:
        email_col = None

    phone_col = max(df.columns, key=lambda c: scores[c]["phone"]) if df.shape[1] > 0 else None
    if scores[phone_col]["phone"] < 0.4:
        phone_col = None

    # Names: pick top two distinct by name score
    name_sorted = sorted(df.columns, key=lambda c: scores[c]["name"], reverse=True)
    first_name_col = None
    last_name_col = None
    if name_sorted:
        top_names = [c for c in name_sorted if scores[c]["name"] >= 0.4][:2]
        if len(top_names) >= 2:
            # shorter avg_len as first name
            a, b = top_names[0], top_names[1]
            first_name_col, last_name_col = (a, b) if scores[a]["avg_len"] <= scores[b]["avg_len"] else (b, a)
        elif len(top_names) == 1:
            # Only one clearly name-like; set as first name only
            first_name_col = top_names[0]

    # Postal code
    postal_col = max(df.columns, key=lambda c: scores[c]["postal"]) if df.shape[1] > 0 else None
    if scores[postal_col]["postal"] < 0.4:
        postal_col = None

    # Customer id: high uniqueness, alphanumeric
    def id_score(col: str) -> float:
        ser = sample[col].astype(str)
        uniq = scores[col]["uniq"]
        alnum = float(np.mean([s.isalnum() for s in ser.values]))
        return uniq * 0.8 + alnum * 0.2

    customer_id_col = max(df.columns, key=id_score) if df.shape[1] > 0 else None
    if scores[customer_id_col]["uniq"] < 0.5:
        customer_id_col = None

    # Preferred store: string-like, low cardinality
    def store_score(col: str) -> float:
        ser = sample[col].astype(str)
        nunique = ser.nunique(dropna=True)
        if nunique <= 1:
            return 0.0
        # Prefer columns that are not email/phone/names and have low unique count
        if col in {email_col, phone_col, first_name_col, last_name_col, customer_id_col}:
            return 0.0
        return 1.0 / nunique

    preferred_store_col = max(df.columns, key=store_score) if df.shape[1] > 0 else None
    # If best candidate still too high cardinality, drop
    if preferred_store_col and sample[preferred_store_col].nunique(dropna=True) > 200:
        preferred_store_col = None

    return {
        "email": email_col,
        "phone": phone_col,
        "first_name": first_name_col,
        "last_name": last_name_col,
        "postal": postal_col,
        "customer_id": customer_id_col,
        "preferred_store": preferred_store_col,
    }


def pick_indices(total: int, target: int, excluded: np.ndarray | None = None) -> np.ndarray:
    if excluded is None:
        excluded = np.array([], dtype=np.int64)
    pool = np.setdiff1d(np.arange(total, dtype=np.int64), excluded, assume_unique=False)
    if target > pool.size:
        raise ValueError(f"Requested {target} indices but only {pool.size} available")
    return np.random.choice(pool, size=target, replace=False)


def main() -> None:
    set_seeds(RANDOM_SEED)
    faker = Faker("fr_FR")
    Faker.seed(RANDOM_SEED)

    if not os.path.exists(ORIG_PATH):
        raise FileNotFoundError(f"Original parquet not found at {ORIG_PATH}")

    df_orig = pd.read_parquet(ORIG_PATH)

    # Infer columns heuristically
    colmap = infer_columns(df_orig)
    orig_columns: List[str] = list(df_orig.columns)
    target_columns: List[str] = list(orig_columns)
    if colmap.get("preferred_store") in target_columns:
        target_columns.remove(colmap["preferred_store"])  # drop preferred store if detected
    # Add new fields
    if "date_of_birth" not in {c.lower() for c in target_columns}:
        target_columns.append("date_of_birth")

    # If there is a postal code-like column, reuse its name; otherwise create one
    postal_candidates = [
        "postal_code",
        "postcode",
        "zip",
        "zip_code",
        "code_postal",
    ]
    postal_code_col = colmap.get("postal")
    if postal_code_col is None:
        postal_code_col = "postal_code"
        if postal_code_col not in target_columns:
            target_columns.append(postal_code_col)

    n = NUM_ROWS

    # Prepare base arrays
    first_names = np.array([faker.first_name() for _ in range(n)], dtype=object)
    last_names = np.array([faker.last_name() for _ in range(n)], dtype=object)
    # Unique customer IDs with a new prefix to ensure no overlap
    def gen_id() -> str:
        return f"CRV-{faker.unique.random_number(digits=10)}"

    customer_ids = np.array([gen_id() for _ in range(n)], dtype=object)

    # Start with fully synthetic emails/phones, then apply overlap rules
    def gen_email(fn: str, ln: str) -> str:
        local = f"{fn}.{ln}".lower().replace("'", "").replace(" ", "")
        domain = random.choice(["gmail.com", "orange.fr", "free.fr", "wanadoo.fr", "sfr.fr", "laposte.net"])
        return f"{local}@{domain}"

    emails = np.array([gen_email(fn, ln) for fn, ln in zip(first_names, last_names)], dtype=object)
    phones = np.array([faker.phone_number() for _ in range(n)], dtype=object)

    # Draw source pools from original
    # Map columns
    email_col = colmap.get("email")
    phone_col = colmap.get("phone")
    first_name_col = colmap.get("first_name")
    last_name_col = colmap.get("last_name")
    customer_id_col = colmap.get("customer_id") or "customer_id"

    # Draw source pools from original for overlaps
    for required, cname in [("email", email_col), ("phone", phone_col), ("first_name", first_name_col), ("last_name", last_name_col)]:
        if cname is None:
            raise ValueError(f"Could not infer {required} column from original dataset; cannot enforce overlap constraints.")
    src_emails = df_orig[email_col].dropna().astype(str).values
    src_phones = df_orig[phone_col].dropna().astype(str).values
    src_first = df_orig[first_name_col].dropna().astype(str).values
    src_last = df_orig[last_name_col].dropna().astype(str).values

    if src_emails.size == 0 or src_phones.size == 0 or src_first.size == 0 or src_last.size == 0:
        raise ValueError("Original dataset lacks sufficient non-null identity fields to enforce overlap constraints.")

    # Define overlap counts (marginals)
    num_match_all = int(0.20 * n)  # 20% all three
    num_match_email_total = int(0.60 * n)  # total email matches including the all-three
    num_match_phone_total = int(0.50 * n)
    num_match_name_total = int(0.35 * n)

    # Sample base triple-match set, then allow overlaps among the extra sets
    idx_all = pick_indices(n, num_match_all)
    remaining_pool = np.setdiff1d(np.arange(n), idx_all)
    # Extras can overlap with each other but not with idx_all to keep exact triple-match count
    idx_email_extra = np.random.choice(remaining_pool, size=max(0, num_match_email_total - num_match_all), replace=False)
    idx_phone_extra = np.random.choice(remaining_pool, size=max(0, num_match_phone_total - num_match_all), replace=False)
    idx_name_extra = np.random.choice(remaining_pool, size=max(0, num_match_name_total - num_match_all), replace=False)

    # Apply matches from original
    # All three from the SAME original record to ensure consistent triple-match
    src_idx_all = np.random.randint(0, df_orig.shape[0], size=idx_all.size)
    emails[idx_all] = df_orig[email_col].astype(str).values[src_idx_all]
    phones[idx_all] = df_orig[phone_col].astype(str).values[src_idx_all]
    first_names[idx_all] = df_orig[first_name_col].astype(str).values[src_idx_all]
    last_names[idx_all] = df_orig[last_name_col].astype(str).values[src_idx_all]

    # Email-only matches (independent draws)
    emails[idx_email_extra] = np.random.choice(src_emails, size=idx_email_extra.size, replace=True)
    # Phone-only matches
    phones[idx_phone_extra] = np.random.choice(src_phones, size=idx_phone_extra.size, replace=True)
    # Name-only matches
    name_pairs = np.column_stack([
        np.random.choice(src_first, size=idx_name_extra.size, replace=True),
        np.random.choice(src_last, size=idx_name_extra.size, replace=True),
    ])
    first_names[idx_name_extra] = name_pairs[:, 0]
    last_names[idx_name_extra] = name_pairs[:, 1]

    # Missingness (avoid disturbing matched indices):
    # Emails missing for 15% outside email-match set
    outside_email = np.setdiff1d(np.arange(n), np.concatenate([idx_all, idx_email_extra]))
    miss_email_idx = np.random.choice(outside_email, size=int(0.15 * n), replace=False)
    emails[miss_email_idx] = None

    # Phones missing for 20% outside phone-match set
    outside_phone = np.setdiff1d(np.arange(n), np.concatenate([idx_all, idx_phone_extra]))
    miss_phone_idx = np.random.choice(outside_phone, size=int(0.20 * n), replace=False)
    phones[miss_phone_idx] = None

    # Duplicate customers: 10% duplicates with minor variations
    dup_fraction = 0.10
    num_dups = int(dup_fraction * n)
    dup_sources = np.random.choice(np.arange(n), size=num_dups, replace=False)
    dup_targets = np.random.choice(np.setdiff1d(np.arange(n), dup_sources), size=num_dups, replace=False)
    # Copy core identity fields to simulate duplicates (different IDs by construction)
    first_names[dup_targets] = first_names[dup_sources]
    last_names[dup_targets] = last_names[dup_sources]
    # For about half, mutate email or phone slightly to simulate inconsistency
    mutate_mask = np.random.rand(num_dups) < 0.5
    for idx_src, idx_tgt, do_mutate in zip(dup_sources, dup_targets, mutate_mask):
        emails[idx_tgt] = emails[idx_src]
        phones[idx_tgt] = phones[idx_src]
        if do_mutate:
            # slight mutation: change domain or add a digit in local part
            if emails[idx_tgt]:
                try:
                    local, domain = str(emails[idx_tgt]).split("@", 1)
                    emails[idx_tgt] = f"{local}{random.randint(0,9)}@{domain}"
                except Exception:
                    pass
            if phones[idx_tgt]:
                phones[idx_tgt] = str(phones[idx_tgt])[:-1] + str(random.randint(0, 9))

    # Date of birth for 40%
    dob = np.array([None] * n, dtype=object)
    has_dob_idx = np.random.choice(np.arange(n), size=int(0.40 * n), replace=False)
    for i in has_dob_idx:
        # Adults 18-90
        year = np.random.randint(1934, 2006)
        month = np.random.randint(1, 13)
        day = np.random.randint(1, 28)
        dob[i] = pd.Timestamp(year=year, month=month, day=day).date()

    # Postal code for 60%
    postal_codes = np.array([None] * n, dtype=object)
    has_postal_idx = np.random.choice(np.arange(n), size=int(0.60 * n), replace=False)
    for i in has_postal_idx:
        postal_codes[i] = faker.postcode()

    # Build output DataFrame with target columns
    out_df = pd.DataFrame(index=np.arange(n))
    for col in target_columns:
        lower = col.lower()
        if col == customer_id_col:
            out_df[col] = customer_ids
        elif col == email_col:
            out_df[col] = emails
        elif col == phone_col:
            out_df[col] = phones
        elif col == first_name_col:
            out_df[col] = first_names
        elif col == last_name_col:
            out_df[col] = last_names
        elif lower == "date_of_birth":
            out_df[col] = dob
        elif lower in {c.lower() for c in [postal_code_col]}:
            out_df[col] = postal_codes
        else:
            # Preserve column, leave as missing; maintain dtype if possible by casting using original sample
            sample = df_orig[col][:1]
            if pd.api.types.is_integer_dtype(df_orig[col].dtype):
                out_df[col] = pd.Series([pd.NA] * n, dtype="Int64")
            elif pd.api.types.is_float_dtype(df_orig[col].dtype):
                out_df[col] = np.nan
            elif pd.api.types.is_bool_dtype(df_orig[col].dtype):
                out_df[col] = pd.Series([pd.NA] * n, dtype="boolean")
            else:
                out_df[col] = pd.Series([None] * n, dtype=object)

    # Write Parquet
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    out_df.to_parquet(OUT_PATH, engine="pyarrow", index=False)

    # Simple validation prints
    email_overlap = np.isin(out_df[email_col].astype(str), df_orig[email_col].astype(str).values, assume_unique=False)
    phone_overlap = np.isin(out_df[phone_col].astype(str), df_orig[phone_col].astype(str).values, assume_unique=False)
    name_overlap = (
        np.isin(out_df[first_name_col].astype(str) + "|" + out_df[last_name_col].astype(str),
               (df_orig[first_name_col].astype(str) + "|" + df_orig[last_name_col].astype(str)).values,
               assume_unique=False)
    )

    print({
        "rows": len(out_df),
        "email_overlap_frac": float(email_overlap.mean()),
        "phone_overlap_frac": float(phone_overlap.mean()),
        "name_overlap_frac": float(name_overlap.mean()),
        "output": OUT_PATH,
    })


if __name__ == "__main__":
    main()


