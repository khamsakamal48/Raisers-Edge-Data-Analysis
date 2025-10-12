# import os
# import logging
# import json
# import sys
# import glob
# from typing import Dict, List, Optional
#
# import pandas as pd
# import requests
# from flatten_json import flatten
# from sqlalchemy import create_engine, text
# from sqlalchemy.exc import SQLAlchemyError
# from sqlalchemy.types import DateTime
# from requests.adapters import HTTPAdapter
# from urllib3 import Retry
# from urllib.parse import quote_plus
# from dotenv import load_dotenv
#
# # ------------------------------- Helper functions -------------------------------
#
# def start_logging():
#     p = os.path.basename(__file__).replace('.py', '').replace(' ', '_')
#     os.makedirs('Logs', exist_ok=True)
#     os.makedirs('Data Dumps', exist_ok=True)
#     logging.basicConfig(filename=f'Logs/{p}.log',
#                         format='%(asctime)s %(levelname)s %(message)s',
#                         filemode='w',
#                         level=logging.DEBUG)
#     logging.info("Starting script")
#     return p
#
# def set_api_request_strategy():
#     retry_strategy = Retry(
#         total=3,
#         status_forcelist=[429, 500, 502, 503, 504],
#         allowed_methods=['HEAD', 'GET', 'OPTIONS', 'GET'],
#         backoff_factor=3
#     )
#     s = requests.Session()
#     adapter = HTTPAdapter(max_retries=retry_strategy)
#     s.mount('https://', adapter)
#     return s
#
# def housekeeping():
#     for f in glob.glob('*_RE_*.json'):
#         try:
#             os.remove(f)
#         except Exception:
#             pass
#
# def retrieve_token():
#     with open('access_token_output.json') as fh:
#         d = json.load(fh)
#     return d.get('access_token')
#
# def get_request_re(url):
#     headers = {
#         'Bb-Api-Subscription-Key': RE_API_KEY,
#         'Authorization': 'Bearer ' + retrieve_token()
#     }
#     return http.get(url, params={}, headers=headers).json()
#
# def pagination_api_request(url):
#     logging.info(f"Paginating {url}")
#     housekeeping()
#     while url:
#         resp = get_request_re(url)
#         # save
#         i = 1
#         while os.path.exists(f'API_Response_RE_{process_name}_{i}.json'):
#             i += 1
#         with open(f'API_Response_RE_{process_name}_{i}.json', 'w') as fh:
#             json.dump(resp, fh, ensure_ascii=False, indent=2)
#         # if next_link exists, continue (Blackbaud returns next_link key)
#         next_link = resp.get('next_link') or resp.get('next')
#         if next_link:
#             url = next_link
#         else:
#             break
#
#
# def load_from_json_to_df():
#     file_list = glob.glob(f'API_Response_RE_{process_name}_*.json')
#     df = pd.DataFrame()
#     for each_file in file_list:
#         with open(each_file, 'r') as fh:
#             content = json.load(fh)
#         vals = content.get('value', [])
#         if not isinstance(vals, list):
#             logging.warning(f"{each_file}: 'value' not list; skipping")
#             continue
#         df_part = pd.DataFrame((flatten(v) for v in vals), columns=None) if vals else pd.DataFrame()
#
#         df = pd.concat([df, df_part], ignore_index=True) if not df_part.empty else df
#
#         # Change id cols to integer
#         id_cols = [x for x in df.columns if (x == 'id' or '_id' in x) and x not in ('lookup_id', 'campaign_id')]
#         for col in id_cols:
#             df[col] = pd.to_numeric(df[col], errors='coerce').astype('Int64')
#             # if col in ('gift_splits_0_campaign_id', 'gift_splits_0_fund_id'):
#             #     df[col] = df[col].astype('Int64')
#             # else:
#             #     df[col] = pd.to_numeric(df[col])
#
#     return df
#
# def detect_date_columns(df: pd.DataFrame) -> List[str]:
#     return [c for c in df.columns if 'date' in c.lower() and 'birth' not in c.lower() and 'deceased' not in c.lower()
#             and not c.lower().endswith('_y') and not c.lower().endswith('_d') and not c.lower().endswith('_m')]
#
# def convert_date_columns(df: pd.DataFrame, date_cols: List[str]) -> pd.DataFrame:
#     for c in date_cols:
#         try:
#             df[c] = pd.to_datetime(df[c], utc=True, errors='coerce')
#         except Exception as e:
#             logging.warning(f"Failed to parse {c}: {e}")
#     return df
#
# def sqlalchemy_dtype_map_for_dates(date_cols: List[str]) -> Dict[str, DateTime]:
#     return {c: DateTime(timezone=True) for c in date_cols}
#
# def connect_db(dbname: str):
#     engine = create_engine(f'postgresql+psycopg2://{DB_USERNAME}:{DB_PASSWORD}@{DB_IP}:5432/{dbname}', pool_recycle=3600)
#     conn = engine.connect()
#     return conn
#
# def drop_table_cascade_if_exists(conn, table_name: str):
#     try:
#         conn.execute(text(f'DROP TABLE IF EXISTS public."{table_name}" CASCADE;'))
#         conn.commit()
#         logging.info(f"Dropped {table_name} (CASCADE) if existed")
#     except SQLAlchemyError as e:
#         logging.exception(f"Failed to DROP TABLE {table_name}: {e}")
#
# def table_has_pk(conn, table_name: str) -> bool:
#     q = text("""
#         SELECT 1
#         FROM information_schema.table_constraints tc
#         WHERE tc.table_schema = 'public'
#           AND tc.table_name = :table
#           AND tc.constraint_type = 'PRIMARY KEY'
#         LIMIT 1;
#     """)
#     try:
#         return conn.execute(q, {"table": table_name}).scalar() is not None
#     except SQLAlchemyError:
#         return False
#
# def add_primary_key(conn, table_name: str, col: str):
#     if not col:
#         return
#     try:
#         if table_has_pk(conn, table_name):
#             logging.info(f"PK exists on {table_name}; skipping creation")
#             return
#         conn.execute(text(f'ALTER TABLE public."{table_name}" ADD PRIMARY KEY ("{col}");'))
#         conn.commit()
#         logging.info(f"Created PRIMARY KEY on {table_name}({col})")
#     except SQLAlchemyError as e:
#         logging.exception(f"Failed create PK on {table_name}({col}): {e}")
#
# def create_index_if_not_exists(conn, table_name: str, col: str):
#     if not col:
#         return
#     idx_name = f"idx_{table_name}_{col}"
#     if len(idx_name) > 63:
#         idx_name = idx_name[:63]
#     try:
#         conn.execute(text(f'CREATE INDEX IF NOT EXISTS "{idx_name}" ON public."{table_name}" ("{col}");'))
#         conn.commit()
#         logging.info(f"Created index {idx_name} on {table_name}({col})")
#     except SQLAlchemyError as e:
#         logging.exception(f"Failed to create index {idx_name} on {table_name}({col}): {e}")
#
# def fk_constraint_exists(conn, table_name: str, constraint_name: str) -> bool:
#     q = text("""
#         SELECT 1 FROM information_schema.table_constraints
#         WHERE table_schema = 'public' AND table_name = :table AND constraint_name = :name AND constraint_type='FOREIGN KEY'
#         LIMIT 1;
#     """)
#     try:
#         return conn.execute(q, {"table": table_name, "name": constraint_name}).scalar() is not None
#     except SQLAlchemyError:
#         return False
#
# def create_fk_constraint(conn, src_table: str, src_col: str, tgt_table: str, tgt_col: str,
#                          src_values: Optional[pd.Series] = None, tgt_values: Optional[pd.Series] = None):
#     fk_name = f"fk_{src_table}_{src_col}"
#     if len(fk_name) > 60:
#         fk_name = fk_name[:60]
#
#     if fk_constraint_exists(conn, src_table, fk_name):
#         logging.info(f"FK {fk_name} already exists on {src_table}; skipping")
#         return
#
#     not_valid = False
#     if src_values is not None and tgt_values is not None:
#         src_set = set(pd.Series(src_values).dropna().unique())
#         tgt_set = set(pd.Series(tgt_values).dropna().unique())
#         missing = src_set - tgt_set
#         if missing:
#             not_valid = True
#             logging.warning(f"FK {src_table}.{src_col} has {len(missing)} values missing in {tgt_table}.{tgt_col}; creating NOT VALID. Sample missing: {list(missing)[:10]}")
#
#     try:
#         if not_valid:
#             sql = text(f'ALTER TABLE public."{src_table}" ADD CONSTRAINT "{fk_name}" FOREIGN KEY ("{src_col}") REFERENCES public."{tgt_table}" ("{tgt_col}") ON UPDATE CASCADE ON DELETE SET NULL NOT VALID;')
#         else:
#             sql = text(f'ALTER TABLE public."{src_table}" ADD CONSTRAINT "{fk_name}" FOREIGN KEY ("{src_col}") REFERENCES public."{tgt_table}" ("{tgt_col}") ON UPDATE CASCADE ON DELETE SET NULL;')
#         conn.execute(sql)
#         conn.commit()
#         logging.info(f"Created FK {fk_name}: {src_table}.{src_col} -> {tgt_table}.{tgt_col}{' (NOT VALID)' if not_valid else ''}")
#     except SQLAlchemyError as e:
#         logging.exception(f"Failed to create FK {fk_name} on {src_table}: {e}")
#
# # ------------------------------- Main script -------------------------------
#
# if __name__ == "__main__":
#     process_name = start_logging()
#     http = set_api_request_strategy()
#     load_dotenv()
#
#     DB_IP = os.getenv("DB_IP")
#     DB_NAME_1 = os.getenv("DB_NAME_1")
#     DB_NAME_2 = os.getenv("DB_NAME_2")
#     DB_USERNAME = os.getenv("DB_USERNAME")
#     DB_PASSWORD = quote_plus(os.getenv("DB_PASSWORD"))
#     RE_API_KEY = os.getenv("RE_API_KEY")
#
#     # final data_to_download as you provided
#     data_to_download = {
#         'constituent_list': 'https://api.sky.blackbaud.com/constituent/v1/constituents?limit=5000&include_inactive=true&include_deceased=true',
#         'phone_list': 'https://api.sky.blackbaud.com/constituent/v1/phones?limit=5000&include_inactive=true',
#         'school_list': 'https://api.sky.blackbaud.com/constituent/v1/educations?limit=5000',
#         'action_list': 'https://api.sky.blackbaud.com/constituent/v1/actions?limit=5000',
#         'address_list': 'https://api.sky.blackbaud.com/constituent/v1/addresses?limit=5000',
#         'gift_list': 'https://api.sky.blackbaud.com/gift/v1/gifts?limit=5000',
#         'gift_custom_fields': 'https://api.sky.blackbaud.com/gift/v1/gifts/customfields?limit=5000',
#         'campaign_list': 'https://api.sky.blackbaud.com/nxt-data-integration/v1/re/campaigns?limit=5000',
#         'relationship_list': 'https://api.sky.blackbaud.com/constituent/v1/relationships?limit=5000',
#         'email_list': 'https://api.sky.blackbaud.com/constituent/v1/emailaddresses?limit=5000&include_inactive=true',
#         'online_presence_list': 'https://api.sky.blackbaud.com/constituent/v1/onlinepresences?limit=5000&include_inactive=true',
#         'constituent_code_list': 'https://api.sky.blackbaud.com/constituent/v1/constituents/constituentcodes?limit=5000&include_inactive=true',
#         'fund_list': 'https://api.sky.blackbaud.com/fundraising/v1/funds?limit=5000',
#         'constituent_custom_fields': 'https://api.sky.blackbaud.com/constituent/v1/constituents/customfields?limit=5000'
#     }
#
#     # Keep DataFrames for PK/FK post-processing
#     table_dfs: Dict[str, pd.DataFrame] = {}
#
#     # --- Download, process, and write to DB_NAME_1 (replace) and DB_NAME_2 (append) ---
#     for table_name, endpoint in data_to_download.items():
#         logging.info(f"Processing {table_name}")
#         housekeeping()
#         try:
#             pagination_api_request(endpoint)
#             df = load_from_json_to_df()
#             if df.empty:
#                 logging.info(f"No rows returned for {table_name}; writing empty table schema")
#                 df = pd.DataFrame()
#
#             # Convert date columns
#             date_cols = detect_date_columns(df)
#
#             if date_cols:
#                 df = convert_date_columns(df, date_cols)
#
#             dtype_map = sqlalchemy_dtype_map_for_dates(date_cols)
#
#             # Dump CSV for records
#             try:
#                 df.to_csv(f"Data Dumps/{table_name}.csv", index=False, quoting=1, lineterminator='\r\n')
#             except Exception:
#                 logging.warning(f"Failed to export CSV for {table_name}")
#
#             # ---------- DB_NAME_1: replace behavior ----------
#             conn1 = connect_db(DB_NAME_1)
#
#             try:
#                 # Drop the table if exists (CASCADE to remove dependent constraints)
#                 drop_table_cascade_if_exists(conn1, table_name)
#
#                 # Write the DataFrame (append will create table)
#                 df.to_sql(table_name, con=conn1, if_exists='replace', index=False, dtype=dtype_map, method="multi")
#                 logging.info(f"DB_NAME_1: Wrote table {table_name} (replaced) with {len(df)} rows")
#             except Exception as e:
#                 logging.exception(f"DB_NAME_1: Failed to write table {table_name}: {e}")
#                 # Attempt a safer replace: ensure table dropped and try again
#                 try:
#                     drop_table_cascade_if_exists(conn1, table_name)
#                     df.to_sql(table_name, con=conn1, if_exists='replace', index=False, dtype=dtype_map, method="multi")
#                     logging.info(f"DB_NAME_1: Retry write succeeded for {table_name}")
#                 except Exception as e2:
#                     logging.exception(f"DB_NAME_1: Retry failed for {table_name}: {e2}")
#             finally:
#                 conn1.close()
#
#             # ---------- DB_NAME_2: historical append ----------
#             conn2 = connect_db(DB_NAME_2)
#             try:
#                 df_hist = df.copy()
#                 df_hist['downloaded_on'] = pd.Timestamp.now()
#                 # If table exists we append duplicates; we leave DB_NAME_2 behavior as append for historical tracking
#                 df_hist.to_sql(table_name, con=conn2, if_exists='append', index=False, dtype=dtype_map, method="multi")
#                 logging.info(f"DB_NAME_2: Appended {len(df_hist)} rows to {table_name}")
#             except Exception as e:
#                 logging.exception(f"DB_NAME_2: Failed to write table {table_name}: {e}")
#             finally:
#                 conn2.close()
#
#             # Save DataFrame in memory for later FK/PK handling
#             table_dfs[table_name] = df
#
#         except Exception as e:
#             logging.exception(f"Top-level processing error for {table_name}: {e}")
#
#     # ------------------------ PHASE 1: CREATE PRIMARY KEYS (DB_NAME_1) ------------------------
#     logging.info("PHASE 1: Creating primary keys on DB_NAME_1")
#     conn_pk = connect_db(DB_NAME_1)
#     try:
#         # Explicit PK plan
#         pk_plan = [
#             ('constituent_list', 'id'),
#             ('gift_list', 'id'),
#             ('fund_list', 'id'),
#             ('campaign_list', 'id'),
#         ]
#         for t, c in pk_plan:
#             # Only create PK if table & column exist in memory data or in DB
#             if t in table_dfs and c in table_dfs[t].columns:
#                 add_primary_key(conn_pk, t, c)
#             else:
#                 # table may exist in DB even if not in memory; attempt to add if column exists in DB
#                 try:
#                     # Check column existence
#                     chk = text("""
#                         SELECT 1
#                         FROM information_schema.columns
#                         WHERE table_schema='public' AND table_name=:table AND column_name=:col
#                         LIMIT 1;
#                     """)
#                     col_exists = conn_pk.execute(chk, {"table": t, "col": c}).scalar() is not None
#                     if col_exists:
#                         add_primary_key(conn_pk, t, c)
#                     else:
#                         logging.warning(f"PK target skipped: {t}.{c} missing")
#                 except Exception as e:
#                     logging.exception(f"Error checking column existence for PK {t}.{c}: {e}")
#
#         # Try to infer other PKs where a column is non-null and unique (skip 'id' since handled)
#         for table_name, df in table_dfs.items():
#             try:
#                 if table_has_pk(conn_pk, table_name):
#                     continue
#                 candidate_pk = None
#                 for col in df.columns:
#                     if col.lower() == 'id':
#                         continue
#                     s = df[col]
#                     if s.notna().all() and s.is_unique:
#                         candidate_pk = col
#                         break
#                 if candidate_pk:
#                     add_primary_key(conn_pk, table_name, candidate_pk)
#             except Exception as e:
#                 logging.exception(f"Failed to infer PK for {table_name}: {e}")
#
#     finally:
#         conn_pk.close()
#
#     # ------------------------ PHASE 2: CREATE FKs (DB_NAME_1) & INDEXES ------------------------
#     logging.info("PHASE 2: Creating foreign keys & indexes on DB_NAME_1")
#
#     # Custom FK rules (easy to extend)
#     CUSTOM_FK_RULES = [
#         # src_table, src_col, tgt_table, tgt_col
#         ('action_list', 'constituent_id', 'constituent_list', 'id'),
#         ('address_list', 'constituent_id', 'constituent_list', 'id'),
#         ('constituent_code_list', 'constituent_id', 'constituent_list', 'id'),
#         ('constituent_custom_fields', 'parent_id', 'constituent_list', 'id'),
#         ('email_list', 'constituent_id', 'constituent_list', 'id'),
#         ('gift_list', 'constituent_id', 'constituent_list', 'id'),
#         ('online_presence_list', 'constituent_id', 'constituent_list', 'id'),
#         ('phone_list', 'constituent_id', 'constituent_list', 'id'),
#         ('relationship_list', 'constituent_id', 'constituent_list', 'id'),
#         ('school_list', 'constituent_id', 'constituent_list', 'id'),
#         ('gift_custom_fields', 'parent_id', 'gift_list', 'id'),
#         ('gift_list', 'gift_splits_0_fund_id', 'fund_list', 'id'),
#         ('gift_list', 'gift_splits_0_campaign_id', 'campaign_list', 'id'),
#     ]
#
#     conn_fk = connect_db(DB_NAME_1)
#     try:
#         # Create indexes on PKs first (explicit list + any pk we created)
#         # PKs defined earlier:
#         pk_index_plan = [
#             ('constituent_list', 'id'),
#             ('gift_list', 'id'),
#             ('fund_list', 'id'),
#             ('campaign_list', 'id'),
#         ]
#         # Add PKs present in table_dfs (if any additional PK candidates were created earlier)
#         for t, df in table_dfs.items():
#             # if table has a column named 'id' and table_has_pk is true, ensure index (safe to try)
#             try:
#                 if 'id' in df.columns:
#                     create_index_if_not_exists(conn_fk, t, 'id')
#             except Exception:
#                 pass
#
#         for t, c in pk_index_plan:
#             create_index_if_not_exists(conn_fk, t, c)
#
#         # Process custom FKs (create indexes on FK columns and create the FK constraint)
#         for src_table, src_col, tgt_table, tgt_col in CUSTOM_FK_RULES:
#             src_df = table_dfs.get(src_table, pd.DataFrame())
#             tgt_df = table_dfs.get(tgt_table, pd.DataFrame())
#
#             # Create index on source FK column if exists
#             if src_table in table_dfs and src_col in src_df.columns:
#                 create_index_if_not_exists(conn_fk, src_table, src_col)
#             else:
#                 # still attempt index creation in DB if column exists (in case source table had empty DataFrame)
#                 try:
#                     chk = text("""
#                         SELECT 1 FROM information_schema.columns
#                         WHERE table_schema='public' AND table_name=:table AND column_name=:col LIMIT 1;
#                     """)
#                     if conn_fk.execute(chk, {"table": src_table, "col": src_col}).scalar() is not None:
#                         create_index_if_not_exists(conn_fk, src_table, src_col)
#                     else:
#                         logging.warning(f"Index skip: {src_table}.{src_col} does not exist in DB (or in memory).")
#                 except Exception as e:
#                     logging.exception(f"Error checking column existence for index {src_table}.{src_col}: {e}")
#
#             # Create index on target PK column (target should exist)
#             if tgt_table in table_dfs and tgt_col in tgt_df.columns:
#                 create_index_if_not_exists(conn_fk, tgt_table, tgt_col)
#             else:
#                 try:
#                     if conn_fk.execute(text("""
#                         SELECT 1 FROM information_schema.columns
#                         WHERE table_schema='public' AND table_name=:table AND column_name=:col LIMIT 1;
#                     """), {"table": tgt_table, "col": tgt_col}).scalar() is not None:
#                         create_index_if_not_exists(conn_fk, tgt_table, tgt_col)
#                 except Exception as e:
#                     logging.exception(f"Error checking column existence for index {tgt_table}.{tgt_col}: {e}")
#
#             # Create FK constraint itself, using in-memory values for best-effort validation
#             src_values = None
#             tgt_values = None
#             if src_table in table_dfs and src_col in table_dfs[src_table].columns:
#                 src_values = table_dfs[src_table][src_col]
#             if tgt_table in table_dfs and tgt_col in table_dfs[tgt_table].columns:
#                 tgt_values = table_dfs[tgt_table][tgt_col]
#
#             # Only attempt FK creation if source and target columns exist in DB (or in-memory)
#             try:
#                 # double-check column existence via information_schema
#                 src_col_exists = conn_fk.execute(text("""
#                     SELECT 1 FROM information_schema.columns
#                     WHERE table_schema='public' AND table_name=:table AND column_name=:col LIMIT 1;
#                 """), {"table": src_table, "col": src_col}).scalar() is not None
#
#                 tgt_col_exists = conn_fk.execute(text("""
#                     SELECT 1 FROM information_schema.columns
#                     WHERE table_schema='public' AND table_name=:table AND column_name=:col LIMIT 1;
#                 """), {"table": tgt_table, "col": tgt_col}).scalar() is not None
#
#                 if src_col_exists and tgt_col_exists:
#                     create_fk_constraint(conn_fk, src_table, src_col, tgt_table, tgt_col, src_values, tgt_values)
#                 else:
#                     logging.warning(f"FK skip: columns missing: {src_table}.{src_col} exists? {src_col_exists}, {tgt_table}.{tgt_col} exists? {tgt_col_exists}")
#             except Exception as e:
#                 logging.exception(f"Error during FK creation check for {src_table}.{src_col} -> {tgt_table}.{tgt_col}: {e}")
#
#     finally:
#         conn_fk.close()
#
#     # ------------------------ FIN ------------------------
#     housekeeping()
#     logging.info("Script finished")
#     sys.exit(0)

import os
import logging
import json
import sys
import glob
from typing import Dict, List, Optional

import pandas as pd
import requests
from flatten_json import flatten
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.types import DateTime
from requests.adapters import HTTPAdapter
from urllib3 import Retry
from urllib.parse import quote_plus
from dotenv import load_dotenv


# ------------------------------- Helper functions -------------------------------

def start_logging():
    p = os.path.basename(__file__).replace('.py', '').replace(' ', '_')
    os.makedirs('Logs', exist_ok=True)
    os.makedirs('Data Dumps', exist_ok=True)
    logging.basicConfig(filename=f'Logs/{p}.log',
                        format='%(asctime)s %(levelname)s %(message)s',
                        filemode='w',
                        level=logging.DEBUG)
    logging.info("Starting script")
    return p


def set_api_request_strategy():
    retry_strategy = Retry(
        total=3,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=['HEAD', 'GET', 'OPTIONS', 'GET'],
        backoff_factor=3
    )
    s = requests.Session()
    adapter = HTTPAdapter(max_retries=retry_strategy)
    s.mount('https://', adapter)
    return s


def housekeeping():
    for f in glob.glob('*_RE_*.json'):
        try:
            os.remove(f)
        except Exception:
            pass


def retrieve_token():
    with open('access_token_output.json') as fh:
        d = json.load(fh)
    return d.get('access_token')


def get_request_re(url):
    headers = {
        'Bb-Api-Subscription-Key': RE_API_KEY,
        'Authorization': 'Bearer ' + retrieve_token()
    }
    return http.get(url, params={}, headers=headers).json()


def pagination_api_request(url):
    logging.info(f"Paginating {url}")
    housekeeping()
    while url:
        resp = get_request_re(url)
        # save
        i = 1
        while os.path.exists(f'API_Response_RE_{process_name}_{i}.json'):
            i += 1
        with open(f'API_Response_RE_{process_name}_{i}.json', 'w') as fh:
            json.dump(resp, fh, ensure_ascii=False, indent=2)
        # if next_link exists, continue (Blackbaud returns next_link key)
        next_link = resp.get('next_link') or resp.get('next')
        if next_link:
            url = next_link
        else:
            break


def load_from_json_to_df():
    file_list = glob.glob(f'API_Response_RE_{process_name}_*.json')
    df = pd.DataFrame()
    for each_file in file_list:
        with open(each_file, 'r') as fh:
            content = json.load(fh)
        vals = content.get('value', [])
        if not isinstance(vals, list):
            logging.warning(f"{each_file}: 'value' not list; skipping")
            continue
        df_part = pd.DataFrame((flatten(v) for v in vals), columns=None) if vals else pd.DataFrame()

        df = pd.concat([df, df_part], ignore_index=True) if not df_part.empty else df

        # Change id cols to integer
        id_cols = [x for x in df.columns if (x == 'id' or '_id' in x) and x not in ('lookup_id', 'campaign_id')]
        for col in id_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce').astype('Int64')

    return df


def detect_date_columns(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if 'date' in c.lower() and 'birth' not in c.lower() and 'deceased' not in c.lower()
            and not c.lower().endswith('_y') and not c.lower().endswith('_d') and not c.lower().endswith('_m')]


def convert_date_columns(df: pd.DataFrame, date_cols: List[str]) -> pd.DataFrame:
    for c in date_cols:
        try:
            df[c] = pd.to_datetime(df[c], utc=True, errors='coerce')
        except Exception as e:
            logging.warning(f"Failed to parse {c}: {e}")
    return df


def sqlalchemy_dtype_map_for_dates(date_cols: List[str]) -> Dict[str, DateTime]:
    return {c: DateTime(timezone=True) for c in date_cols}


def connect_db(dbname: str):
    engine = create_engine(f'postgresql+psycopg2://{DB_USERNAME}:{DB_PASSWORD}@{DB_IP}:5432/{dbname}',
                           pool_recycle=3600)
    conn = engine.connect()
    return conn


def table_exists(conn, table_name: str) -> bool:
    """Check if a table exists in the database"""
    q = text("""
             SELECT 1
             FROM information_schema.tables
             WHERE table_schema = 'public'
               AND table_name = :table LIMIT 1;
             """)
    try:
        return conn.execute(q, {"table": table_name}).scalar() is not None
    except SQLAlchemyError:
        return False


def truncate_table(conn, table_name: str):
    """Truncate a table (preserves structure, constraints, and views)"""
    try:
        conn.execute(text(f'TRUNCATE TABLE public."{table_name}" CASCADE;'))
        conn.commit()
        logging.info(f"Truncated table {table_name}")
    except SQLAlchemyError as e:
        logging.exception(f"Failed to TRUNCATE TABLE {table_name}: {e}")
        raise


def disable_foreign_keys(conn, table_name: str):
    """Disable foreign key constraints on a table"""
    try:
        conn.execute(text(f'ALTER TABLE public."{table_name}" DISABLE TRIGGER ALL;'))
        conn.commit()
        logging.info(f"Disabled triggers/FKs on {table_name}")
    except SQLAlchemyError as e:
        logging.exception(f"Failed to disable FKs on {table_name}: {e}")


def enable_foreign_keys(conn, table_name: str):
    """Re-enable foreign key constraints on a table"""
    try:
        conn.execute(text(f'ALTER TABLE public."{table_name}" ENABLE TRIGGER ALL;'))
        conn.commit()
        logging.info(f"Enabled triggers/FKs on {table_name}")
    except SQLAlchemyError as e:
        logging.exception(f"Failed to enable FKs on {table_name}: {e}")


def get_primary_key_column(conn, table_name: str) -> Optional[str]:
    """Get the primary key column name for a table"""
    q = text("""
             SELECT a.attname
             FROM pg_index i
                      JOIN pg_attribute a ON a.attrelid = i.indrelid AND a.attnum = ANY (i.indkey)
                      JOIN pg_class c ON c.oid = i.indrelid
             WHERE c.relname = :table
               AND c.relnamespace = (SELECT oid FROM pg_namespace WHERE nspname = 'public')
               AND i.indisprimary LIMIT 1;
             """)
    try:
        result = conn.execute(q, {"table": table_name}).scalar()
        conn.commit()
        return result
    except SQLAlchemyError as e:
        conn.rollback()
        logging.exception(f"Failed to get PK for {table_name}: {e}")
        return None


def remove_duplicates_from_df(df: pd.DataFrame, pk_col: str) -> pd.DataFrame:
    """Remove duplicate rows based on primary key column, keeping the first occurrence"""
    if pk_col and pk_col in df.columns:
        original_len = len(df)
        df = df.drop_duplicates(subset=[pk_col], keep='first')
        if len(df) < original_len:
            logging.warning(f"Removed {original_len - len(df)} duplicate rows based on PK column '{pk_col}'")
    return df


def table_has_pk(conn, table_name: str) -> bool:
    q = text("""
             SELECT 1
             FROM information_schema.table_constraints tc
             WHERE tc.table_schema = 'public'
               AND tc.table_name = :table
               AND tc.constraint_type = 'PRIMARY KEY' LIMIT 1;
             """)
    try:
        return conn.execute(q, {"table": table_name}).scalar() is not None
    except SQLAlchemyError:
        return False
    q = text("""
             SELECT 1
             FROM information_schema.table_constraints tc
             WHERE tc.table_schema = 'public'
               AND tc.table_name = :table
               AND tc.constraint_type = 'PRIMARY KEY' LIMIT 1;
             """)
    try:
        return conn.execute(q, {"table": table_name}).scalar() is not None
    except SQLAlchemyError:
        return False


def add_primary_key(conn, table_name: str, col: str):
    if not col:
        return
    try:
        if table_has_pk(conn, table_name):
            logging.info(f"PK exists on {table_name}; skipping creation")
            return
        conn.execute(text(f'ALTER TABLE public."{table_name}" ADD PRIMARY KEY ("{col}");'))
        conn.commit()
        logging.info(f"Created PRIMARY KEY on {table_name}({col})")
    except SQLAlchemyError as e:
        logging.exception(f"Failed create PK on {table_name}({col}): {e}")


def create_index_if_not_exists(conn, table_name: str, col: str):
    if not col:
        return
    idx_name = f"idx_{table_name}_{col}"
    if len(idx_name) > 63:
        idx_name = idx_name[:63]
    try:
        conn.execute(text(f'CREATE INDEX IF NOT EXISTS "{idx_name}" ON public."{table_name}" ("{col}");'))
        conn.commit()
        logging.info(f"Created index {idx_name} on {table_name}({col})")
    except SQLAlchemyError as e:
        logging.exception(f"Failed to create index {idx_name} on {table_name}({col}): {e}")


def fk_constraint_exists(conn, table_name: str, constraint_name: str) -> bool:
    q = text("""
             SELECT 1
             FROM information_schema.table_constraints
             WHERE table_schema = 'public'
               AND table_name = :table
               AND constraint_name = :name
               AND constraint_type = 'FOREIGN KEY' LIMIT 1;
             """)
    try:
        return conn.execute(q, {"table": table_name, "name": constraint_name}).scalar() is not None
    except SQLAlchemyError:
        return False


def create_fk_constraint(conn, src_table: str, src_col: str, tgt_table: str, tgt_col: str,
                         src_values: Optional[pd.Series] = None, tgt_values: Optional[pd.Series] = None):
    fk_name = f"fk_{src_table}_{src_col}"
    if len(fk_name) > 60:
        fk_name = fk_name[:60]

    if fk_constraint_exists(conn, src_table, fk_name):
        logging.info(f"FK {fk_name} already exists on {src_table}; skipping")
        return

    not_valid = False
    if src_values is not None and tgt_values is not None:
        src_set = set(pd.Series(src_values).dropna().unique())
        tgt_set = set(pd.Series(tgt_values).dropna().unique())
        missing = src_set - tgt_set
        if missing:
            not_valid = True
            logging.warning(
                f"FK {src_table}.{src_col} has {len(missing)} values missing in {tgt_table}.{tgt_col}; creating NOT VALID. Sample missing: {list(missing)[:10]}")

    try:
        if not_valid:
            sql = text(
                f'ALTER TABLE public."{src_table}" ADD CONSTRAINT "{fk_name}" FOREIGN KEY ("{src_col}") REFERENCES public."{tgt_table}" ("{tgt_col}") ON UPDATE CASCADE ON DELETE SET NULL NOT VALID;')
        else:
            sql = text(
                f'ALTER TABLE public."{src_table}" ADD CONSTRAINT "{fk_name}" FOREIGN KEY ("{src_col}") REFERENCES public."{tgt_table}" ("{tgt_col}") ON UPDATE CASCADE ON DELETE SET NULL;')
        conn.execute(sql)
        conn.commit()
        logging.info(
            f"Created FK {fk_name}: {src_table}.{src_col} -> {tgt_table}.{tgt_col}{' (NOT VALID)' if not_valid else ''}")
    except SQLAlchemyError as e:
        logging.exception(f"Failed to create FK {fk_name} on {src_table}: {e}")


# ------------------------------- Main script -------------------------------

if __name__ == "__main__":
    process_name = start_logging()
    http = set_api_request_strategy()
    load_dotenv()

    DB_IP = os.getenv("DB_IP")
    DB_NAME_1 = os.getenv("DB_NAME_1")
    DB_NAME_2 = os.getenv("DB_NAME_2")
    DB_USERNAME = os.getenv("DB_USERNAME")
    DB_PASSWORD = quote_plus(os.getenv("DB_PASSWORD"))
    RE_API_KEY = os.getenv("RE_API_KEY")

    # Data to download - ORDERED BY DEPENDENCY
    # Parent tables first, then child tables that reference them
    data_to_download = {
        # Level 1: Independent parent tables (no foreign keys)
        'constituent_list': 'https://api.sky.blackbaud.com/constituent/v1/constituents?limit=5000&include_inactive=true&include_deceased=true',
        'fund_list': 'https://api.sky.blackbaud.com/fundraising/v1/funds?limit=5000',
        'campaign_list': 'https://api.sky.blackbaud.com/nxt-data-integration/v1/re/campaigns?limit=5000',

        # Level 2: Tables that depend on constituent_list
        'phone_list': 'https://api.sky.blackbaud.com/constituent/v1/phones?limit=5000&include_inactive=true',
        'school_list': 'https://api.sky.blackbaud.com/constituent/v1/educations?limit=5000',
        'action_list': 'https://api.sky.blackbaud.com/constituent/v1/actions?limit=5000',
        'address_list': 'https://api.sky.blackbaud.com/constituent/v1/addresses?limit=5000',
        'relationship_list': 'https://api.sky.blackbaud.com/constituent/v1/relationships?limit=5000',
        'email_list': 'https://api.sky.blackbaud.com/constituent/v1/emailaddresses?limit=5000&include_inactive=true',
        'online_presence_list': 'https://api.sky.blackbaud.com/constituent/v1/onlinepresences?limit=5000&include_inactive=true',
        'constituent_code_list': 'https://api.sky.blackbaud.com/constituent/v1/constituents/constituentcodes?limit=5000&include_inactive=true',
        'constituent_custom_fields': 'https://api.sky.blackbaud.com/constituent/v1/constituents/customfields?limit=5000',

        # Level 3: Tables that depend on constituent_list, fund_list, and campaign_list
        'gift_list': 'https://api.sky.blackbaud.com/gift/v1/gifts?limit=5000',

        # Level 4: Tables that depend on gift_list
        'gift_custom_fields': 'https://api.sky.blackbaud.com/gift/v1/gifts/customfields?limit=5000',
    }

    # Keep DataFrames for PK/FK post-processing
    table_dfs: Dict[str, pd.DataFrame] = {}

    # --- Download, process, and write to DB_NAME_1 (truncate & append) and DB_NAME_2 (append) ---
    for table_name, endpoint in data_to_download.items():
        logging.info(f"Processing {table_name}")
        housekeeping()
        try:
            pagination_api_request(endpoint)
            df = load_from_json_to_df()
            if df.empty:
                logging.info(f"No rows returned for {table_name}; will create/truncate table")
                df = pd.DataFrame()

            # Convert date columns
            date_cols = detect_date_columns(df)

            if date_cols:
                df = convert_date_columns(df, date_cols)

            dtype_map = sqlalchemy_dtype_map_for_dates(date_cols)

            # Dump CSV for records
            try:
                df.to_csv(f"Data Dumps/{table_name}.csv", index=False, quoting=1, lineterminator='\r\n')
            except Exception:
                logging.warning(f"Failed to export CSV for {table_name}")

            # ---------- DB_NAME_1: truncate & append behavior ----------
            conn1 = connect_db(DB_NAME_1)

            try:
                # Check if table exists
                if table_exists(conn1, table_name):
                    # Get the primary key column (if exists) to check for duplicates
                    pk_col = get_primary_key_column(conn1, table_name)

                    # Remove duplicates from DataFrame based on PK
                    if pk_col:
                        df = remove_duplicates_from_df(df, pk_col)

                    # Table exists - truncate it and temporarily disable FK checks
                    disable_foreign_keys(conn1, table_name)
                    truncate_table(conn1, table_name)
                    # Append data to the now-empty table
                    df.to_sql(table_name, con=conn1, if_exists='append', index=False, dtype=dtype_map, method="multi")
                    conn1.commit()
                    # Re-enable FK checks
                    enable_foreign_keys(conn1, table_name)
                    logging.info(f"DB_NAME_1: Truncated and appended {len(df)} rows to {table_name}")
                else:
                    # Table doesn't exist - create it
                    df.to_sql(table_name, con=conn1, if_exists='replace', index=False, dtype=dtype_map, method="multi")
                    conn1.commit()
                    logging.info(f"DB_NAME_1: Created new table {table_name} with {len(df)} rows")
            except Exception as e:
                conn1.rollback()
                logging.exception(f"DB_NAME_1: Failed to write table {table_name}: {e}")
                # Try to re-enable FKs even if there was an error
                try:
                    enable_foreign_keys(conn1, table_name)
                except:
                    pass
            finally:
                conn1.close()

            # ---------- DB_NAME_2: historical append ----------
            conn2 = connect_db(DB_NAME_2)
            try:
                df_hist = df.copy()
                df_hist['downloaded_on'] = pd.Timestamp.now()
                # If table exists we append duplicates; we leave DB_NAME_2 behavior as append for historical tracking
                df_hist.to_sql(table_name, con=conn2, if_exists='append', index=False, dtype=dtype_map, method="multi")
                logging.info(f"DB_NAME_2: Appended {len(df_hist)} rows to {table_name}")
            except Exception as e:
                logging.exception(f"DB_NAME_2: Failed to write table {table_name}: {e}")
            finally:
                conn2.close()

            # Save DataFrame in memory for later FK/PK handling
            table_dfs[table_name] = df

        except Exception as e:
            logging.exception(f"Top-level processing error for {table_name}: {e}")

    # ------------------------ PHASE 1: CREATE PRIMARY KEYS (DB_NAME_1) ------------------------
    logging.info("PHASE 1: Creating primary keys on DB_NAME_1")
    conn_pk = connect_db(DB_NAME_1)
    try:
        # Explicit PK plan
        pk_plan = [
            ('constituent_list', 'id'),
            ('gift_list', 'id'),
            ('fund_list', 'id'),
            ('campaign_list', 'id'),
            ('phone_list', 'id'),
            ('school_list', 'id'),
            ('action_list', 'id'),
            ('address_list', 'id'),
            ('relationship_list', 'id'),
            ('email_list', 'id'),
            ('online_presence_list', 'id'),
            ('gift_custom_fields', 'id'),
            ('constituent_custom_fields', 'id'),
            # Note: constituent_code_list uses 'id' as PK, not constituent_id
            # because one constituent can have multiple codes
            ('constituent_code_list', 'id'),
        ]
        for t, c in pk_plan:
            # Only create PK if table & column exist in memory data or in DB
            if t in table_dfs and c in table_dfs[t].columns:
                add_primary_key(conn_pk, t, c)
            else:
                # table may exist in DB even if not in memory; attempt to add if column exists in DB
                try:
                    # Check column existence
                    chk = text("""
                               SELECT 1
                               FROM information_schema.columns
                               WHERE table_schema = 'public'
                                 AND table_name = :table
                                 AND column_name = :col LIMIT 1;
                               """)
                    col_exists = conn_pk.execute(chk, {"table": t, "col": c}).scalar() is not None
                    if col_exists:
                        add_primary_key(conn_pk, t, c)
                    else:
                        logging.warning(f"PK target skipped: {t}.{c} missing")
                except Exception as e:
                    logging.exception(f"Error checking column existence for PK {t}.{c}: {e}")

        # Try to infer other PKs where a column is non-null and unique (skip 'id' since handled)
        for table_name, df in table_dfs.items():
            try:
                if table_has_pk(conn_pk, table_name):
                    continue
                candidate_pk = None
                for col in df.columns:
                    if col.lower() == 'id':
                        continue
                    s = df[col]
                    if s.notna().all() and s.is_unique:
                        candidate_pk = col
                        break
                if candidate_pk:
                    add_primary_key(conn_pk, table_name, candidate_pk)
            except Exception as e:
                logging.exception(f"Failed to infer PK for {table_name}: {e}")

    finally:
        conn_pk.close()

    # ------------------------ PHASE 2: CREATE FKs (DB_NAME_1) & INDEXES ------------------------
    logging.info("PHASE 2: Creating foreign keys & indexes on DB_NAME_1")

    # Custom FK rules (easy to extend)
    CUSTOM_FK_RULES = [
        # src_table, src_col, tgt_table, tgt_col
        ('action_list', 'constituent_id', 'constituent_list', 'id'),
        ('address_list', 'constituent_id', 'constituent_list', 'id'),
        ('constituent_code_list', 'constituent_id', 'constituent_list', 'id'),
        ('constituent_custom_fields', 'parent_id', 'constituent_list', 'id'),
        ('email_list', 'constituent_id', 'constituent_list', 'id'),
        ('gift_list', 'constituent_id', 'constituent_list', 'id'),
        ('online_presence_list', 'constituent_id', 'constituent_list', 'id'),
        ('phone_list', 'constituent_id', 'constituent_list', 'id'),
        ('relationship_list', 'constituent_id', 'constituent_list', 'id'),
        ('school_list', 'constituent_id', 'constituent_list', 'id'),
        ('gift_custom_fields', 'parent_id', 'gift_list', 'id'),
        ('gift_list', 'gift_splits_0_fund_id', 'fund_list', 'id'),
        ('gift_list', 'gift_splits_0_campaign_id', 'campaign_list', 'id'),
    ]

    conn_fk = connect_db(DB_NAME_1)
    try:
        # Create indexes on PKs first (explicit list + any pk we created)
        # PKs defined earlier:
        pk_index_plan = [
            ('constituent_list', 'id'),
            ('gift_list', 'id'),
            ('fund_list', 'id'),
            ('campaign_list', 'id'),
        ]
        # Add PKs present in table_dfs (if any additional PK candidates were created earlier)
        for t, df in table_dfs.items():
            # if table has a column named 'id' and table_has_pk is true, ensure index (safe to try)
            try:
                if 'id' in df.columns:
                    create_index_if_not_exists(conn_fk, t, 'id')
            except Exception:
                pass

        for t, c in pk_index_plan:
            create_index_if_not_exists(conn_fk, t, c)

        # Process custom FKs (create indexes on FK columns and create the FK constraint)
        for src_table, src_col, tgt_table, tgt_col in CUSTOM_FK_RULES:
            src_df = table_dfs.get(src_table, pd.DataFrame())
            tgt_df = table_dfs.get(tgt_table, pd.DataFrame())

            # Create index on source FK column if exists
            if src_table in table_dfs and src_col in src_df.columns:
                create_index_if_not_exists(conn_fk, src_table, src_col)
            else:
                # still attempt index creation in DB if column exists (in case source table had empty DataFrame)
                try:
                    chk = text("""
                               SELECT 1
                               FROM information_schema.columns
                               WHERE table_schema = 'public'
                                 AND table_name = :table
                                 AND column_name = :col LIMIT 1;
                               """)
                    if conn_fk.execute(chk, {"table": src_table, "col": src_col}).scalar() is not None:
                        create_index_if_not_exists(conn_fk, src_table, src_col)
                    else:
                        logging.warning(f"Index skip: {src_table}.{src_col} does not exist in DB (or in memory).")
                except Exception as e:
                    logging.exception(f"Error checking column existence for index {src_table}.{src_col}: {e}")

            # Create index on target PK column (target should exist)
            if tgt_table in table_dfs and tgt_col in tgt_df.columns:
                create_index_if_not_exists(conn_fk, tgt_table, tgt_col)
            else:
                try:
                    if conn_fk.execute(text("""
                                            SELECT 1
                                            FROM information_schema.columns
                                            WHERE table_schema = 'public'
                                              AND table_name = :table
                                              AND column_name = :col LIMIT 1;
                                            """), {"table": tgt_table, "col": tgt_col}).scalar() is not None:
                        create_index_if_not_exists(conn_fk, tgt_table, tgt_col)
                except Exception as e:
                    logging.exception(f"Error checking column existence for index {tgt_table}.{tgt_col}: {e}")

            # Create FK constraint itself, using in-memory values for best-effort validation
            src_values = None
            tgt_values = None
            if src_table in table_dfs and src_col in table_dfs[src_table].columns:
                src_values = table_dfs[src_table][src_col]
            if tgt_table in table_dfs and tgt_col in table_dfs[tgt_table].columns:
                tgt_values = table_dfs[tgt_table][tgt_col]

            # Only attempt FK creation if source and target columns exist in DB (or in-memory)
            try:
                # double-check column existence via information_schema
                src_col_exists = conn_fk.execute(text("""
                                                      SELECT 1
                                                      FROM information_schema.columns
                                                      WHERE table_schema = 'public'
                                                        AND table_name = :table
                                                        AND column_name = :col LIMIT 1;
                                                      """), {"table": src_table, "col": src_col}).scalar() is not None

                tgt_col_exists = conn_fk.execute(text("""
                                                      SELECT 1
                                                      FROM information_schema.columns
                                                      WHERE table_schema = 'public'
                                                        AND table_name = :table
                                                        AND column_name = :col LIMIT 1;
                                                      """), {"table": tgt_table, "col": tgt_col}).scalar() is not None

                if src_col_exists and tgt_col_exists:
                    create_fk_constraint(conn_fk, src_table, src_col, tgt_table, tgt_col, src_values, tgt_values)
                else:
                    logging.warning(
                        f"FK skip: columns missing: {src_table}.{src_col} exists? {src_col_exists}, {tgt_table}.{tgt_col} exists? {tgt_col_exists}")
            except Exception as e:
                logging.exception(
                    f"Error during FK creation check for {src_table}.{src_col} -> {tgt_table}.{tgt_col}: {e}")

    finally:
        conn_fk.close()

    # ------------------------ FIN ------------------------
    housekeeping()
    logging.info("Script finished")
    sys.exit(0)