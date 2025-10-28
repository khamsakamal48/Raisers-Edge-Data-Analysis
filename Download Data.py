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
                        level=logging.INFO)
    logging.info("Starting script")
    return p


def set_api_request_strategy():
    retry_strategy = Retry(
        total=3,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=['HEAD', 'GET', 'OPTIONS'],
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
        except OSError as e:
            logging.warning(f"Error removing file {f}: {e}")


def retrieve_token():
    try:
        with open('access_token_output.json') as fh:
            d = json.load(fh)
        return d.get('access_token')
    except FileNotFoundError:
        logging.critical("access_token_output.json not found. Please generate a token.")
        sys.exit(1)


def get_request_re(url):
    token = retrieve_token()
    if not token:
        logging.critical("Failed to retrieve API token.")
        return None
    headers = {
        'Bb-Api-Subscription-Key': RE_API_KEY,
        'Authorization': 'Bearer ' + token
    }
    response = http.get(url, params={}, headers=headers)
    response.raise_for_status()
    return response.json()


def pagination_api_request(url):
    logging.info(f"Paginating {url}")
    housekeeping()
    page_num = 1
    while url:
        try:
            resp = get_request_re(url)
            if not resp:
                break

            file_path = f'API_Response_RE_{process_name}_{page_num}.json'
            with open(file_path, 'w', encoding='utf-8') as fh:
                json.dump(resp, fh, ensure_ascii=False, indent=2)

            next_link = resp.get('next_link') or resp.get('next')
            if next_link:
                url = next_link
                page_num += 1
            else:
                break
        except requests.exceptions.RequestException as e:
            logging.error(f"API request failed for {url}: {e}")
            break


def load_from_json_to_df():
    file_list = glob.glob(f'API_Response_RE_{process_name}_*.json')
    all_dfs = []
    for each_file in file_list:
        with open(each_file, 'r', encoding='utf-8') as fh:
            content = json.load(fh)

        vals = content.get('value', [])
        if not isinstance(vals, list):
            logging.warning(f"{each_file}: 'value' is not a list; skipping")
            continue

        if vals:
            df_part = pd.DataFrame((flatten(v) for v in vals))
            all_dfs.append(df_part)

    if not all_dfs:
        return pd.DataFrame()

    df = pd.concat(all_dfs, ignore_index=True)

    id_cols = [x for x in df.columns if (x == 'id' or '_id' in x) and x not in ('lookup_id', 'campaign_id')]
    for col in id_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').astype('Int64')

    return df


def detect_date_columns(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if 'date' in c.lower() and 'birth' not in c.lower() and 'deceased' not in c.lower()
            and not c.lower().endswith(('_y', '_d', '_m'))]


def convert_date_columns(df: pd.DataFrame, date_cols: List[str]) -> pd.DataFrame:
    for c in date_cols:
        df[c] = pd.to_datetime(df[c], utc=True, errors='coerce')
    return df


def sqlalchemy_dtype_map_for_dates(date_cols: List[str]) -> Dict[str, DateTime]:
    return {c: DateTime(timezone=True) for c in date_cols}


def connect_db(dbname: str):
    try:
        engine = create_engine(f'postgresql+psycopg2://{DB_USERNAME}:{DB_PASSWORD}@{DB_IP}:5432/{dbname}',
                               pool_recycle=3600)
        conn = engine.connect()
        return conn
    except Exception as e:
        logging.critical(f"Failed to connect to database {dbname}: {e}")
        sys.exit(1)


def table_exists(conn, table_name: str) -> bool:
    q = text("""
             SELECT EXISTS (SELECT 1
                            FROM information_schema.tables
                            WHERE table_schema = 'public'
                              AND table_name = :table);
             """)
    return conn.execute(q, {"table": table_name}).scalar_one()


def truncate_table(conn, table_name: str, use_cascade: bool = False):
    try:
        cascade_keyword = 'CASCADE' if use_cascade else ''
        conn.execute(text(f'TRUNCATE TABLE public."{table_name}" {cascade_keyword};'))
        logging.info(f"Successfully truncated table {table_name}" + (" using CASCADE" if use_cascade else ""))
    except SQLAlchemyError as e:
        logging.error(f"Failed to TRUNCATE TABLE {table_name}: {e}")
        raise


def disable_foreign_keys(conn, table_name: str):
    try:
        conn.execute(text(f'ALTER TABLE public."{table_name}" DISABLE TRIGGER ALL;'))
        logging.info(f"Disabled triggers/FKs on {table_name}")
    except SQLAlchemyError as e:
        logging.error(f"Failed to disable FKs on {table_name}: {e}")


def enable_foreign_keys(conn, table_name: str):
    try:
        conn.execute(text(f'ALTER TABLE public."{table_name}" ENABLE TRIGGER ALL;'))
        logging.info(f"Enabled triggers/FKs on {table_name}")
    except SQLAlchemyError as e:
        logging.error(f"Failed to enable FKs on {table_name}: {e}")


def get_primary_key_column(conn, table_name: str) -> Optional[str]:
    q = text("""
             SELECT a.attname
             FROM pg_index i
                      JOIN pg_attribute a ON a.attrelid = i.indrelid AND a.attnum = ANY (i.indkey)
             WHERE i.indrelid = (SELECT oid
                                 FROM pg_class
                                 WHERE relname = :table
                                   AND relnamespace = (SELECT oid
                                                       FROM pg_namespace
                                                       WHERE nspname = 'public'))
               AND i.indisprimary;
             """)
    try:
        result = conn.execute(q, {"table": table_name}).scalar_one_or_none()
        return result
    except SQLAlchemyError as e:
        logging.error(f"Failed to get PK for {table_name}: {e}")
        return None


def get_primary_key_constraint_name(conn, table_name: str) -> Optional[str]:
    """Get the name of the primary key constraint on a table."""
    q = text("""
             SELECT con.conname
             FROM pg_catalog.pg_constraint con
                      INNER JOIN pg_catalog.pg_class rel ON rel.oid = con.conrelid
                      INNER JOIN pg_catalog.pg_namespace nsp ON nsp.oid = rel.relnamespace
             WHERE nsp.nspname = 'public'
               AND rel.relname = :table
               AND con.contype = 'p' LIMIT 1;
             """)
    try:
        return conn.execute(q, {"table": table_name}).scalar_one_or_none()
    except SQLAlchemyError as e:
        logging.error(f"Failed to get PK constraint name for {table_name}: {e}")
        return None


def remove_duplicates_from_df(df: pd.DataFrame, pk_col: str) -> pd.DataFrame:
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


def add_primary_key(conn, table_name: str, col: str):
    if not col:
        return
    try:
        if table_has_pk(conn, table_name):
            logging.info(f"PK exists on {table_name}; skipping creation")
            return
        conn.execute(text(f'ALTER TABLE public."{table_name}" ADD PRIMARY KEY ("{col}");'))
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
    DB_PASSWORD = quote_plus(os.getenv("DB_PASSWORD", ""))
    RE_API_KEY = os.getenv("RE_API_KEY")

    # Check if DB_NAME_2 is available and not blank
    use_db_name_2 = DB_NAME_2 and DB_NAME_2.strip()
    if not use_db_name_2:
        logging.info("DB_NAME_2 is not configured or is blank. Skipping all DB_NAME_2 related activities.")

    TABLES_WITH_CASCADE = {
        'constituent_list',
        'fund_list',
        'campaign_list',
        'gift_list',
    }

    data_to_download = {
        'constituent_list': 'https://api.sky.blackbaud.com/constituent/v1/constituents?limit=5000&include_inactive=true&include_deceased=true',
        'fund_list': 'https://api.sky.blackbaud.com/fundraising/v1/funds?limit=5000',
        'campaign_list': 'https://api.sky.blackbaud.com/nxt-data-integration/v1/re/campaigns?limit=5000',
        'phone_list': 'https://api.sky.blackbaud.com/constituent/v1/phones?limit=5000&include_inactive=true',
        'school_list': 'https://api.sky.blackbaud.com/constituent/v1/educations?limit=5000',
        'action_list': 'https://api.sky.blackbaud.com/constituent/v1/actions?limit=5000',
        'address_list': 'https://api.sky.blackbaud.com/constituent/v1/addresses?limit=5000',
        'relationship_list': 'https://api.sky.blackbaud.com/constituent/v1/relationships?limit=5000',
        'email_list': 'https://api.sky.blackbaud.com/constituent/v1/emailaddresses?limit=5000&include_inactive=true',
        'online_presence_list': 'https://api.sky.blackbaud.com/constituent/v1/onlinepresences?limit=5000&include_inactive=true',
        'constituent_code_list': 'https://api.sky.blackbaud.com/constituent/v1/constituents/constituentcodes?limit=5000&include_inactive=true',
        'constituent_custom_fields': 'https://api.sky.blackbaud.com/constituent/v1/constituents/customfields?limit=5000',
        'gift_list': 'https://api.sky.blackbaud.com/gift/v1/gifts?limit=5000',
        'gift_custom_fields': 'https://api.sky.blackbaud.com/gift/v1/gifts/customfields?limit=5000',
    }

    pk_plan = [
        ('constituent_list', 'id'), ('gift_list', 'id'), ('fund_list', 'id'),
        ('campaign_list', 'id'), ('phone_list', 'id'), ('school_list', 'id'),
        ('action_list', 'id'), ('address_list', 'id'), ('relationship_list', 'id'),
        ('email_list', 'id'), ('online_presence_list', 'id'), ('gift_custom_fields', 'id'),
        ('constituent_custom_fields', 'id'), ('constituent_code_list', 'id'),
    ]
    pk_map = dict(pk_plan)

    table_dfs: Dict[str, pd.DataFrame] = {}

    for table_name, endpoint in data_to_download.items():
        logging.info(f"--- Processing {table_name} ---")
        housekeeping()
        try:
            pagination_api_request(endpoint)
            df = load_from_json_to_df()
            if df.empty:
                logging.info(f"No rows returned for {table_name}; ensuring table exists but is empty.")

            date_cols = detect_date_columns(df)
            if date_cols:
                df = convert_date_columns(df, date_cols)
            dtype_map = sqlalchemy_dtype_map_for_dates(date_cols)

            try:
                df.to_csv(f"Data Dumps/{table_name}.csv", index=False, quoting=1, lineterminator='\r\n')
            except Exception as e:
                logging.warning(f"Failed to export CSV for {table_name}: {e}")

            # DB_NAME_1: truncate & append logic
            with connect_db(DB_NAME_1) as conn1:
                with conn1.begin():  # Manages the transaction block
                    if table_exists(conn1, table_name):
                        intended_pk = pk_map.get(table_name)

                        # Check for and correct a wrongly defined primary key before proceeding
                        existing_pk_col = get_primary_key_column(conn1, table_name)
                        if existing_pk_col and intended_pk and existing_pk_col != intended_pk:
                            logging.warning(f"Table '{table_name}' has an incorrect PK on '{existing_pk_col}'. "
                                            f"Expected '{intended_pk}'. Dropping the incorrect constraint.")
                            pk_constraint_name = get_primary_key_constraint_name(conn1, table_name)
                            if pk_constraint_name:
                                conn1.execute(text(
                                    f'ALTER TABLE public."{table_name}" DROP CONSTRAINT IF EXISTS "{pk_constraint_name}";'))
                                logging.info(f"Dropped incorrect PK constraint '{pk_constraint_name}'.")
                            else:
                                logging.error(
                                    f"Could not find constraint name to drop PK on {table_name}, but an incorrect PK column was detected.")

                        # Use the key from our plan for de-duplication
                        if intended_pk:
                            df = remove_duplicates_from_df(df, intended_pk)

                        disable_foreign_keys(conn1, table_name)
                        use_cascade = table_name in TABLES_WITH_CASCADE
                        truncate_table(conn1, table_name, use_cascade=use_cascade)

                        if not df.empty:
                            df.to_sql(table_name, con=conn1, if_exists='append', index=False, dtype=dtype_map,
                                      method="multi")

                        enable_foreign_keys(conn1, table_name)
                        logging.info(f"DB_NAME_1: Truncated and appended {len(df)} rows to {table_name}")
                    else:
                        df.to_sql(table_name, con=conn1, if_exists='replace', index=False, dtype=dtype_map,
                                  method="multi")
                        logging.info(f"DB_NAME_1: Created new table {table_name} with {len(df)} rows")

            # DB_NAME_2: historical append logic (only if DB_NAME_2 is configured)
            if use_db_name_2:
                with connect_db(DB_NAME_2) as conn2:
                    with conn2.begin():  # Manages the transaction block
                        df_hist = df.copy()
                        if not df_hist.empty:
                            df_hist['downloaded_on'] = pd.Timestamp.now(tz='UTC')
                            hist_dtype_map = {**dtype_map, 'downloaded_on': DateTime(timezone=True)}
                            df_hist.to_sql(table_name, con=conn2, if_exists='append', index=False, dtype=hist_dtype_map,
                                           method="multi")
                            logging.info(f"DB_NAME_2: Appended {len(df_hist)} rows to {table_name}")
            else:
                logging.debug(f"Skipping DB_NAME_2 operations for {table_name} (not configured)")

            table_dfs[table_name] = df

        except Exception as e:
            logging.exception(f"Top-level processing error for {table_name}: {e}")

    # --- PHASE 1 & 2: Create PKs, FKs, and Indexes ---
    logging.info("--- Post-processing: Creating keys and indexes on DB_NAME_1 ---")

    CUSTOM_FK_RULES = [
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

    with connect_db(DB_NAME_1) as conn:
        with conn.begin():  # Manages one transaction for all post-processing
            logging.info("Phase 1: Creating Primary Keys")
            for t, c in pk_plan:
                if t in table_dfs and c in table_dfs[t].columns:
                    add_primary_key(conn, t, c)

            logging.info("Phase 2: Creating Indexes and Foreign Keys")
            for src_table, src_col, tgt_table, tgt_col in CUSTOM_FK_RULES:
                if src_table in table_dfs and src_col in table_dfs[src_table].columns:
                    create_index_if_not_exists(conn, src_table, src_col)

                if tgt_table in table_dfs and tgt_col in table_dfs[tgt_table].columns:
                    create_index_if_not_exists(conn, tgt_table, tgt_col)

                src_values = table_dfs.get(src_table, pd.DataFrame()).get(src_col)
                tgt_values = table_dfs.get(tgt_table, pd.DataFrame()).get(tgt_col)
                create_fk_constraint(conn, src_table, src_col, tgt_table, tgt_col, src_values, tgt_values)

    housekeeping()
    logging.info("--- Script finished successfully ---")
    sys.exit(0)

