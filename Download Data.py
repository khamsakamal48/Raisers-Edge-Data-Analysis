import os
import logging
import json
import sys
import glob
import time
from typing import Dict, List, Optional, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import Manager, cpu_count

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
    """Clean up temporary JSON files and checkpoint files."""
    for f in glob.glob('*_RE_*.json'):
        try:
            os.remove(f)
        except OSError as e:
            logging.warning(f"Error removing file {f}: {e}")
    for f in glob.glob('checkpoint_*.json'):
        try:
            os.remove(f)
        except OSError as e:
            logging.warning(f"Error removing checkpoint file {f}: {e}")


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


def connect_db(dbname: str, db_ip: str = None, db_username: str = None, db_password: str = None):
    """
    Create a database connection with optimized pool settings for parallel workers.
    Each connection is disposed after use to prevent pool exhaustion.

    Args:
        dbname: Database name to connect to
        db_ip: Database IP (uses global DB_IP if not provided)
        db_username: Database username (uses global DB_USERNAME if not provided)
        db_password: Database password (uses global DB_PASSWORD if not provided)
    """
    try:
        # Use provided parameters or fall back to global variables
        _db_ip = db_ip if db_ip is not None else DB_IP
        _db_username = db_username if db_username is not None else DB_USERNAME
        _db_password = db_password if db_password is not None else DB_PASSWORD

        # Use smaller pool size for worker processes to prevent connection exhaustion
        # pool_size=1 and max_overflow=0 means each engine uses at most 1 connection
        engine = create_engine(
            f'postgresql+psycopg2://{_db_username}:{_db_password}@{_db_ip}:5432/{dbname}',
            pool_recycle=3600,
            pool_size=1,
            max_overflow=0,
            pool_pre_ping=True  # Verify connections before using them
        )
        conn = engine.connect()
        return conn
    except Exception as e:
        logging.critical(f"Failed to connect to database {dbname}: {e}")
        raise  # Raise exception instead of sys.exit to allow proper error handling in workers


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


# ------------------------------- Checkpoint/State Management -------------------------------

def load_checkpoint(table_name: str) -> Dict:
    """
    Load checkpoint state for a table download.
    Returns dict with keys: last_page, last_url, is_complete, downloaded_pages (list)
    """
    checkpoint_file = f'checkpoint_{table_name}.json'
    if os.path.exists(checkpoint_file):
        try:
            with open(checkpoint_file, 'r') as f:
                checkpoint = json.load(f)
                logging.info(f"Loaded checkpoint for {table_name}: page {checkpoint.get('last_page', 0)}, "
                           f"complete={checkpoint.get('is_complete', False)}")
                return checkpoint
        except Exception as e:
            logging.warning(f"Failed to load checkpoint for {table_name}: {e}")
    return {'last_page': 0, 'last_url': None, 'is_complete': False, 'downloaded_pages': []}


def save_checkpoint(table_name: str, page_num: int, next_url: Optional[str], is_complete: bool, downloaded_pages: List[int]):
    """
    Save checkpoint state for a table download.
    """
    checkpoint = {
        'last_page': page_num,
        'last_url': next_url,
        'is_complete': is_complete,
        'downloaded_pages': downloaded_pages,
        'timestamp': time.time()
    }
    checkpoint_file = f'checkpoint_{table_name}.json'
    try:
        with open(checkpoint_file, 'w') as f:
            json.dump(checkpoint, f, indent=2)
        logging.debug(f"Saved checkpoint for {table_name}: page {page_num}, complete={is_complete}")
    except Exception as e:
        logging.warning(f"Failed to save checkpoint for {table_name}: {e}")


def clear_checkpoint(table_name: str):
    """
    Remove checkpoint file for a table after successful complete download.
    """
    checkpoint_file = f'checkpoint_{table_name}.json'
    if os.path.exists(checkpoint_file):
        try:
            os.remove(checkpoint_file)
            logging.info(f"Cleared checkpoint for {table_name}")
        except Exception as e:
            logging.warning(f"Failed to clear checkpoint for {table_name}: {e}")


def retry_with_backoff(func, max_retries=5, initial_delay=1, max_delay=60, backoff_factor=2):
    """
    Retry a function with exponential backoff.

    Args:
        func: Function to retry (should be a lambda or callable)
        max_retries: Maximum number of retry attempts
        initial_delay: Initial delay in seconds
        max_delay: Maximum delay in seconds
        backoff_factor: Multiplier for delay after each retry

    Returns:
        Result of func() if successful

    Raises:
        Last exception if all retries fail
    """
    delay = initial_delay
    last_exception = None

    for attempt in range(max_retries):
        try:
            return func()
        except requests.exceptions.RequestException as e:
            last_exception = e
            if attempt < max_retries - 1:
                # Check if it's a permanent error (4xx except 429)
                if hasattr(e, 'response') and e.response is not None:
                    status_code = e.response.status_code
                    if 400 <= status_code < 500 and status_code != 429:
                        logging.error(f"Permanent error (status {status_code}), not retrying: {e}")
                        raise

                logging.warning(f"Request failed (attempt {attempt + 1}/{max_retries}): {e}")
                logging.info(f"Retrying in {delay} seconds...")
                time.sleep(delay)
                delay = min(delay * backoff_factor, max_delay)
            else:
                logging.error(f"All {max_retries} retry attempts failed")
                raise

    raise last_exception


# ------------------------------- Download function (sequential) -------------------------------

def download_table_data(table_name: str, endpoint: str, re_api_key: str) -> Tuple[pd.DataFrame, bool]:
    """
    Download data for a single table sequentially with resume capability.
    Returns tuple of (DataFrame, is_complete) where is_complete indicates successful download.
    Uses checkpoints to resume from network failures.
    """
    logging.info(f"Starting download for {table_name}")

    # Set up HTTP session
    http = set_api_request_strategy()

    # Create a unique process name for this download
    process_name = f"download_{table_name}"
    download_complete = False  # Track if download completed successfully

    try:
        # Load checkpoint to check if we have an incomplete download
        checkpoint = load_checkpoint(table_name)

        # If we have a complete download, check if the files still exist
        if checkpoint['is_complete']:
            existing_files = glob.glob(f'API_Response_RE_{process_name}_*.json')
            if existing_files:
                logging.info(f"Found complete checkpoint for {table_name} with existing files, loading data")
                download_complete = True  # Mark as complete since we have checkpoint
                # Skip download, just load the existing files
            else:
                logging.warning(f"Checkpoint says complete but files missing, restarting download for {table_name}")
                checkpoint = {'last_page': 0, 'last_url': None, 'is_complete': False, 'downloaded_pages': []}
                clear_checkpoint(table_name)

        if not checkpoint['is_complete']:
            # Need to download (either fresh start or resume)
            if checkpoint['last_page'] > 0:
                logging.info(f"Resuming download for {table_name} from page {checkpoint['last_page'] + 1}")
                # Keep existing files, we'll continue from where we left off
                url = checkpoint['last_url']
                page_num = checkpoint['last_page'] + 1
            else:
                logging.info(f"Starting fresh download for {table_name}")
                # Clean up any old JSON files for this table
                for f in glob.glob(f'API_Response_RE_{process_name}_*.json'):
                    try:
                        os.remove(f)
                    except OSError as e:
                        logging.warning(f"Error removing file {f}: {e}")
                url = endpoint
                page_num = 1

            # Paginate API and save to JSON files
            token = retrieve_token()
            if not token:
                logging.critical(f"Failed to retrieve API token for {table_name}")
                return pd.DataFrame(), False

            headers = {
                'Bb-Api-Subscription-Key': re_api_key,
                'Authorization': 'Bearer ' + token
            }

            downloaded_pages = checkpoint.get('downloaded_pages', [])

            while url:
                try:
                    # Use retry logic with exponential backoff
                    def make_request():
                        response = http.get(url, params={}, headers=headers)
                        response.raise_for_status()
                        return response.json()

                    logging.info(f"Downloading {table_name} page {page_num}")
                    resp = retry_with_backoff(make_request, max_retries=5, initial_delay=2, max_delay=60)

                    # Save the page immediately
                    file_path = f'API_Response_RE_{process_name}_{page_num}.json'
                    with open(file_path, 'w', encoding='utf-8') as fh:
                        json.dump(resp, fh, ensure_ascii=False, indent=2)

                    downloaded_pages.append(page_num)
                    logging.info(f"Successfully saved {table_name} page {page_num}")

                    # Check for next page
                    next_link = resp.get('next_link') or resp.get('next')

                    if next_link:
                        # More pages to download - save checkpoint
                        save_checkpoint(table_name, page_num, next_link, False, downloaded_pages)
                        url = next_link
                        page_num += 1
                    else:
                        # No more pages - download complete!
                        logging.info(f"Download complete for {table_name} - reached end at page {page_num}")
                        save_checkpoint(table_name, page_num, None, True, downloaded_pages)
                        download_complete = True  # Mark as complete
                        break

                except requests.exceptions.RequestException as e:
                    logging.error(f"API request failed for {table_name} at page {page_num}: {e}")
                    logging.error(f"Download incomplete. You can resume by running the script again.")
                    logging.error(f"Progress saved: {len(downloaded_pages)} pages downloaded")
                    # Save checkpoint before exiting
                    save_checkpoint(table_name, page_num - 1, url, False, downloaded_pages)
                    return pd.DataFrame(), False  # Return empty to signal failure
                except Exception as e:
                    logging.exception(f"Unexpected error downloading {table_name} at page {page_num}: {e}")
                    # Save checkpoint before exiting
                    save_checkpoint(table_name, page_num - 1, url, False, downloaded_pages)
                    return pd.DataFrame(), False

        # Load JSON files into DataFrame
        file_list = glob.glob(f'API_Response_RE_{process_name}_*.json')
        if not file_list:
            logging.error(f"No data files found for {table_name}")
            return pd.DataFrame(), False

        logging.info(f"Loading {len(file_list)} pages for {table_name}")
        all_dfs = []
        for each_file in sorted(file_list):  # Sort to maintain order
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
            df = pd.DataFrame()
        else:
            df = pd.concat(all_dfs, ignore_index=True)

            id_cols = [x for x in df.columns if (x == 'id' or '_id' in x) and x not in ('lookup_id', 'campaign_id')]
            for col in id_cols:
                df[col] = pd.to_numeric(df[col], errors='coerce').astype('Int64')

        # Detect and convert date columns
        date_cols = [c for c in df.columns if 'date' in c.lower() and 'birth' not in c.lower()
                     and 'deceased' not in c.lower() and not c.lower().endswith(('_y', '_d', '_m'))]
        if date_cols:
            for c in date_cols:
                df[c] = pd.to_datetime(df[c], utc=True, errors='coerce')

        # Export to CSV
        try:
            df.to_csv(f"Data Dumps/{table_name}.csv", index=False, quoting=1, lineterminator='\r\n')
            logging.info(f"Exported {table_name} to CSV with {len(df)} rows")
        except Exception as e:
            logging.warning(f"Failed to export CSV for {table_name}: {e}")

        # Clean up JSON files and checkpoint only if download was complete
        if download_complete:
            for f in glob.glob(f'API_Response_RE_{process_name}_*.json'):
                try:
                    os.remove(f)
                except OSError as e:
                    logging.warning(f"Error removing file {f}: {e}")
            clear_checkpoint(table_name)
            logging.info(f"Cleaned up temporary files and checkpoint for {table_name}")

        logging.info(f"Download completed for {table_name} with {len(df)} rows")
        return df, download_complete

    except Exception as e:
        logging.exception(f"Error downloading {table_name}: {e}")
        return pd.DataFrame(), False


# ------------------------------- Worker function for parallel database loading -------------------------------

def load_table_to_db_worker(args: Tuple) -> Tuple[str, bool]:
    """
    Worker function to load a single table to database in parallel.
    Returns (table_name, success) where success is True if loading succeeded.
    """
    table_name, df, pk_map, tables_with_cascade, tables_skip_truncate, use_db_name_2, env_vars = args

    # Set up logging for this worker
    worker_log_file = f'Logs/Load_Data_{table_name}.log'
    logging.basicConfig(
        filename=worker_log_file,
        format='%(asctime)s %(levelname)s %(message)s',
        filemode='w',
        level=logging.INFO,
        force=True
    )

    # Extract environment variables
    DB_IP = env_vars['DB_IP']
    DB_NAME_1 = env_vars['DB_NAME_1']
    DB_NAME_2 = env_vars['DB_NAME_2']
    DB_USERNAME = env_vars['DB_USERNAME']
    DB_PASSWORD = env_vars['DB_PASSWORD']

    logging.info(f"Worker started loading {table_name} to database")
    logging.info(f"DataFrame size: {len(df)} rows, {len(df.columns)} columns")
    logging.info(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")

    # Create a copy of the dataframe to avoid any shared memory issues
    df = df.copy()

    try:
        # Detect date columns for dtype mapping
        date_cols = [c for c in df.columns if 'date' in c.lower() and 'birth' not in c.lower()
                     and 'deceased' not in c.lower() and not c.lower().endswith(('_y', '_d', '_m'))]
        dtype_map = {c: DateTime(timezone=True) for c in date_cols}

        # DB_NAME_1: Replace table logic (handles schema changes automatically)
        conn1 = None
        engine1 = None
        try:
            conn1 = connect_db(DB_NAME_1, DB_IP, DB_USERNAME, DB_PASSWORD)
            engine1 = conn1.engine
            with conn1.begin():
                # Remove duplicates before loading
                intended_pk = pk_map.get(table_name)
                if intended_pk:
                    df = remove_duplicates_from_df(df, intended_pk)

                # Use 'replace' to handle schema changes automatically
                # This drops and recreates the table with the correct schema
                if not df.empty:
                    df.to_sql(table_name, con=conn1, if_exists='replace', index=False, dtype=dtype_map,
                              method="multi")
                    logging.info(f"DB_NAME_1: Replaced {table_name} with {len(df)} rows")
                else:
                    # For empty DataFrames, only replace if table doesn't exist
                    if not table_exists(conn1, table_name):
                        df.to_sql(table_name, con=conn1, if_exists='replace', index=False, dtype=dtype_map,
                                  method="multi")
                        logging.info(f"DB_NAME_1: Created empty table {table_name}")
                    else:
                        logging.warning(f"DB_NAME_1: Skipping empty DataFrame for existing table {table_name}")
        finally:
            # Properly close connection and dispose engine to free resources
            if conn1:
                conn1.close()
            if engine1:
                engine1.dispose()
                logging.info(f"Disposed DB_NAME_1 engine for {table_name}")

        # DB_NAME_2: historical append logic
        if use_db_name_2:
            conn2 = None
            engine2 = None
            try:
                conn2 = connect_db(DB_NAME_2, DB_IP, DB_USERNAME, DB_PASSWORD)
                engine2 = conn2.engine
                with conn2.begin():
                    df_hist = df.copy()
                    if not df_hist.empty:
                        df_hist['downloaded_on'] = pd.Timestamp.now(tz='UTC')
                        hist_dtype_map = {**dtype_map, 'downloaded_on': DateTime(timezone=True)}
                        df_hist.to_sql(table_name, con=conn2, if_exists='append', index=False, dtype=hist_dtype_map,
                                       method="multi")
                        logging.info(f"DB_NAME_2: Appended {len(df_hist)} rows to {table_name}")
            finally:
                # Properly close connection and dispose engine to free resources
                if conn2:
                    conn2.close()
                if engine2:
                    engine2.dispose()
                    logging.info(f"Disposed DB_NAME_2 engine for {table_name}")

        logging.info(f"Worker completed loading {table_name} successfully")
        return (table_name, True)

    except MemoryError as e:
        logging.exception(f"Worker ran out of memory loading {table_name}: {e}")
        logging.error(f"DataFrame size: {len(df)} rows, {df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")
        logging.error(f"Consider reducing max_workers or processing {table_name} separately")
        return (table_name, False)
    except SQLAlchemyError as e:
        logging.exception(f"Database error while loading {table_name}: {e}")
        logging.error(f"Error type: {type(e).__name__}")
        return (table_name, False)
    except Exception as e:
        logging.exception(f"Unexpected error loading {table_name}: {e}")
        logging.error(f"Error type: {type(e).__name__}")
        logging.error(f"DataFrame shape: {df.shape}")
        return (table_name, False)
    finally:
        # Explicit cleanup to help with memory management
        try:
            del df  # Explicitly delete the dataframe
            import gc
            gc.collect()
            logging.info(f"Worker cleanup completed for {table_name}")
        except Exception as cleanup_error:
            logging.warning(f"Error during cleanup for {table_name}: {cleanup_error}")


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

    # Tables that should NOT be truncated because their parent uses CASCADE
    # This prevents lock contention during parallel loading
    TABLES_SKIP_TRUNCATE = {
        'gift_custom_fields',  # Will be cascaded by gift_list
        # Add other child tables here if they have CASCADE parents and are loaded in parallel
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
        'notes': 'https://api.sky.blackbaud.com/constituent/v1/notes?limit=5000',
        'opportunity_list': 'https://api.sky.blackbaud.com/opportunity/v1/opportunities?limit=5000',
    }

    pk_plan = [
        ('constituent_list', 'id'), ('gift_list', 'id'), ('fund_list', 'id'),
        ('campaign_list', 'id'), ('phone_list', 'id'), ('school_list', 'id'),
        ('action_list', 'id'), ('address_list', 'id'), ('relationship_list', 'id'),
        ('email_list', 'id'), ('online_presence_list', 'id'), ('gift_custom_fields', 'id'),
        ('constituent_custom_fields', 'id'), ('constituent_code_list', 'id'),
        ('notes', 'id'), ('opportunity_list', 'id')
    ]
    pk_map = dict(pk_plan)

    # Package environment variables for workers
    env_vars = {
        'DB_IP': DB_IP,
        'DB_NAME_1': DB_NAME_1,
        'DB_NAME_2': DB_NAME_2,
        'DB_USERNAME': DB_USERNAME,
        'DB_PASSWORD': DB_PASSWORD,
    }

    # ============================================================================
    # PHASE 1: Download all data SEQUENTIALLY
    # ============================================================================
    logging.info("=" * 80)
    logging.info("PHASE 1: Starting sequential download of all tables")
    logging.info("=" * 80)

    table_dfs: Dict[str, pd.DataFrame] = {}
    incomplete_downloads = []

    for table_name, endpoint in data_to_download.items():
        logging.info(f"Downloading table: {table_name}")
        df, is_complete = download_table_data(table_name, endpoint, RE_API_KEY)

        # Validate that the download completed successfully
        if not df.empty and is_complete:
            table_dfs[table_name] = df
            logging.info(f"Successfully downloaded {table_name} with {len(df)} rows - COMPLETE")
        elif not df.empty and not is_complete:
            logging.error(f"Downloaded {table_name} but download is INCOMPLETE - will not load to database")
            logging.error(f"Run the script again to resume the incomplete download for {table_name}")
            incomplete_downloads.append(table_name)
            table_dfs[table_name] = pd.DataFrame()  # Store empty to signal failure
        else:
            logging.warning(f"Downloaded empty or failed to download {table_name}")
            incomplete_downloads.append(table_name)
            table_dfs[table_name] = df  # Store empty DataFrame to track attempt

    successful_count = len([t for t, df in table_dfs.items() if not df.empty])
    logging.info(f"Sequential download completed. Downloaded {successful_count}/{len(data_to_download)} tables successfully.")

    if incomplete_downloads:
        logging.warning("=" * 80)
        logging.warning(f"INCOMPLETE DOWNLOADS DETECTED: {len(incomplete_downloads)} tables")
        logging.warning(f"Tables with incomplete downloads: {', '.join(incomplete_downloads)}")
        logging.warning("These tables will NOT be loaded to the database.")
        logging.warning("To resume incomplete downloads, run this script again.")
        logging.warning("The script will automatically resume from where it left off.")
        logging.warning("=" * 80)

    # ============================================================================
    # PHASE 2: Load all data to database IN PARALLEL (with dependency management)
    # ============================================================================
    logging.info("=" * 80)
    logging.info("PHASE 2: Starting parallel loading to database with dependency management")
    logging.info("=" * 80)

    # Determine number of worker processes (use up to 50% of available cores, minimum 1, max 8)
    # Reduced from 75% to prevent memory exhaustion with large datasets
    max_workers = min(8, max(1, int(cpu_count() * 0.5)))
    logging.info(f"Starting parallel database loading with {max_workers} workers (detected {cpu_count()} cores)")

    # Define table loading order in batches to respect FK dependencies
    # Tables in the same batch can be loaded in parallel (no dependencies between them)
    # Tables in later batches depend on tables in earlier batches
    LOADING_BATCHES = [
        # Batch 1: Independent parent tables
        ['fund_list', 'campaign_list'],
        # Batch 2: constituent_list (parent of many tables)
        ['constituent_list'],
        # Batch 3: Tables that depend on constituent_list
        ['action_list', 'phone_list', 'school_list', 'email_list',
         'online_presence_list', 'constituent_code_list', 'address_list',
         'relationship_list', 'constituent_custom_fields', 'notes'],
        # Batch 4: gift_list (depends on constituent_list, fund_list, campaign_list)
        ['gift_list', 'opportunity_list'],
        # Batch 5: gift_custom_fields (depends on gift_list)
        ['gift_custom_fields'],
    ]

    # Load tables to database in batches
    load_results: Dict[str, bool] = {}

    for batch_num, batch_tables in enumerate(LOADING_BATCHES, start=1):
        logging.info(f"Loading batch {batch_num}/{len(LOADING_BATCHES)}: {', '.join(batch_tables)}")

        # Prepare arguments for tables in this batch
        batch_worker_args = []
        for table_name in batch_tables:
            if table_name in table_dfs and not table_dfs[table_name].empty:
                batch_worker_args.append((table_name, table_dfs[table_name], pk_map,
                                         TABLES_WITH_CASCADE, TABLES_SKIP_TRUNCATE,
                                         use_db_name_2, env_vars))
            elif table_name in table_dfs:
                logging.warning(f"Skipping {table_name} in batch {batch_num}: DataFrame is empty")
                load_results[table_name] = False
            else:
                logging.warning(f"Skipping {table_name} in batch {batch_num}: not in downloaded tables")

        if not batch_worker_args:
            logging.info(f"No tables to load in batch {batch_num}, skipping")
            continue

        # Load tables in this batch in parallel
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit all load tasks for this batch
            future_to_table = {executor.submit(load_table_to_db_worker, args): args[0]
                             for args in batch_worker_args}

            # Process completed tasks as they finish
            for future in as_completed(future_to_table):
                table_name = future_to_table[future]
                try:
                    result_table_name, success = future.result()
                    load_results[result_table_name] = success
                    if success:
                        logging.info(f"Successfully loaded {result_table_name} to database")
                    else:
                        logging.error(f"Failed to load {result_table_name} to database")
                except Exception as e:
                    logging.exception(f"Exception occurred while loading {table_name}: {e}")
                    load_results[table_name] = False

        # Check if any critical tables failed in this batch
        failed_in_batch = [t for t in batch_tables if load_results.get(t) == False]
        if failed_in_batch:
            logging.warning(f"Batch {batch_num} completed with failures: {', '.join(failed_in_batch)}")

            # Check if critical parent tables failed - if so, warn about dependent tables
            if batch_num == 1 and ('fund_list' in failed_in_batch or 'campaign_list' in failed_in_batch):
                logging.warning("Critical parent tables failed in batch 1. gift_list may have FK issues.")
            elif batch_num == 2 and 'constituent_list' in failed_in_batch:
                logging.error("constituent_list failed in batch 2. Skipping all dependent tables in later batches.")
                # Mark all dependent tables as failed and skip remaining batches
                dependent_tables = ['action_list', 'phone_list', 'school_list', 'email_list',
                                   'online_presence_list', 'constituent_code_list', 'address_list',
                                   'relationship_list', 'constituent_custom_fields', 'gift_list',
                                   'gift_custom_fields', 'notes', 'opportunity_list']
                for dep_table in dependent_tables:
                    load_results[dep_table] = False
                logging.info(f"Marked {len(dependent_tables)} dependent tables as failed due to constituent_list failure")
                break  # Exit the batch loading loop
            elif batch_num == 4 and 'gift_list' in failed_in_batch:
                logging.warning("gift_list failed in batch 4. Skipping gift_custom_fields in batch 5.")
                load_results['gift_custom_fields'] = False
        else:
            logging.info(f"Batch {batch_num} completed successfully")

        logging.info(f"Batch {batch_num} complete. Progress: {sum(1 for v in load_results.values() if v)}/{len(load_results)} tables loaded so far")

    successful_loads = sum(1 for success in load_results.values() if success)
    failed_loads = [table for table, success in load_results.items() if not success]
    logging.info(f"Parallel database loading completed. Loaded {successful_loads}/{len(load_results)} tables successfully.")
    if failed_loads:
        logging.warning(f"Failed tables: {', '.join(failed_loads)}")
        logging.warning("Phase 3 will skip post-processing for failed tables.")

    # ============================================================================
    # PHASE 3: Post-processing - Create PKs, FKs, and Indexes
    # ============================================================================
    logging.info("=" * 80)
    logging.info("PHASE 3: Post-processing - Creating keys and indexes on DB_NAME_1")
    logging.info("=" * 80)

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
        ('notes', 'constituent_id', 'constituent_list', 'id'),
        ('opportunity_list', 'constituent_id', 'constituent_list', 'id'),
        ('opportunity_list', 'fund_id', 'fund_list', 'id'),
        ('opportunity_list', 'campaign_id', 'campaign_list', 'id')
    ]

    with connect_db(DB_NAME_1) as conn:
        # Phase 1: Creating Primary Keys (each in its own transaction)
        logging.info("Phase 1: Creating Primary Keys")
        for t, c in pk_plan:
            # Only process tables that were successfully loaded
            if not load_results.get(t, False):
                logging.warning(f"Skipping PK creation for {t}: table was not successfully loaded")
                continue

            if t in table_dfs and c in table_dfs[t].columns:
                try:
                    with conn.begin():  # Individual transaction for each PK
                        add_primary_key(conn, t, c)
                except Exception as e:
                    logging.error(f"Failed to create PK on {t}({c}), but continuing: {e}")
                    # Continue with next table instead of failing everything

        # Phase 2: Creating Indexes and Foreign Keys (each in its own transaction)
        logging.info("Phase 2: Creating Indexes and Foreign Keys")
        for src_table, src_col, tgt_table, tgt_col in CUSTOM_FK_RULES:
            # Only process if both source and target tables were successfully loaded
            if not load_results.get(src_table, False):
                logging.warning(f"Skipping FK {src_table}.{src_col}: source table was not successfully loaded")
                continue
            if not load_results.get(tgt_table, False):
                logging.warning(f"Skipping FK {src_table}.{src_col}: target table {tgt_table} was not successfully loaded")
                continue

            try:
                with conn.begin():  # Individual transaction for indexes
                    if src_table in table_dfs and src_col in table_dfs[src_table].columns:
                        create_index_if_not_exists(conn, src_table, src_col)

                    if tgt_table in table_dfs and tgt_col in table_dfs[tgt_table].columns:
                        create_index_if_not_exists(conn, tgt_table, tgt_col)
            except Exception as e:
                logging.error(f"Failed to create indexes for FK {src_table}.{src_col}, but continuing: {e}")

            try:
                with conn.begin():  # Individual transaction for FK
                    src_values = table_dfs.get(src_table, pd.DataFrame()).get(src_col)
                    tgt_values = table_dfs.get(tgt_table, pd.DataFrame()).get(tgt_col)
                    create_fk_constraint(conn, src_table, src_col, tgt_table, tgt_col, src_values, tgt_values)
            except Exception as e:
                logging.error(f"Failed to create FK {src_table}.{src_col}, but continuing: {e}")

    housekeeping()
    logging.info("--- Script finished successfully ---")
    sys.exit(0)

