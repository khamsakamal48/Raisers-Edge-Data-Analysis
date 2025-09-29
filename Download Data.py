# import os
# import logging
# import base64
# import msal
# import datetime
# import requests
# import pandas as pd
# import glob
# import json
# import sys
#
# from dotenv import load_dotenv
# from datetime import datetime
# from requests.adapters import HTTPAdapter
# from urllib3 import Retry
# from sqlalchemy import create_engine
# from urllib.parse import quote_plus
# from flatten_json import flatten
# from typing import Dict, List, Tuple, Optional
# from sqlalchemy.types import DateTime
# from sqlalchemy import create_engine, text, inspect
# from sqlalchemy.exc import SQLAlchemyError
#
#
# ########################################################################################################################
# #                                                    FUNCTIONS                                                         #
# ########################################################################################################################
# def set_current_directory():
#     logging.info('Setting current directory')
#
#     os.chdir(os.getcwd())
#
#
# def start_logging():
#     # Get File Name of existing script
#     p = os.path.basename(__file__).replace('.py', '').replace(' ', '_')
#
#     logging.basicConfig(filename=f'Logs/{p}.log', format='%(asctime)s %(message)s', filemode='w',
#                         level=logging.DEBUG)
#
#     # Printing the output to file for debugging
#     logging.info('Starting the Script')
#
#     return p
#
#
# def stop_logging():
#     logging.info('Stopping the Script')
#
#
# def set_api_request_strategy():
#     logging.info('Setting API Request strategy')
#
#     # API Request strategy
#     logging.info('Setting API Request Strategy')
#
#     retry_strategy = Retry(
#         total=3,
#         status_forcelist=[429, 500, 502, 503, 504],
#         allowed_methods=['HEAD', 'GET', 'OPTIONS'],
#         backoff_factor=10
#     )
#
#     adapter = HTTPAdapter(max_retries=retry_strategy)
#     h = requests.Session()
#     h.mount('https://', adapter)
#
#     return h
#
#
# def get_recipients(email_list):
#     value = []
#
#     for email in email_list:
#         email = {
#             'emailAddress': {
#                 'address': email
#             }
#         }
#
#         value.append(email)
#
#     return value
#
#
# def pagination_api_request(url):
#     logging.info('Paginating API requests')
#
#     # Housekeeping
#     housekeeping()
#
#     # Pagination request to retrieve list
#     while url:
#         # Blackbaud API GET request
#         re_api_response = get_request_re(url)
#
#         # Incremental File name
#         i = 1
#         while os.path.exists(f'API_Response_RE_{process_name}_{i}.json'):
#             i += 1
#
#         with open(f'API_Response_RE_{process_name}_{i}.json', 'w') as list_output:
#             json.dump(re_api_response, list_output, ensure_ascii=False, sort_keys=True, indent=4)
#
#         # Check if a variable is present in file
#         with open(f'API_Response_RE_{process_name}_{i}.json') as list_output_last:
#
#             if 'next_link' in list_output_last.read():
#                 url = re_api_response['next_link']
#
#             else:
#                 break
#
#             break
#
#
# def retrieve_token():
#     logging.info('Retrieve token for API connections')
#
#     with open('access_token_output.json') as access_token_output:
#         d = json.load(access_token_output)
#         access_token = d['access_token']
#
#     return access_token
#
#
# def get_request_re(url):
#     logging.info('Running GET Request from RE function')
#
#     # Request Headers for Blackbaud API request
#     headers = {
#         # Request headers
#         'Bb-Api-Subscription-Key': RE_API_KEY,
#         'Authorization': 'Bearer ' + retrieve_token(),
#     }
#
#     return http.get(url, params={}, headers=headers).json()
#
#
# def housekeeping():
#     logging.info('Doing Housekeeping')
#
#     # Housekeeping
#     multiple_files = glob.glob('*_RE_*.json')
#
#     # Iterate over the list of filepaths & remove each file.
#     logging.info('Removing old JSON files')
#     for each_file in multiple_files:
#         try:
#             os.remove(each_file)
#         except FileNotFoundError:
#             pass
#
#
# def load_from_json_to_df():
#     logging.info('Loading from JSON to DataFrame')
#
#     # Get a list of all the file paths that ends with wildcard from in specified directory
#     file_list = glob.glob('API_Response_RE_*.json')
#
#     df = pd.DataFrame()
#
#     for each_file in file_list:
#         # Open Each JSON File
#         with open(each_file, 'r') as json_file:
#             # Load JSON File
#             json_content = json.load(json_file)
#
#             # Load from JSON to pandas dataframe
#             df_ = pd.DataFrame(flatten(d) for d in json_content['value'])
#
#             # Append/Concat dataframes
#             df = pd.concat([df, df_])
#
#     return df
#
#
# def connect_db(db):
#     logging.info('Connecting to Database')
#
#     # Create an engine instance
#     alchemy_engine = create_engine(f'postgresql+psycopg2://{DB_USERNAME}:{DB_PASSWORD}@{DB_IP}:5432/{db}',
#                                    pool_recycle=3600)
#
#     # Connect to PostgresSQL server
#     db_connection = alchemy_engine.connect()
#
#     return db_connection
#
#
# def disconnect_db():
#     logging.info('Disconnecting from Database')
#
#     if db_conn:
#         db_conn.close()
#
#
# def load_to_db(df, table):
#     logging.info('Loading to Database')
#
#     # Loading to SQL DB
#     df.to_sql(table, db_conn, if_exists='append', index=False)
#
#
# # Functions to encode email addresses and website URLs
# def encode_data(value: str):
#     # For other values
#     return value[0] + 'x' * (len(value) - 2) + value[-1]
#
#
# def encode_email(value):
#     # If it's an email address
#     parts = value.split('@')
#     if len(parts) == 2:  # Check if the split produced exactly two parts
#         username, domain = parts[0], parts[1]
#         username = username[0] + 'x' * (len(username) - 2) + username[-1]
#         return f"{username}@{domain}"
#     else:
#         # If there's no "@" symbol, just return the original value
#         return value
#
#
# def detect_date_columns(df: pd.DataFrame) -> List[str]:
#     # any column with 'date' in name (case-insensitive)
#     return [c for c in df.columns if 'date' in c.lower() and not 'birth' in c.lower() and not 'deceased' in c.lower()]
#
#
# def convert_date_columns(df: pd.DataFrame, date_cols: List[str]) -> pd.DataFrame:
#     for col in date_cols:
#         # parse ISO-like strings; coerce errors to NaT; keep timezone info by setting utc=True
#         # if you want to preserve original timezone, you can skip utc=True but using utc=True normalizes.
#         try:
#             df[col] = pd.to_datetime(df[col], utc=True, errors="coerce")
#         except Exception as e:
#             logging.warning(f"Failed to parse dates in column {col}: {e}")
#     return df
#
#
# def sqlalchemy_dtype_map_for_dates(date_cols: List[str]) -> Dict[str, DateTime]:
#     # Map date columns to timezone-aware DateTime for Postgres (TIMESTAMP WITH TIME ZONE)
#     dtypes = {}
#     for c in date_cols:
#         dtypes[c] = DateTime(timezone=True)
#     return dtypes
#
#
# try:
#
#     # Start Logging for Debugging
#     process_name = start_logging()
#
#     # Set current directory
#     set_current_directory()
#
#     # Set API request strategy
#     http = set_api_request_strategy()
#
#     # Retrieve contents from .env file
#     load_dotenv()
#
#     DB_IP = os.getenv("DB_IP")
#     DB_NAME_1 = os.getenv("DB_NAME_1")
#     DB_NAME_2 = os.getenv("DB_NAME_2")
#     DB_USERNAME = os.getenv("DB_USERNAME")
#     DB_PASSWORD = quote_plus(os.getenv("DB_PASSWORD"))
#     AUTH_CODE = os.getenv("AUTH_CODE")
#     REDIRECT_URL = os.getenv("REDIRECT_URL")
#     CLIENT_ID = os.getenv("CLIENT_ID")
#     RE_API_KEY = os.getenv("RE_API_KEY")
#     CONSTITUENT_LIST = os.getenv("CONSTITUENT_LIST")
#
#     # Data to download
#     data_to_download = {
#         'constituent_list': 'https://api.sky.blackbaud.com/constituent/v1/constituents?limit=5000&include_inactive=true&include_deceased=true',
#         'phone_list': 'https://api.sky.blackbaud.com/constituent/v1/phones?limit=5000',
#         'school_list': 'https://api.sky.blackbaud.com/constituent/v1/educations?limit=5000',
#         'action_list': 'https://api.sky.blackbaud.com/constituent/v1/actions?limit=5000',
#         'address_list': 'https://api.sky.blackbaud.com/constituent/v1/addresses?limit=5000',
#         'gift_list': 'https://api.sky.blackbaud.com/gift/v1/gifts?limit=5000',
#         'gift_custom_fields': 'https://api.sky.blackbaud.com/gift/v1/gifts/customfields?limit=5000',
#         'campaign_list': 'https://api.sky.blackbaud.com/nxt-data-integration/v1/re/campaigns?limit=5000',
#         'relationship_list': 'https://api.sky.blackbaud.com/constituent/v1/relationships?limit=5000',
#         'email_list': 'https://api.sky.blackbaud.com/constituent/v1/emailaddresses?limit=5000',
#         'online_presence_list': 'https://api.sky.blackbaud.com/constituent/v1/onlinepresences?limit=5000',
#         'constituent_code_list': 'https://api.sky.blackbaud.com/constituent/v1/constituents/constituentcodes?limit=5000&include_inactive=true',
#         'fund_list': 'https://api.sky.blackbaud.com/fundraising/v1/funds?limit=5000',
#         'constituent_custom_fields': 'https://api.sky.blackbaud.com/constituent/v1/constituents/customfields?limit=5000'
#     }
#
#     # Keep DataFrame meta for PK/FK detection after upload
#     table_dfs: Dict[str, pd.DataFrame] = {}
#
#     # Explicit foreign key mapping (table.column -> target_table.target_column)
#     EXPLICIT_FK_RULES = {
#         # means: in ANY table, if you see 'constituent_id', link to constituent_list.id
#         "constituent_id": ("constituent_list", "id"),
#         "parent_id": ("constituent_list", "id"),
#     }
#
#     # candidate FK column names to wire up as FKs if possible
#     FK_CANDIDATES = ["parent_id", "constituent_id", "campaign_id", "fund_id"]
#
#     # Loop across each data point
#     for table_name, endpoint in data_to_download.items():
#         logging.info(f'Working on {table_name}')
#
#         # Housekeeping
#         housekeeping()
#
#         # Download Data
#         pagination_api_request(endpoint)
#
#         # Load to DataFrame
#         data = load_from_json_to_df()
#
#         # Change data type of date columns
#         date_cols = detect_date_columns(data)
#         if date_cols:
#             logging.info("Detected date columns: %s", date_cols)
#             data = convert_date_columns(data, date_cols).copy()
#         else:
#             logging.debug("No date columns found in %s", table_name)
#
#         # Convert numeric-ish columns to numeric where safe
#         for col in data.columns:
#             if col.lower().endswith("_id") or "date" in col.lower() or col.lower() in ['id', 'deceased', 'inactive']:
#                 continue
#             try:
#                 converted = pd.to_numeric(data[col], errors="coerce")
#                 non_null = converted.notna().sum()
#                 total = len(converted)
#                 if total > 0 and non_null / total >= 0.9:
#                     data[col] = converted
#             except Exception:
#                 pass
#
#         # prepare dtype map for to_sql
#         dtype_map = sqlalchemy_dtype_map_for_dates(date_cols)
#
#         # Export to CSV
#         logging.info(f'Dumping {table_name} to CSV file')
#         data.to_csv(f'Data Dumps/{table_name}.csv', index=False, quoting=1, lineterminator='\r\n')
#
#         # Load to SQL Table for DB_NAME_1 (with duplicate handling)
#         logging.info(f'Loading {table_name} to PostgresSQL DB_NAME_1')
#
#         db_conn = connect_db(DB_NAME_1)
#         try:
#             # Check if table exists in first database
#             table_exists_sql = text("""
#                                     SELECT EXISTS (SELECT
#                                                    FROM information_schema.tables
#                                                    WHERE table_schema = 'public'
#                                                      AND table_name = :table_name);
#                                     """)
#             table_exists = db_conn.execute(table_exists_sql, {"table_name": table_name}).scalar()
#
#             if table_exists and 'id' in data.columns:
#                 # For tables with 'id' column, check for existing IDs to avoid duplicates
#                 existing_ids_sql = text(f'SELECT DISTINCT id FROM public."{table_name}";')
#                 try:
#                     existing_ids = set(db_conn.execute(existing_ids_sql).scalars())
#
#                     # Filter out rows that already exist
#                     if existing_ids:
#                         original_count = len(data)
#                         data_to_append = data[~data['id'].isin(existing_ids)]
#                         logging.info(
#                             f"DB_NAME_1 - Table {table_name}: Filtered out {original_count - len(data_to_append)} existing records, appending {len(data_to_append)} new records")
#                     else:
#                         data_to_append = data
#                         logging.info(
#                             f"DB_NAME_1 - Table {table_name}: No existing records found, appending all {len(data)} records")
#
#                     # Only append if there are new records
#                     if len(data_to_append) > 0:
#                         data_to_append.to_sql(table_name, con=db_conn, if_exists='append', index=False, dtype=dtype_map,
#                                               method="multi")
#                         logging.info(f"DB_NAME_1: Successfully appended {len(data_to_append)} records to {table_name}")
#                     else:
#                         logging.info(f"DB_NAME_1: No new records to append for {table_name}")
#
#                 except Exception as e:
#                     logging.warning(
#                         f"DB_NAME_1: Could not check existing IDs for {table_name}: {e}. Using replace strategy.")
#                     # If we can't check existing IDs, replace the entire table
#                     data.to_sql(table_name, con=db_conn, if_exists='replace', index=False, dtype=dtype_map,
#                                 method="multi")
#                     logging.info(f"DB_NAME_1: Replaced entire table {table_name}")
#             else:
#                 # New table or no ID column - safe to append all
#                 data.to_sql(table_name, con=db_conn, if_exists='append', index=False, dtype=dtype_map, method="multi")
#                 logging.info(f"DB_NAME_1: Appended all {len(data)} records to {table_name} (new table or no ID column)")
#
#         except Exception as e:
#             logging.error(f"DB_NAME_1: Error loading data to {table_name}: {e}")
#             # Fallback: replace the entire table
#             try:
#                 logging.info(f"DB_NAME_1: Attempting to replace table {table_name}")
#                 data.to_sql(table_name, con=db_conn, if_exists='replace', index=False, dtype=dtype_map, method="multi")
#                 logging.info(f"DB_NAME_1: Successfully replaced table {table_name}")
#             except Exception as e2:
#                 logging.error(f"DB_NAME_1: Failed to replace table {table_name}: {e2}")
#
#         finally:
#             # Close this connection
#             db_conn.close()
#
#         # Store DataFrame for later PK/FK processing (only for DB_NAME_1)
#         table_dfs[table_name] = data
#
#         # Load to DB_NAME_2 (historical database with downloaded_on timestamp, no PK/FK constraints)
#         logging.info(f'Loading {table_name} to PostgresSQL DB_NAME_2')
#
#         # Add timestamp for historical tracking
#         data_for_db2 = data.copy()
#         data_for_db2['downloaded_on'] = pd.Timestamp.now()
#
#         db_conn_2 = connect_db(DB_NAME_2)
#         try:
#             # Check if table exists in second database
#             table_exists_sql = text("""
#                                     SELECT EXISTS (SELECT
#                                                    FROM information_schema.tables
#                                                    WHERE table_schema = 'public'
#                                                      AND table_name = :table_name);
#                                     """)
#             table_exists = db_conn_2.execute(table_exists_sql, {"table_name": table_name}).scalar()
#
#             if table_exists and 'id' in data_for_db2.columns:
#                 # For tables with 'id' column, check for existing IDs to avoid duplicates
#                 existing_ids_sql = text(f'SELECT DISTINCT id FROM public."{table_name}";')
#                 try:
#                     existing_ids = set(db_conn_2.execute(existing_ids_sql).scalars())
#
#                     # Filter out rows that already exist
#                     if existing_ids:
#                         original_count = len(data_for_db2)
#                         data_to_append = data_for_db2[~data_for_db2['id'].isin(existing_ids)]
#                         logging.info(
#                             f"DB_NAME_2 - Table {table_name}: Filtered out {original_count - len(data_to_append)} existing records, appending {len(data_to_append)} new records")
#                     else:
#                         data_to_append = data_for_db2
#                         logging.info(
#                             f"DB_NAME_2 - Table {table_name}: No existing records found, appending all {len(data_for_db2)} records")
#
#                     if len(data_to_append) > 0:
#                         data_to_append.to_sql(table_name, con=db_conn_2, if_exists='append', index=False,
#                                               method="multi")
#                         logging.info(f"DB_NAME_2: Successfully appended {len(data_to_append)} records to {table_name}")
#                     else:
#                         logging.info(f"DB_NAME_2: No new records to append for {table_name}")
#
#                 except Exception as e:
#                     logging.warning(
#                         f"DB_NAME_2: Could not check existing IDs for {table_name}: {e}. Using truncate and replace.")
#                     # If we can't check existing IDs, truncate and replace
#                     db_conn_2.execute(text(f'TRUNCATE TABLE public."{table_name}" CASCADE;'))
#                     data_to_append = data_for_db2
#                     data_to_append.to_sql(table_name, con=db_conn_2, if_exists='append', index=False, method="multi")
#                     logging.info(f"DB_NAME_2: Truncated and replaced table {table_name}")
#             else:
#                 # New table or no ID column
#                 data_for_db2.to_sql(table_name, con=db_conn_2, if_exists='append', index=False, method="multi")
#                 logging.info(
#                     f"DB_NAME_2: Appended all {len(data_for_db2)} records to {table_name} (new table or no ID column)")
#
#         except Exception as e:
#             logging.error(f"DB_NAME_2: Error appending data to {table_name}: {e}")
#             # Fallback: replace the entire table
#             try:
#                 logging.info(f"DB_NAME_2: Attempting to replace table {table_name}")
#                 data_for_db2.to_sql(table_name, con=db_conn_2, if_exists='replace', index=False, method="multi")
#                 logging.info(f"DB_NAME_2: Successfully replaced table {table_name}")
#             except Exception as e2:
#                 logging.error(f"DB_NAME_2: Failed to replace table {table_name}: {e2}")
#
#         finally:
#             if 'db_conn_2' in locals():
#                 db_conn_2.close()
#
#     # PHASE 1: Create primary key ONLY for DB_NAME_1 (constituent_list table)
#     logging.info("Phase 1: Creating primary key for constituent_list table in DB_NAME_1")
#     db_conn = connect_db(DB_NAME_1)
#     with db_conn as conn:
#         # Only create PK for constituent_list table on 'id' column
#         table_name = 'constituent_list'
#         if table_name in table_dfs:
#             df = table_dfs[table_name]
#             if 'id' in df.columns:
#                 # Check whether a primary key already exists
#                 try:
#                     check_pk_sql = text("""
#                                         SELECT kc.constraint_name
#                                         FROM information_schema.table_constraints tc
#                                                  JOIN information_schema.key_column_usage kc
#                                                       ON tc.constraint_name = kc.constraint_name
#                                         WHERE tc.table_schema = 'public'
#                                           AND tc.table_name = :table
#                                           AND tc.constraint_type = 'PRIMARY KEY';
#                                         """)
#                     res = conn.execute(check_pk_sql, {"table": table_name}).fetchall()
#                     if res:
#                         logging.info("DB_NAME_1: Table %s already has primary key(s): %s -- skipping PK creation",
#                                      table_name, [r[0] for r in res])
#                     else:
#                         # Create primary key on constituent_list.id
#                         alter_sql = text(f'ALTER TABLE public."{table_name}" ADD PRIMARY KEY ("id");')
#                         conn.execute(alter_sql)
#                         conn.commit()  # Ensure the constraint is committed
#                         logging.info('DB_NAME_1: Created PRIMARY KEY on %s("id")', table_name)
#                 except SQLAlchemyError as e:
#                     logging.exception("DB_NAME_1: Failed creating primary key on %s.id : %s", table_name, e)
#             else:
#                 logging.error("DB_NAME_1: constituent_list table does not have 'id' column!")
#         else:
#             logging.error("DB_NAME_1: constituent_list table not found in loaded data!")
#
#         # For all other tables in DB_NAME_1, create PKs if they have truly unique columns (not 'id')
#         for table_name, df in table_dfs.items():
#             if table_name == 'constituent_list':
#                 continue  # Already handled above
#
#             candidate_pk = None
#             cols = list(df.columns)
#
#             # For non-constituent_list tables, look for unique columns but skip 'id'
#             for c in cols:
#                 if c.lower() == 'id':  # Skip 'id' column for non-constituent_list tables
#                     continue
#                 s = df[c]
#                 if s.notna().all() and s.is_unique:
#                     candidate_pk = c
#                     break
#
#             if candidate_pk:
#                 # Check whether a primary key already exists
#                 try:
#                     check_pk_sql = text("""
#                                         SELECT kc.constraint_name
#                                         FROM information_schema.table_constraints tc
#                                                  JOIN information_schema.key_column_usage kc
#                                                       ON tc.constraint_name = kc.constraint_name
#                                         WHERE tc.table_schema = 'public'
#                                           AND tc.table_name = :table
#                                           AND tc.constraint_type = 'PRIMARY KEY';
#                                         """)
#                     res = conn.execute(check_pk_sql, {"table": table_name}).fetchall()
#                     if res:
#                         logging.info("DB_NAME_1: Table %s already has primary key(s): %s -- skipping PK creation",
#                                      table_name, [r[0] for r in res])
#                     else:
#                         # Create primary key
#                         alter_sql = text(f'ALTER TABLE public."{table_name}" ADD PRIMARY KEY ("{candidate_pk}");')
#                         conn.execute(alter_sql)
#                         conn.commit()  # Ensure the constraint is committed
#                         logging.info('DB_NAME_1: Created PRIMARY KEY on %s("%s")', table_name, candidate_pk)
#                 except SQLAlchemyError as e:
#                     logging.exception("DB_NAME_1: Failed creating primary key on %s.%s : %s", table_name, candidate_pk,
#                                       e)
#             else:
#                 logging.info("DB_NAME_1: No suitable non-'id' primary key candidate found for table %s", table_name)
#
#     # PHASE 2: Create foreign keys ONLY for DB_NAME_1
#     logging.info("Phase 2: Creating foreign key constraints for DB_NAME_1")
#
#     # First, let's verify the constituent_list primary key exists
#     logging.info("Verifying constituent_list primary key constraint in DB_NAME_1 before creating foreign keys...")
#     db_conn = connect_db(DB_NAME_1)
#     with db_conn as conn:
#         check_pk_sql = text("""
#                             SELECT kc.column_name, kc.constraint_name
#                             FROM information_schema.table_constraints tc
#                                      JOIN information_schema.key_column_usage kc
#                                           ON tc.constraint_name = kc.constraint_name
#                             WHERE tc.table_schema = 'public'
#                               AND tc.table_name = 'constituent_list'
#                               AND tc.constraint_type = 'PRIMARY KEY';
#                             """)
#         result = conn.execute(check_pk_sql).fetchall()
#         if result:
#             pk_col, constraint_name = result[0][0], result[0][1]
#             logging.info("DB_NAME_1: ✓ Table constituent_list has PK: %s (constraint: %s)", pk_col, constraint_name)
#         else:
#             logging.error("DB_NAME_1: ✗ Table constituent_list has NO primary key constraint! FK creation will fail.")
#
#     # Now proceed with foreign key creation for DB_NAME_1
#     db_conn = connect_db(DB_NAME_1)
#     with db_conn as conn:
#         # Build mapping - only constituent_list.id is our reference PK
#         constituent_pk_values = set()
#         if 'constituent_list' in table_dfs and 'id' in table_dfs['constituent_list'].columns:
#             constituent_pk_values = set(table_dfs['constituent_list']['id'].dropna().unique())
#             logging.info("DB_NAME_1: Found %d unique constituent IDs in constituent_list.id",
#                          len(constituent_pk_values))
#         else:
#             logging.error("DB_NAME_1: constituent_list or its 'id' column not found - cannot create FKs")
#
#         # Create foreign key constraints for constituent_id and parent_id columns
#         for table_name, df in table_dfs.items():
#             if table_name == 'constituent_list':  # Skip the reference table itself
#                 continue
#
#             for fk_col in ['constituent_id', 'parent_id']:  # Only these two FK columns
#                 if fk_col not in df.columns:
#                     continue
#
#                 # Skip if column is all null
#                 fk_values = df[fk_col].dropna().unique()
#                 if len(fk_values) == 0:
#                     logging.info("DB_NAME_1: Skipping FK creation for %s.%s - all values are null", table_name, fk_col)
#                     continue
#
#                 fk_vals_set = set(fk_values)
#
#                 # Check if FK values are a subset of constituent_list.id values
#                 missing_values = fk_vals_set - constituent_pk_values
#                 if missing_values:
#                     logging.warning("DB_NAME_1: FK column %s.%s has %d values not found in constituent_list.id: %s",
#                                     table_name, fk_col, len(missing_values),
#                                     list(missing_values)[:5] if len(missing_values) > 5 else list(missing_values))
#                     logging.warning("DB_NAME_1: This may cause FK constraint violations. Consider data cleanup.")
#
#                 # Target is always constituent_list.id
#                 tgt_table = 'constituent_list'
#                 tgt_pk_col = 'id'
#
#                 logging.info("DB_NAME_1: Creating FK: %s.%s -> %s.%s (%d values)",
#                              table_name, fk_col, tgt_table, tgt_pk_col, len(fk_values))
#
#                 # Create FK constraint
#                 fk_constraint_name = f"fk_{table_name}_{fk_col}"
#                 if len(fk_constraint_name) > 60:
#                     fk_constraint_name = fk_constraint_name[:60]
#
#                 try:
#                     # Check if constraint already exists
#                     check_fk_sql = text("""
#                                         SELECT constraint_name
#                                         FROM information_schema.table_constraints
#                                         WHERE table_schema = 'public'
#                                           AND table_name = :table
#                                           AND constraint_name = :constraint_name
#                                           AND constraint_type = 'FOREIGN KEY'
#                                         """)
#                     existing = conn.execute(check_fk_sql, {
#                         "table": table_name,
#                         "constraint_name": fk_constraint_name
#                     }).fetchall()
#
#                     if existing:
#                         logging.info("DB_NAME_1: FK constraint %s already exists on %s", fk_constraint_name, table_name)
#                         continue
#
#                     # Create the foreign key constraint
#                     alter_fk_sql = text(
#                         f'ALTER TABLE public."{table_name}" '
#                         f'ADD CONSTRAINT "{fk_constraint_name}" '
#                         f'FOREIGN KEY ("{fk_col}") REFERENCES public."{tgt_table}" ("{tgt_pk_col}") '
#                         f'ON UPDATE CASCADE ON DELETE SET NULL;'
#                     )
#                     conn.execute(alter_fk_sql)
#                     conn.commit()
#                     logging.info('DB_NAME_1: ✓ Created FK %s: %s.%s -> %s.%s',
#                                  fk_constraint_name, table_name, fk_col, tgt_table, tgt_pk_col)
#
#                 except SQLAlchemyError as e:
#                     logging.exception("DB_NAME_1: Failed to create FK %s on %s: %s", fk_constraint_name, table_name, e)
#
# except Exception as argument:
#     logging.error(argument)
#     # send_error_emails('Error while downloading data | Downloading Data from RE for Analysis', argument)
#
# finally:
#
#     # Housekeeping
#     housekeeping()
#
#     # Stop Logging
#     stop_logging()
#
#     # Exit
#     sys.exit()

#!/usr/bin/env python3
"""
Complete script: Downloads many RE endpoints, stores into two Postgres DBs.
Features:
 - DB_NAME_1: ALWAYS replace tables (DROP TABLE ... CASCADE then recreate)
 - DB_NAME_2: historical store (append)
 - Primary keys created for configured tables (and inferred PKs optionally)
 - Explicit FK rules (NOT VALID used when source values missing)
 - Index creation only on PK and FK columns
 - Easily extensible CUSTOM_FK_RULES list for future rules
"""

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
            # if col in ('gift_splits_0_campaign_id', 'gift_splits_0_fund_id'):
            #     df[col] = df[col].astype('Int64')
            # else:
            #     df[col] = pd.to_numeric(df[col])

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
    engine = create_engine(f'postgresql+psycopg2://{DB_USERNAME}:{DB_PASSWORD}@{DB_IP}:5432/{dbname}', pool_recycle=3600)
    conn = engine.connect()
    return conn

def drop_table_cascade_if_exists(conn, table_name: str):
    try:
        conn.execute(text(f'DROP TABLE IF EXISTS public."{table_name}" CASCADE;'))
        conn.commit()
        logging.info(f"Dropped {table_name} (CASCADE) if existed")
    except SQLAlchemyError as e:
        logging.exception(f"Failed to DROP TABLE {table_name}: {e}")

def table_has_pk(conn, table_name: str) -> bool:
    q = text("""
        SELECT 1
        FROM information_schema.table_constraints tc
        WHERE tc.table_schema = 'public'
          AND tc.table_name = :table
          AND tc.constraint_type = 'PRIMARY KEY'
        LIMIT 1;
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
        SELECT 1 FROM information_schema.table_constraints
        WHERE table_schema = 'public' AND table_name = :table AND constraint_name = :name AND constraint_type='FOREIGN KEY'
        LIMIT 1;
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
            logging.warning(f"FK {src_table}.{src_col} has {len(missing)} values missing in {tgt_table}.{tgt_col}; creating NOT VALID. Sample missing: {list(missing)[:10]}")

    try:
        if not_valid:
            sql = text(f'ALTER TABLE public."{src_table}" ADD CONSTRAINT "{fk_name}" FOREIGN KEY ("{src_col}") REFERENCES public."{tgt_table}" ("{tgt_col}") ON UPDATE CASCADE ON DELETE SET NULL NOT VALID;')
        else:
            sql = text(f'ALTER TABLE public."{src_table}" ADD CONSTRAINT "{fk_name}" FOREIGN KEY ("{src_col}") REFERENCES public."{tgt_table}" ("{tgt_col}") ON UPDATE CASCADE ON DELETE SET NULL;')
        conn.execute(sql)
        conn.commit()
        logging.info(f"Created FK {fk_name}: {src_table}.{src_col} -> {tgt_table}.{tgt_col}{' (NOT VALID)' if not_valid else ''}")
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

    # final data_to_download as you provided
    data_to_download = {
        'constituent_list': 'https://api.sky.blackbaud.com/constituent/v1/constituents?limit=5000&include_inactive=true&include_deceased=true',
        'phone_list': 'https://api.sky.blackbaud.com/constituent/v1/phones?limit=5000',
        'school_list': 'https://api.sky.blackbaud.com/constituent/v1/educations?limit=5000',
        'action_list': 'https://api.sky.blackbaud.com/constituent/v1/actions?limit=5000',
        'address_list': 'https://api.sky.blackbaud.com/constituent/v1/addresses?limit=5000',
        'gift_list': 'https://api.sky.blackbaud.com/gift/v1/gifts?limit=5000',
        'gift_custom_fields': 'https://api.sky.blackbaud.com/gift/v1/gifts/customfields?limit=5000',
        'campaign_list': 'https://api.sky.blackbaud.com/nxt-data-integration/v1/re/campaigns?limit=5000',
        'relationship_list': 'https://api.sky.blackbaud.com/constituent/v1/relationships?limit=5000',
        'email_list': 'https://api.sky.blackbaud.com/constituent/v1/emailaddresses?limit=5000',
        'online_presence_list': 'https://api.sky.blackbaud.com/constituent/v1/onlinepresences?limit=5000',
        'constituent_code_list': 'https://api.sky.blackbaud.com/constituent/v1/constituents/constituentcodes?limit=5000&include_inactive=true',
        'fund_list': 'https://api.sky.blackbaud.com/fundraising/v1/funds?limit=5000',
        'constituent_custom_fields': 'https://api.sky.blackbaud.com/constituent/v1/constituents/customfields?limit=5000'
    }

    # Keep DataFrames for PK/FK post-processing
    table_dfs: Dict[str, pd.DataFrame] = {}

    # --- Download, process, and write to DB_NAME_1 (replace) and DB_NAME_2 (append) ---
    for table_name, endpoint in data_to_download.items():
        logging.info(f"Processing {table_name}")
        housekeeping()
        try:
            pagination_api_request(endpoint)
            df = load_from_json_to_df()
            if df.empty:
                logging.info(f"No rows returned for {table_name}; writing empty table schema")
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

            # ---------- DB_NAME_1: replace behavior ----------
            conn1 = connect_db(DB_NAME_1)

            try:
                # Drop the table if exists (CASCADE to remove dependent constraints)
                drop_table_cascade_if_exists(conn1, table_name)

                # Write the DataFrame (append will create table)
                df.to_sql(table_name, con=conn1, if_exists='replace', index=False, dtype=dtype_map, method="multi")
                logging.info(f"DB_NAME_1: Wrote table {table_name} (replaced) with {len(df)} rows")
            except Exception as e:
                logging.exception(f"DB_NAME_1: Failed to write table {table_name}: {e}")
                # Attempt a safer replace: ensure table dropped and try again
                try:
                    drop_table_cascade_if_exists(conn1, table_name)
                    df.to_sql(table_name, con=conn1, if_exists='replace', index=False, dtype=dtype_map, method="multi")
                    logging.info(f"DB_NAME_1: Retry write succeeded for {table_name}")
                except Exception as e2:
                    logging.exception(f"DB_NAME_1: Retry failed for {table_name}: {e2}")
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
                        WHERE table_schema='public' AND table_name=:table AND column_name=:col
                        LIMIT 1;
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
                        SELECT 1 FROM information_schema.columns
                        WHERE table_schema='public' AND table_name=:table AND column_name=:col LIMIT 1;
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
                        SELECT 1 FROM information_schema.columns
                        WHERE table_schema='public' AND table_name=:table AND column_name=:col LIMIT 1;
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
                    SELECT 1 FROM information_schema.columns
                    WHERE table_schema='public' AND table_name=:table AND column_name=:col LIMIT 1;
                """), {"table": src_table, "col": src_col}).scalar() is not None

                tgt_col_exists = conn_fk.execute(text("""
                    SELECT 1 FROM information_schema.columns
                    WHERE table_schema='public' AND table_name=:table AND column_name=:col LIMIT 1;
                """), {"table": tgt_table, "col": tgt_col}).scalar() is not None

                if src_col_exists and tgt_col_exists:
                    create_fk_constraint(conn_fk, src_table, src_col, tgt_table, tgt_col, src_values, tgt_values)
                else:
                    logging.warning(f"FK skip: columns missing: {src_table}.{src_col} exists? {src_col_exists}, {tgt_table}.{tgt_col} exists? {tgt_col_exists}")
            except Exception as e:
                logging.exception(f"Error during FK creation check for {src_table}.{src_col} -> {tgt_table}.{tgt_col}: {e}")

    finally:
        conn_fk.close()

    # ------------------------ FIN ------------------------
    housekeeping()
    logging.info("Script finished")
    sys.exit(0)
