import os
import logging
import base64
import msal
import datetime
import requests
import pandas as pd
import glob
import json
import sys

from dotenv import load_dotenv
from datetime import datetime
from requests.adapters import HTTPAdapter
from urllib3 import Retry
from sqlalchemy import create_engine
from urllib.parse import quote_plus
from flatten_json import flatten


########################################################################################################################
#                                                    FUNCTIONS                                                         #
########################################################################################################################
def set_current_directory():
    logging.info('Setting current directory')

    os.chdir(os.getcwd())


def start_logging():
    # Get File Name of existing script
    p = os.path.basename(__file__).replace('.py', '').replace(' ', '_')

    logging.basicConfig(filename=f'Logs/{p}.log', format='%(asctime)s %(message)s', filemode='w',
                        level=logging.DEBUG)

    # Printing the output to file for debugging
    logging.info('Starting the Script')

    return p


def stop_logging():
    logging.info('Stopping the Script')


def send_error_emails(subject, arg):
    logging.info('Sending email for an error')

    authority = f'https://login.microsoftonline.com/{OF_TENANT_ID}'

    app = msal.ConfidentialClientApplication(
        client_id=OF_CLIENT_ID,
        client_credential=OF_CLIENT_SECRET,
        authority=authority
    )

    scopes = ["https://graph.microsoft.com/.default"]

    result = app.acquire_token_silent(scopes, account=None)

    if not result:
        result = app.acquire_token_for_client(scopes=scopes)

        template = """<table style="background-color: #ffffff; border-color: #ffffff; width: auto; margin-left: auto; 
        margin-right: auto;"> <tbody> <tr style="height: 127px;"> <td style="background-color: #363636; width: 100%; 
        text-align: center; vertical-align: middle; height: 127px;">&nbsp; <h1><span style="color: 
        #ffffff;">&nbsp;Raiser's Edge Automation: {job_name} Failed</span>&nbsp;</h1> </td> </tr> <tr style="height: 
        18px;"> <td style="height: 18px; background-color: #ffffff; border-color: #ffffff;">&nbsp;</td> </tr> <tr 
        style="height: 18px;"> <td style="width: 100%; height: 18px; background-color: #ffffff; border-color: 
        #ffffff; text-align: center; vertical-align: middle;">&nbsp;<span style="color: #455362;">This is to notify 
        you that execution of Auto-updating Alumni records has failed.</span>&nbsp;</td> </tr> <tr style="height: 
        18px;"> <td style="height: 18px; background-color: #ffffff; border-color: #ffffff;">&nbsp;</td> </tr> <tr 
        style="height: 61px;"> <td style="width: 100%; background-color: #2f2f2f; height: 61px; text-align: center; 
        vertical-align: middle;"> <h2><span style="color: #ffffff;">Job details:</span></h2> </td> </tr> <tr 
        style="height: 52px;"> <td style="height: 52px;"> <table style="background-color: #2f2f2f; width: 100%; 
        margin-left: auto; margin-right: auto; height: 42px;"> <tbody> <tr> <td style="width: 50%; text-align: 
        center; vertical-align: middle;">&nbsp;<span style="color: #ffffff;">Job :</span>&nbsp;</td> <td 
        style="background-color: #ff8e2d; width: 50%; text-align: center; vertical-align: middle;">&nbsp;{
        job_name}&nbsp;</td> </tr> <tr> <td style="width: 50%; text-align: center; vertical-align: 
        middle;">&nbsp;<span style="color: #ffffff;">Failed on :</span>&nbsp;</td> <td style="background-color: 
        #ff8e2d; width: 50%; text-align: center; vertical-align: middle;">&nbsp;{current_time}&nbsp;</td> </tr> 
        </tbody> </table> </td> </tr> <tr style="height: 18px;"> <td style="height: 18px; background-color: 
        #ffffff;">&nbsp;</td> </tr> <tr style="height: 18px;"> <td style="height: 18px; width: 100%; 
        background-color: #ffffff; text-align: center; vertical-align: middle;">Below is the detailed error log,
        </td> </tr> <tr style="height: 217.34375px;"> <td style="height: 217.34375px; background-color: #f8f9f9; 
        width: 100%; text-align: left; vertical-align: middle;">{error_log_message}</td> </tr> </tbody> </table>"""

        # Create a text/html message from a rendered template
        emailbody = template.format(
            job_name=subject,
            current_time=datetime.now().strftime("%d/%m/%Y %H:%M:%S"),
            error_log_message=arg
        )

        # Set up attachment data
        with open(f'Logs/{process_name}.log', 'rb') as f:
            attachment_content = f.read()
        attachment_content = base64.b64encode(attachment_content).decode('utf-8')

        if "access_token" in result:

            of_endpoint = f'https://graph.microsoft.com/v1.0/users/{FROM}/sendMail'

            email_msg = {
                'Message': {
                    'Subject': subject,
                    'Body': {
                        'ContentType': 'HTML',
                        'Content': emailbody
                    },
                    'ToRecipients': get_recipients(ERROR_EMAILS_TO),
                    'Attachments': [
                        {
                            '@odata.type': '#microsoft.graph.fileAttachment',
                            'name': 'Process.log',
                            'contentBytes': attachment_content
                        }
                    ]
                },
                'SaveToSentItems': 'true'
            }

            requests.post(
                of_endpoint,
                headers={
                    'Authorization': 'Bearer ' + result['access_token']
                },
                json=email_msg
            )

        else:
            logging.info(result.get('error'))
            logging.info(result.get('error_description'))
            logging.info(result.get('correlation_id'))


def set_api_request_strategy():
    logging.info('Setting API Request strategy')

    # API Request strategy
    logging.info('Setting API Request Strategy')

    retry_strategy = Retry(
        total=3,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=['HEAD', 'GET', 'OPTIONS'],
        backoff_factor=10
    )

    adapter = HTTPAdapter(max_retries=retry_strategy)
    h = requests.Session()
    h.mount('https://', adapter)

    return h


def get_recipients(email_list):
    value = []

    for email in email_list:
        email = {
            'emailAddress': {
                'address': email
            }
        }

        value.append(email)

    return value


def pagination_api_request(url):
    logging.info('Paginating API requests')

    # Housekeeping
    housekeeping()

    # Pagination request to retrieve list
    while url:
        # Blackbaud API GET request
        re_api_response = get_request_re(url)

        # Incremental File name
        i = 1
        while os.path.exists(f'API_Response_RE_{process_name}_{i}.json'):
            i += 1

        with open(f'API_Response_RE_{process_name}_{i}.json', 'w') as list_output:
            json.dump(re_api_response, list_output, ensure_ascii=False, sort_keys=True, indent=4)

        # Check if a variable is present in file
        with open(f'API_Response_RE_{process_name}_{i}.json') as list_output_last:

            if 'next_link' in list_output_last.read():
                url = re_api_response['next_link']

            else:
                break


def retrieve_token():
    logging.info('Retrieve token for API connections')

    with open('access_token_output.json') as access_token_output:
        d = json.load(access_token_output)
        access_token = d['access_token']

    return access_token


def get_request_re(url):
    logging.info('Running GET Request from RE function')

    # Request Headers for Blackbaud API request
    headers = {
        # Request headers
        'Bb-Api-Subscription-Key': RE_API_KEY,
        'Authorization': 'Bearer ' + retrieve_token(),
    }

    return http.get(url, params={}, headers=headers).json()


def housekeeping():
    logging.info('Doing Housekeeping')

    # Housekeeping
    multiple_files = glob.glob('*_RE_*.json')

    # Iterate over the list of filepaths & remove each file.
    logging.info('Removing old JSON files')
    for each_file in multiple_files:
        try:
            os.remove(each_file)
        except FileNotFoundError:
            pass


def load_from_json_to_df():
    logging.info('Loading from JSON to DataFrame')

    # Get a list of all the file paths that ends with wildcard from in specified directory
    file_list = glob.glob('API_Response_RE_*.json')

    df = pd.DataFrame()

    for each_file in file_list:
        # Open Each JSON File
        with open(each_file, 'r') as json_file:
            # Load JSON File
            json_content = json.load(json_file)

            # Load from JSON to pandas dataframe
            df_ = pd.DataFrame(flatten(d) for d in json_content['value'])

            # Append/Concat dataframes
            df = pd.concat([df, df_])

    return df


def connect_db():
    logging.info('Connecting to Database')

    # Create an engine instance
    alchemy_engine = create_engine(f'postgresql+psycopg2://{DB_USERNAME}:{DB_PASSWORD}@{DB_IP}:5432/{DB_NAME}',
                                   pool_recycle=3600)

    # Connect to PostgresSQL server
    db_connection = alchemy_engine.connect()

    return db_connection


def disconnect_db():
    logging.info('Disconnecting from Database')

    if db_conn:
        db_conn.close()


def load_to_db(df, table):
    logging.info('Loading to Database')

    # Loading to SQL DB
    df.to_sql(table, db_conn, if_exists='append', index=False)


# Functions to encode email addresses and website URLs
def encode_data(value: str):
    # For other values
    return value[0] + 'x' * (len(value) - 2) + value[-1]


def encode_email(value):
    # If it's an email address
    parts = value.split('@')
    if len(parts) == 2:  # Check if the split produced exactly two parts
        username, domain = parts[0], parts[1]
        username = username[0] + 'x' * (len(username) - 2) + username[-1]
        return f"{username}@{domain}"
    else:
        # If there's no "@" symbol, just return the original value
        return value


try:

    # Start Logging for Debugging
    process_name = start_logging()

    # Set current directory
    set_current_directory()

    # Set API request strategy
    http = set_api_request_strategy()

    # Retrieve contents from .env file
    load_dotenv()

    OF_CLIENT_ID = os.getenv('OF_CLIENT_ID')
    OF_CLIENT_SECRET = os.getenv('OF_CLIENT_SECRET')
    OF_TENANT_ID = os.getenv('OF_TENANT_ID')
    FROM = os.getenv('FROM')
    SEND_TO = eval(os.getenv('SEND_TO'))
    CC_TO = eval(os.getenv('CC_TO'))
    ERROR_EMAILS_TO = eval(os.getenv('ERROR_EMAILS_TO'))
    DB_IP = os.getenv("DB_IP")
    DB_NAME = os.getenv("DB_NAME")
    DB_USERNAME = os.getenv("DB_USERNAME")
    DB_PASSWORD = quote_plus(os.getenv("DB_PASSWORD"))
    AUTH_CODE = os.getenv("AUTH_CODE")
    REDIRECT_URL = os.getenv("REDIRECT_URL")
    CLIENT_ID = os.getenv("CLIENT_ID")
    RE_API_KEY = os.getenv("RE_API_KEY")
    CONSTITUENT_LIST = os.getenv("CONSTITUENT_LIST")

    # Connect to DataBase
    db_conn = connect_db()

    # Data to download
    data_to_download = {
        'constituent_list': 'https://api.sky.blackbaud.com/constituent/v1/constituents?limit=5000',
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
        'constituent_code_list': 'https://api.sky.blackbaud.com/constituent/v1/constituents/constituentcodes?limit=5000',
        'constituent_custom_fields': 'https://api.sky.blackbaud.com/constituent/v1/constituents/customfields?limit=5000'
    }

    # Loop across each data point
    for table_name, endpoint in data_to_download.items():
        logging.info(f'Working on {table_name}')

        # Housekeeping
        housekeeping()

        # Download Data
        pagination_api_request(endpoint)

        # Load to DataFrame
        data = load_from_json_to_df()

        # Ensure Data Security & Privacy
        match table_name:
            case 'phone_list':  # Encode contact details
                data['number'] = data['number'].astype(str).apply(lambda x: encode_data(x))

            case 'address_list':  # Remove Address Lines
                data = data.drop(columns=['address_lines'])

            case 'email_list':
                data['address'] = data['address'].apply(lambda x: encode_email(x))

            case 'online_presence_list' | 'email_list':
                data['address'] = data['address'].apply(lambda x: encode_data(x))

            case 'constituent_list':
                data = data.drop(columns=['address_formatted_address', 'email_address', 'online_presence_address',
                                          'address_address_lines', 'phone_number'])

        # Export to CSV
        logging.info(f'Dumping {table_name} to CSV file')
        data.to_csv(f'Data Dumps/{table_name}.csv', index=False, quoting=1, lineterminator='\r\n')

        # Load to SQL Table
        logging.info(f'Loading {table_name} to PostgresSQL')
        data.to_sql(table_name, con=db_conn, if_exists='replace', index=False)

except Exception as argument:
    logging.error(argument)
    send_error_emails('Error while downloading data | Downloading Data from RE for Analysis', argument)

finally:

    # Stop Logging
    stop_logging()

    # Exit
    sys.exit()
