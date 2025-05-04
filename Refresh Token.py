import json
import requests
import os
import logging
import sys
import msal
import base64

from dotenv import load_dotenv
from requests.adapters import HTTPAdapter
from urllib3 import Retry
from datetime import datetime

def api_request_strategy():

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


def set_directory():
    os.chdir(os.getcwd())


def retrieve_refresh_token():
    with open('access_token_output.json') as access_token_output:
        data = json.load(access_token_output)
        refresh_token = data["refresh_token"]

    return refresh_token


def get_token():
    url = 'https://oauth2.sky.blackbaud.com/token'

    # Request Headers for Blackbaud API request
    headers = {
        'Content-Type': 'application/x-www-form-urlencoded',
        'Authorization': 'Basic ' + AUTH_CODE
    }

    # Request parameters for Blackbaud API request
    data = {
        'grant_type': 'refresh_token',
        'refresh_token': retrieve_refresh_token()
    }

    # API Request
    response = http.post(url, data=data, headers=headers).json()

    # Write output to JSON file
    with open('access_token_output.json', 'w') as response_output:
        json.dump(response, response_output, ensure_ascii=False, sort_keys=True, indent=4)


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


def send_error_emails(subject):
    logging.info('Sending email for an error')

    authority = f'https://login.microsoftonline.com/{TENANT_ID}'

    app = msal.ConfidentialClientApplication(
        client_id=O_CLIENT_ID,
        client_credential=CLIENT_SECRET,
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
        email_body = template.format(
            job_name=subject,
            current_time=datetime.now().strftime("%d/%m/%Y %H:%M:%S"),
            error_log_message=Argument
        )

        # Set up attachment data
        with open(f'Logs/{process_name}.log', 'rb') as f:
            attachment_content = f.read()
        attachment_content = base64.b64encode(attachment_content).decode('utf-8')

        if "access_token" in result:

            endpoint = f'https://graph.microsoft.com/v1.0/users/{FROM}/sendMail'

            email_msg = {
                'Message': {
                    'Subject': subject,
                    'Body': {
                        'ContentType': 'HTML',
                        'Content': email_body
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
                endpoint,
                headers={
                    'Authorization': 'Bearer ' + result['access_token']
                },
                json=email_msg
            )

        else:
            logging.info(result.get('error'))
            logging.info(result.get('error_description'))
            logging.info(result.get('correlation_id'))


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


try:

    # Set current directory
    set_directory()

    # Start Logging
    process_name = start_logging()

    # Set API Request strategy
    http = api_request_strategy()

    # Load env variables
    load_dotenv()
    AUTH_CODE = os.getenv('AUTH_CODE')
    # OF_CLIENT_ID = os.getenv('O_CLIENT_ID')
    # OF_CLIENT_SECRET = os.getenv('CLIENT_SECRET')
    # OF_TENANT_ID = os.getenv('TENANT_ID')
    # FROM = os.getenv('FROM')
    # SEND_TO = eval(os.getenv('SEND_TO'))
    # CC_TO = eval(os.getenv('CC_TO'))
    # ERROR_EMAILS_TO = eval(os.getenv('ERROR_EMAILS_TO'))

    # Blackbaud Token URL
    get_token()

except Exception as Argument:
    logging.error(Argument)
    # send_error_emails('Error while refreshing token | Downloading Data from RE for Analysis')

finally:

    # Stop Logging
    stop_logging()

    # Exit
    sys.exit()
