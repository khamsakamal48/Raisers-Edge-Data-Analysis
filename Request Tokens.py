import json
import requests
import os

from dotenv import load_dotenv

load_dotenv()


def set_directory():
    os.chdir(os.getcwd())


def get_token():
    # Blackbaud Token URL
    url = 'https://oauth2.sky.blackbaud.com/token'

    url_for_user = (f'https://oauth2.sky.blackbaud.com/authorization?client_id={CLIENT_ID}&response_type=code'
                    f'&redirect_uri={REDIRECT_URL}&state=fdf80155')

    print("Please go to this link to get your access code " + url_for_user)

    access_code = input("Enter Access Code: ")

    # Request Headers for Blackbaud API request
    headers = {
        # Request headers
        'Content-Type': 'application/x-www-form-urlencoded',
        'Authorization': 'Basic ' + AUTH_CODE
    }

    # Request parameters for Blackbaud API request
    data = {
        'grant_type': 'authorization_code',
        'redirect_uri': REDIRECT_URL,
        'code': access_code
    }

    # API Request
    response = requests.post(url, data=data, headers=headers).json()

    # Write output to JSON file
    with open("access_token_output.json", "w") as response_output:
        json.dump(response, response_output, ensure_ascii=False, sort_keys=True, indent=4)


try:

    # Set current directory
    set_directory()

    # Retrieve contents from .env file
    # Get authorization code by encoding client_id:client_secret at https://www.base64encode.org
    AUTH_CODE = os.getenv("AUTH_CODE")
    REDIRECT_URL = os.getenv("REDIRECT_URL")
    CLIENT_ID = os.getenv("CLIENT_ID")

    # Blackbaud Token URL
    get_token()

except Exception as Argument:
    print(Argument)
    pass
