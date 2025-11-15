import os
import logging
import json
import sys
from typing import List, Dict

import requests
from requests.adapters import HTTPAdapter
from urllib3 import Retry
from dotenv import load_dotenv


# ------------------------------- Helper functions -------------------------------

def start_logging():
    """Initialize logging for the script"""
    p = os.path.basename(__file__).replace('.py', '').replace(' ', '_')
    os.makedirs('Logs', exist_ok=True)
    logging.basicConfig(filename=f'Logs/{p}.log',
                        format='%(asctime)s %(levelname)s %(message)s',
                        filemode='w',
                        level=logging.INFO)
    logging.info("Starting script")
    return p


def set_api_request_strategy():
    """Set up HTTP session with retry strategy"""
    retry_strategy = Retry(
        total=3,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=['HEAD', 'GET', 'OPTIONS', 'DELETE'],
        backoff_factor=3
    )
    s = requests.Session()
    adapter = HTTPAdapter(max_retries=retry_strategy)
    s.mount('https://', adapter)
    return s


def retrieve_token():
    """Retrieve access token from the token file"""
    try:
        with open('access_token_output.json') as fh:
            d = json.load(fh)
        return d.get('access_token')
    except FileNotFoundError:
        logging.critical("access_token_output.json not found. Please generate a token.")
        sys.exit(1)


def delete_request_re(url: str, re_api_key: str, http_session: requests.Session) -> bool:
    """
    Make a DELETE request to the Raiser's Edge API

    Args:
        url: The API endpoint URL to delete
        re_api_key: The RE API subscription key
        http_session: The requests session object

    Returns:
        bool: True if deletion was successful, False otherwise
    """
    token = retrieve_token()
    if not token:
        logging.critical("Failed to retrieve API token.")
        return False

    headers = {
        'Bb-Api-Subscription-Key': re_api_key,
        'Authorization': 'Bearer ' + token
    }

    try:
        response = http_session.delete(url, headers=headers)
        response.raise_for_status()
        logging.info(f"Successfully deleted: {url}")
        return True
    except requests.exceptions.HTTPError as e:
        logging.error(f"HTTP error occurred while deleting {url}: {e}")
        if e.response is not None:
            logging.error(f"Status code: {e.response.status_code}")
            logging.error(f"Response body: {e.response.text}")
        return False
    except requests.exceptions.RequestException as e:
        logging.error(f"Request failed for DELETE {url}: {e}")
        return False


def delete_consent_by_id(consent_id: str, re_api_key: str, http_session: requests.Session) -> bool:
    """
    Delete a single consent record by ID

    Args:
        consent_id: The consent ID to delete
        re_api_key: The RE API subscription key
        http_session: The requests session object

    Returns:
        bool: True if deletion was successful, False otherwise
    """
    url = f"https://api.sky.blackbaud.com/consent/constituents/consents/{consent_id}"
    logging.info(f"Attempting to delete consent ID: {consent_id}")
    return delete_request_re(url, re_api_key, http_session)


def delete_consent_batch(consent_ids: List[str], re_api_key: str, http_session: requests.Session) -> Dict[str, bool]:
    """
    Delete multiple consent records

    Args:
        consent_ids: List of consent IDs to delete
        re_api_key: The RE API subscription key
        http_session: The requests session object

    Returns:
        Dict[str, bool]: Dictionary mapping consent_id to deletion success status
    """
    results = {}
    total = len(consent_ids)

    logging.info(f"Starting batch deletion of {total} consent records")

    for idx, consent_id in enumerate(consent_ids, 1):
        logging.info(f"Processing {idx}/{total}: Consent ID {consent_id}")
        success = delete_consent_by_id(consent_id, re_api_key, http_session)
        results[consent_id] = success

    successful = sum(1 for v in results.values() if v)
    failed = total - successful

    logging.info(f"Batch deletion completed: {successful} successful, {failed} failed")

    return results


def delete_consent_from_file(file_path: str, re_api_key: str, http_session: requests.Session) -> Dict[str, bool]:
    """
    Delete consent records from a file containing consent IDs (one per line)

    Args:
        file_path: Path to file containing consent IDs
        re_api_key: The RE API subscription key
        http_session: The requests session object

    Returns:
        Dict[str, bool]: Dictionary mapping consent_id to deletion success status
    """
    try:
        with open(file_path, 'r') as f:
            consent_ids = [line.strip() for line in f if line.strip()]

        logging.info(f"Loaded {len(consent_ids)} consent IDs from {file_path}")
        return delete_consent_batch(consent_ids, re_api_key, http_session)

    except FileNotFoundError:
        logging.error(f"File not found: {file_path}")
        return {}
    except Exception as e:
        logging.exception(f"Error reading file {file_path}: {e}")
        return {}


# ------------------------------- Main script -------------------------------

if __name__ == "__main__":
    process_name = start_logging()
    http = set_api_request_strategy()
    load_dotenv()

    RE_API_KEY = os.getenv("RE_API_KEY")

    if not RE_API_KEY:
        logging.critical("RE_API_KEY not found in environment variables")
        sys.exit(1)

    # Example usage - uncomment and modify as needed:

    # Option 1: Delete a single consent by ID
    # consent_id = 13566  # Replace with actual consent ID
    # success = delete_consent_by_id(consent_id, RE_API_KEY, http)
    # if success:
    #     print(f"Successfully deleted consent {consent_id}")
    # else:
    #     print(f"Failed to delete consent {consent_id}")

    # Option 2: Delete multiple consents by IDs
    consent_ids = []  # Replace with actual consent IDs
    results = delete_consent_batch(consent_ids, RE_API_KEY, http)
    print(f"Deletion results: {results}")

    # Option 3: Delete consents from a file (one ID per line)
    # results = delete_consent_from_file("consent_ids.txt", RE_API_KEY, http)
    # print(f"Deleted {sum(1 for v in results.values() if v)} out of {len(results)} consents")

    logging.info("Script execution completed")
    print("Script completed. Check logs for details.")

    logging.info("--- Script finished ---")
