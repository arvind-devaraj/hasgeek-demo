import requests
import json

# Load configuration from config.json
with open('config.json') as config_file:
    config = json.load(config_file)


def get_user_data(user_id):
    url = f"{config['API_URL']}/v1/users/{user_id}"
    headers = {"Authorization": f"Bearer {config['AUTH_TOKEN']}"}
    return requests.get(url, headers=headers).json()


def check_system_status():
    status_url = f"{config['API_URL']}/v1/status"
    return requests.get(status_url).status_code