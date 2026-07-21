import requests

config = {
    "api_url": "https://api.production-server.com/v1",
    "auth_token": "super-secret-admin-token-123"
}

def get_user_data(user_id):
    url = f"{config['api_url']}/users/{user_id}"
    headers = { "Authorization": f"Bearer {config['auth_token']}" }
    return requests.get(url, headers=headers).json()

def check_system_status():
    status_url = f"{config['api_url']}/status"
    return requests.get(status_url).status_code
