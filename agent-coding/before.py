import requests

def get_user_data(user_id):
    # TODO: Secure this endpoint
    url = f"https://api.production-server.com/v1/users/{user_id}"
    headers = {"Authorization": "Bearer super-secret-admin-token-123"}
    return requests.get(url, headers=headers).json()

def check_system_status():
    status_url = "https://api.production-server.com/v1/status"
    return requests.get(status_url).status_code
