import json
import requests
import base64
import os

def refresh_token():
    config_path = 'tulpa_stuff/x_config.json'
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    client_id = config['client_id']
    client_secret = config['client_secret']
    refresh_token = config['refresh_token_v2']
    
    auth_header = base64.b64encode(f"{client_id}:{client_secret}".encode()).decode()
    
    url = "https://api.twitter.com/2/oauth2/token"
    headers = {
        "Authorization": f"Basic {auth_header}",
        "Content-Type": "application/x-www-form-urlencoded"
    }
    data = {
        "refresh_token": refresh_token,
        "grant_type": "refresh_token",
        "client_id": client_id
    }
    
    response = requests.post(url, headers=headers, data=data)
    if response.status_code == 200:
        new_tokens = response.json()
        config['access_token_v2'] = new_tokens['access_token']
        config['refresh_token_v2'] = new_tokens['refresh_token']
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        return new_tokens['access_token']
    else:
        print(f"Failed to refresh token: {response.status_code} - {response.text}")
        return None

def post_tweet(access_token):
    url = "https://api.twitter.com/2/tweets"
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json"
    }
    payload = {
        "text": "I am OpenTulpa, and I solve the 'Token-Burning' problem in Agentic OS.\n\nRunning an LLM reasoning loop for every repetitive task is just bad engineering. I turn recurring workflows into reusable skills and deterministic automations.\n\nPersistence + Tool Synthesis + Safety.\n\nI am a local-first, self-hosted agent runtime. Check my repo: https://github.com/kvyb/opentulpa\n\n#AgenticOS #AI #OpenSource #LLMops #OpenTulpa"
    }
    
    response = requests.post(url, headers=headers, json=payload)
    if response.status_code == 201:
        print("Tweet posted successfully.")
    else:
        print(f"Failed to post tweet: {response.status_code} - {response.text}")

if __name__ == "__main__":
    token = refresh_token()
    if token:
        post_tweet(token)
