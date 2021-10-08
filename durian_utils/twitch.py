'''
CREATED BY: Chris Gardner
CREATED DATE: 2021-08-11
LAST UPDATED: 2021-08-11

This API wrapper will be used to collect and downlowad streams/clips from different game titles.

For more Details:
https://dev.twitch.tv/docs/api/
'''

import requests
import streamlink
import tempfile
import os
from . import utils

class twitch_api(object):
    def __init__(self, client_id, client_secret):
        self.client_id = client_id
        self.client_secret = client_secret
        self.access_token = self.get_access_token(client_id, client_secret)
        self.headers = {
            "Authorization": "Bearer {}".format(self.access_token),
            "Client-Id": self.client_id
        }
        self.tempdir = tempfile.TemporaryDirectory()
        print("twitch_api object initialized")
        print("Temporary Directory set up at: {}".format(self.tempdir))
    
    def __del__(self):
        self.tempdir.cleanup()
        print("Twitch api destroyed. Temp Dir cleaned up.")
    
    def create_tempdir(self):
        self.tempdir = tempfile.TemporaryDirectory()

    def cleanup_tempdir(self):
        self.tempdir.cleanup()

    def get_access_token(self, client_id, client_secret):
        '''
        Functions to get access_token from twitch api
        '''
        data = {
            "client_id": client_id,
            "client_secret": client_secret,
            "grant_type": "client_credentials"
        }
        url = 'https://id.twitch.tv/oauth2/token'
        try:
            response = requests.post(url, data)
        except Exception as e:
            print("Trouble Authentication: ", e)
        else:
            return response.json()['access_token']
    
    def get_user_vods(self, username, period, vod_type, debug=False):
        '''
        Collects metadata for vods from a username for a certain period.
        Args:
            username: streamers username (ex: scottynada, durianil)
            period: period from which to gather vods (day, week, month, all)
            vod_type: highlight, upload, archive
        '''
        user_url = "https://api.twitch.tv/helix/users"
        user_response = requests.get(user_url, headers=self.headers, params={"login": username})
        if debug:
            print(user_response.json())
        user_id = user_response.json()['data'][0]['id']

        params = {
                "user_id": user_id,
                "period": period,
                "after": "",
                "type": vod_type
        }

        user_vid_url = "https://api.twitch.tv/helix/videos"
        vid_response = requests.get(
            user_vid_url, 
            headers=self.headers, 
            params=params
        )
        if debug:
            print(vid_response.json())
        r = vid_response.json()
        data = r['data']
        while r.get('pagination'):
            params['after'] = r.get('pagination').get('cursor')
            response = requests.get(user_vid_url, headers=self.headers, params=params)
            r = response.json()
            data += r['data']

        print("Finished Gathering Vods")
        return data

    def get_user_clips(self, user_name, start, end, debug=False):
        '''
        Collects clips from a specified timeframe and downloads them to a specific folder.

        Args:
            user_name: plaintext str of the user (ex: durianirl or scottynada)
            start: start date of period to fetch clips (YYYY-MM-DDTHH:mm:ssZ)
            end: end date of period to fetch clips (YYYY-MM-DDTHH:mm:ssZ)
        '''
        #get game ID
        url = "https://api.twitch.tv/helix/users"
        params = {
            "login": user_name
        }
        response = requests.get(url, headers=self.headers, params=params)
        user_id = response.json()['data'][0]['id']
        if debug:
            print(response.json())
        print("User ID Fetched")

        #Fetch metadata for clips withing timeframe
        params = {
            "broadcaster_id": user_id,
            "started_at": start,
            "ended_at": end,
            "after": ''
        }
        url = "https://api.twitch.tv/helix/clips"
        response = requests.get(url, headers=self.headers, params=params)
        r = response.json()
        if debug:
            print(r)
        
        data = r['data']
        while r.get('pagination'):
            params['after'] = r.get('pagination').get('cursor')
            response = requests.get(url, headers=self.headers, params=params)
            r = response.json()
            data += r['data']

        print("Finished Gathering Clips")
        return data


    def get_game_clips(self, game_name, start, end, debug=False):
        '''
        Collects clips from a specified timeframe and downloads them to a specific folder.

        Args:
            game_name: plaintext str of the game title (ex: Clash Royale or TEPPEN or Fortnite)
            start: start date of period to fetch clips (YYYY-MM-DDTHH:mm:ssZ)
            end: end date of period to fetch clips (YYYY-MM-DDTHH:mm:ssZ)
        '''
        #get game ID
        url = "https://api.twitch.tv/helix/games"
        params = {
            "name": game_name
        }
        response = requests.get(url, headers=self.headers, params=params)
        game_id = response.json()['data'][0]['id']
        if debug:
            print(response.json())
        print("Game ID Fetched")

        #Fetch metadata for clips withing timeframe
        params = {
            "game_id": game_id,
            "started_at": start,
            "ended_at": end,
            "after": ''
        }
        url = "https://api.twitch.tv/helix/clips"
        response = requests.get(url, headers=self.headers, params=params)
        r = response.json()
        if debug:
            print(r)
        
        data = r['data']
        while r.get('pagination'):
            params['after'] = r.get('pagination').get('cursor')
            response = requests.get(url, headers=self.headers, params=params)
            r = response.json()
            data += r['data']

        print("Finished Gathering Clips")
        return data
    
    def download_clip(self, clip:dict, debug=False):
        '''
        downloads url to a file in the temporary directory
        Args:
            clip: metadata of a clip (dict)
        '''
        return utils.download_video_url(self.tempdir.name, clip['id'], clip['url'], debug=False)
