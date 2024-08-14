import os
import json
from dotenv import load_dotenv
from youtube_transcript_api import YouTubeTranscriptApi
# from pytubefix import YouTube
# from openai import OpenAI
# import yt_dlp
import re

# load_dotenv()

def get_youtube_id(url):
    # Regular expression to extract the video ID
    match = re.search(r'(?<=v=)[^&]+', url)
    if match:
        return match.group(0)
    return None

def transcribe_video(youtubelink):
    # print("Video to audio...")
    # # Convert Youtube video to audio
    # try:
    #     # video = YouTube(youtubelink, use_oauth=True, allow_oauth_cache=True)
    #     # print("video object", video)
    #     # print("video streams", video.streams)
    #     # stream = video.streams.filter(only_audio=True).first()
    #     # print("stream object", stream)
    #     # stream.download(filename="audio.mp3")
    #     # print("Audio File downloaded in MP3")

    #     ydl_opts = {
    #     'format': 'bestaudio/best',
    #     'postprocessors': [{
    #         'key': 'FFmpegExtractAudio',
    #         'preferredcodec': 'mp3',
    #         'preferredquality': '192',
    #     }],
    #     'outtmpl': 'audio',
    #     }
        
    #     with yt_dlp.YoutubeDL(ydl_opts) as ydl:
    #         ydl.download([youtubelink])
    #         print("Audio downloaded successfully as audio.mp3")

    #     client = OpenAI()

    #     with open("./audio.mp3", "rb") as audio_file:
    #         transcript = client.audio.transcriptions.create(
    #             model="whisper-1", 
    #             file=audio_file
    #         )

    #     print(transcript.text)
    #     return transcript.text
    # except Exception as e:
    #     raise Exception(f"Error transcribing video: {str(e)}")
    # finally:
    #     # Remove the audio file
    #     if os.path.exists("audio.mp3"):
    #         os.remove("audio.mp3")
    #         print("Audio file removed.")
    id = get_youtube_id(youtubelink)
    data = YouTubeTranscriptApi.get_transcript(id)
# list = YouTubeTranscriptApi.list_transcripts(id)

    transcript = ''

    for value in data:
        for key,val in value.items():
            if key == 'text':
                transcript += " "
                transcript += val

    temp = transcript.splitlines()
    final_transcript = ' '.join(temp)
    print("Transcript: ", final_transcript)
    return final_transcript