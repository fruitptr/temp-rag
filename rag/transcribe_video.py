import os
import json
from dotenv import load_dotenv
# from pytubefix import YouTube
from openai import OpenAI
import yt_dlp

load_dotenv()

def transcribe_video(youtubelink):
    print("Video to audio...")
    # Convert Youtube video to audio
    try:
        # video = YouTube(youtubelink, use_oauth=True, allow_oauth_cache=True)
        # print("video object", video)
        # print("video streams", video.streams)
        # stream = video.streams.filter(only_audio=True).first()
        # print("stream object", stream)
        # stream.download(filename="audio.mp3")
        # print("Audio File downloaded in MP3")

        ydl_opts = {
            'format': 'bestaudio/best',
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
                'preferredquality': '192',
            }],
            'outtmpl': 'audio.mp3',
            'ratelimit': '100K',  # Limit download speed to 100KB/s
            'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3',  # Set a common user-agent
        }
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([youtubelink])
            print("Audio downloaded successfully as audio.mp3")

        client = OpenAI()

        with open("./audio.mp3", "rb") as audio_file:
            transcript = client.audio.transcriptions.create(
                model="whisper-1", 
                file=audio_file
            )

        print(transcript.text)
        return transcript.text
    except Exception as e:
        raise Exception(f"Error transcribing video: {str(e)}")
    finally:
        # Remove the audio file
        if os.path.exists("audio.mp3"):
            os.remove("audio.mp3")
            print("Audio file removed.")
