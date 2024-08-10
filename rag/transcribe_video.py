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
        # video = YouTube(youtubelink)
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
        'outtmpl': 'audio',
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
