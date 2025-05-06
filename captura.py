import yt_dlp as youtube_dl
import datetime

x = datetime.datetime.now()
x = x.strftime("%Y%m%d%H%M%S")

url="https://youtu.be/JhRyLKfmACM" 
ydl_opts={'format': 'bestvideo[height=480][ext=mp4]+bestaudio[ext=m4a]/best[height=480]',
    'outtmpl': f'video/video{x}.mp4',
    'merge_output_format': 'mp4'}
ydl=youtube_dl.YoutubeDL(ydl_opts)
info_dict=ydl.extract_info(url, download=True)

#To stop the download to press Ctrl+C or Ctrl+Z in terminal
