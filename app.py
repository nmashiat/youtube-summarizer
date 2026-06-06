from flask import Flask, request, jsonify, render_template
from youtube_transcript_api import YouTubeTranscriptApi
from anthropic import Anthropic
import whisper
import yt_dlp
import tempfile
import os
import re
import json

app = Flask(__name__)
client = Anthropic(api_key="YOUR_API_KEY_HERE")

def get_video_id(url):
    patterns = [
        r'[?&]v=([a-zA-Z0-9_-]{11})',
        r'youtu\.be/([a-zA-Z0-9_-]{11})',
        r'shorts/([a-zA-Z0-9_-]{11})',
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None

def get_transcript(url, video_id):
    try:
        ytt = YouTubeTranscriptApi()
        transcript_data = ytt.fetch(video_id)
        transcript = ' '.join([entry.text for entry in transcript_data])
        return transcript, "captions"
    except Exception:
        return None, None

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            ydl_opts = {
                'format': 'bestaudio/best',
                'outtmpl': f'{tmpdir}/audio.%(ext)s',
                'postprocessors': [{'key': 'FFmpegExtractAudio', 'preferredcodec': 'mp3'}],
                'quiet': True
            }
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([f'https://www.youtube.com/watch?v={video_id}'])
            model = whisper.load_model("base")
            result = model.transcribe(f'{tmpdir}/audio.mp3')
            return result["text"], "whisper"
    except Exception as e:
        return None, None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/summarize', methods=['POST'])
def summarize():
    data = request.json
    url = data.get('url', '')
    length = data.get('length', 'brief')
    fmt = data.get('format', 'paragraph')

    video_id = get_video_id(url)
    if not video_id:
        return jsonify({'error': 'Invalid YouTube URL'}), 400

    transcript, source = get_transcript(url, video_id)
    if not transcript:
        return jsonify({'error': 'Could not fetch transcript'}), 400

    prompt = f"""Summarize this YouTube transcript.
Length: {length}
Format: {'bullet points' if fmt == 'bullets' else 'paragraph'}

Respond in JSON only (no markdown):
{{
  "summary": "...",
  "key_points": ["...", "...", "...", "...", "..."],
  "topics": ["...", "...", "..."],
  "sentiment": "positive | neutral | educational | entertaining | mixed"
}}

TRANSCRIPT:
{transcript[:8000]}"""

    response = client.messages.create(
        model="claude-opus-4-5",
        max_tokens=1000,
        messages=[{"role": "user", "content": prompt}]
    )

    raw = response.content[0].text.replace('```json','').replace('```','').strip()
    result = json.loads(raw)
    result['video_id'] = video_id
    result['source'] = source
    return jsonify(result)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(debug=False, host='0.0.0.0', port=port)