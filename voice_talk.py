import requests
import os
import pyaudio
import wave
import keyboard
import faster_whisper
import torch
import pygame

# Konstanta untuk script
CHUNK_SIZE = 1024  # Ukuran chunk untuk dibaca/ditulis pada setiap waktu
XI_API_KEY = "sk_edc265442891c54353092293473e2ff279c89477993c5f3e"  # Ganti dengan API key ElevenLabs Anda
VOICE_ID = "MF3mGyEYCl7XYWbV9V6O"  # Ganti dengan ID suara yang Anda inginkan
GEMINI_API_KEY = "AIzaSyBLLo6MrFeC8HvPondbNdovTjah7r7HogA"  # Ganti dengan API key Gemini Anda

# Set lingkungan
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# System prompt
system_prompt = {
    'role': 'system',
    'content': 'Anda adalah ElavenAI, yang di ciptakan oleh seseorang yang bernama Egal , Anda adalah model kecerdasan buatan yang sangat powerful'
}

# Load model Whisper untuk transkripsi
model = faster_whisper.WhisperModel(
    model_size_or_path="tiny.en",
    device='cuda' if torch.cuda.is_available() else 'cpu'
)

# Inisialisasi Pygame
pygame.mixer.init()

# Cache untuk hasil teks dan audio
cache_responses = {}

def generate_gemini(messages):
    """Menghasilkan respon menggunakan Gemini API, dengan cache."""
    global answer
    answer = ""
    
    # Cache key berdasarkan pesan pengguna
    cache_key = "\n".join([msg['content'] for msg in messages])
    
    # Cek di cache terlebih dahulu
    if cache_key in cache_responses:
        return cache_responses[cache_key]

    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent?key={GEMINI_API_KEY}"
    headers = {
        "Content-Type": "application/json"
    }

    payload = {
        "contents": [{
            "parts": [{
                "text": cache_key
            }]
        }]
    }

    response = requests.post(url, json=payload, headers=headers)
    
    if response.status_code == 200:
        result = response.json()
        
        if 'candidates' in result and len(result['candidates']) > 0:
            if 'content' in result['candidates'][0]:
                parts = result['candidates'][0]['content']['parts']
                answer = "".join([part['text'] for part in parts])
                cache_responses[cache_key] = answer  # Simpan ke cache
            else:
                answer = "Error: 'content' not available."
        else:
            answer = "Error: No valid candidates."
    else:
        answer = f"Error: {response.status_code} - {response.text}"

    return answer

def play_audio(file_path):
    """Memutar audio dari file."""
    if not pygame.mixer.get_init():
        print("Pygame mixer tidak berhasil diinisialisasi.")
        return

    pygame.mixer.music.load(file_path)
    pygame.mixer.music.play()
    
    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(10)

def generate_speech_with_elevenlabs(text):
    """Menghasilkan suara menggunakan ElevenLabs API."""
    # Cache berdasarkan teks yang dihasilkan
    if text in cache_responses:
        print("Menggunakan audio dari cache")
        play_audio(cache_responses[text])
        return
    
    # Cek panjang teks
    if len(text) > 10000:
        print("Teks terlalu panjang. Silakan kurangi teks menjadi kurang dari 10.000 karakter.")
        return
    
    tts_url = f"https://api.elevenlabs.io/v1/text-to-speech/{VOICE_ID}/stream"
    headers = {
        "Accept": "application/json",
        "xi-api-key": XI_API_KEY
    }
    data = {
        "text": text,
        "model_id": "eleven_multilingual_v2",
        "voice_settings": {
            "stability": 0.5,
            "similarity_boost": 0.75
        }
    }

    response = requests.post(tts_url, headers=headers, json=data, stream=True)

    if response.ok:
        try:
            file_path = f"audio_{hash(text)}.mp3"
            with open(file_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=1024):
                    f.write(chunk)
            print(f"Audio disimpan ke {file_path}")
            
            # Simpan hasil audio ke cache
            cache_responses[text] = file_path
            play_audio(file_path)
        except PermissionError as e:
            print(f"PermissionError: {e}. Coba tutup aplikasi lain yang mungkin menggunakan file tersebut.")
        except Exception as e:
            print(f"Terjadi kesalahan saat menyimpan audio: {e}")
    else:
        print(f"Error: {response.status_code} - {response.text}")
        # Jika kuota habis, beri tahu pengguna
        if response.status_code == 401:
            print("Permintaan gagal: Kuota Anda habis. Silakan periksa akun Anda untuk detail kuota.")

# Menggunakan generate_gemini untuk memastikan panjang teks sebelum mengonversi ke suara
def handle_user_input(user_input):
    generated_text = generate_gemini([system_prompt, {'role': 'user', 'content': user_input}])
    print(f"Respon Asisten: {generated_text}")
    generate_speech_with_elevenlabs(generated_text)

def record_audio():
    """Rekam audio dari mikrofon hingga pengguna menekan spasi."""
    print("Saya siap mendengarkan. Tekan spasi saat Anda selesai.\n")
    audio = pyaudio.PyAudio()
    frames = []
    py_stream = audio.open(rate=16000, format=pyaudio.paInt16, channels=1, input=True, frames_per_buffer=512)
    
    while not keyboard.is_pressed('space'):
        frames.append(py_stream.read(512))
    py_stream.stop_stream()
    py_stream.close()
    audio.terminate()

    # Simpan audio yang direkam ke file WAV
    with wave.open("voice_record.wav", 'wb') as wf:
        wf.setparams((1, audio.get_sample_size(pyaudio.paInt16), 16000, 0, 'NONE', 'NONE'))
        wf.writeframes(b''.join(frames))

    return "voice_record.wav"

def transcribe_audio(audio_file):
    """Transkripsi audio menggunakan model Whisper."""
    segments, _ = model.transcribe(audio_file, language="en")
    return " ".join([seg.text for seg in segments])

# Loop utama untuk mendapatkan input dari pengguna dan memutar audio
while True:
    user_input = input("Masukkan teks yang ingin ditanyakan (atau ketik 'exit' untuk keluar): ")
    
    if user_input.lower() == 'exit':
        break

    handle_user_input(user_input)

# Loop utama untuk merekam dan menghasilkan respon
while True:
    # Tunggu hingga pengguna menekan spasi untuk mulai merekam
    print("\n\nTekan spasi saat Anda siap. ", end="", flush=True)
    keyboard.wait('space')
    while keyboard.is_pressed('space'):
        pass

    # Rekam audio
    audio_file = record_audio()

    # Transkripsi rekaman menggunakan Whisper
    user_text = transcribe_audio(audio_file)
    print(f'>>>{user_text}\n<<< ', end="", flush=True)

    # Hasilkan respon menggunakan API Gemini
    generated_text = generate_gemini([system_prompt, {'role': 'user', 'content': user_text}])
    print(f"Respon Asisten: {generated_text}")

    # Ubah respon Gemini menjadi suara menggunakan ElevenLabs
    generate_speech_with_elevenlabs(generated_text)
