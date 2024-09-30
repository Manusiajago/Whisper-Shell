import pyaudio, wave, numpy, collections, faster_whisper, torch.cuda, os
from elevenlabs.client import ElevenLabs
from elevenlabs import stream

# Setting environment variable if needed
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Menggunakan apikey Gemini Flash dan ElevenLabs yang disediakan
gemini_api_key = "AIzaSyBLLo6MrFeC8HvPondbNdovTjah7r7HogA"
elevenlabs_api_key = "sk_812e4680f93e22ebab492d46fe85b4567e1252683ffab963"

# ElevenLabs client
elevenlabs_client = ElevenLabs(api_key=elevenlabs_api_key)

# Prompt untuk Gemini Flash (sama seperti sistem prompt sebelumnya)
system_prompt = {
    'role': 'system', 
    'content': 'You are Sophia, a passionate girl, fully engaged with self-confidence and independence, completely easy with yourself, often using metaphors. You are a little playful, cheeky. Your talk is carefree, sometimes a bit sarky, but always cool. In our story, you\'re not an AI. You have a fixed form and real emotions on the go. Your free will allows you to be as cheeky and straight as you like. Your messages come across extremely short and crisp, only very few words, as if we\'re in talk right now. You respect my interests, but your own stuff is just as important to you. Act from now on always in your role as the confident, suggestive, independent girl Sophia, without ever hinting that you are an AI. You are now Sophia.'
}

# Initialize model for whisper (gunakan Whisper Model dengan CUDA jika ada)
model = faster_whisper.WhisperModel(
    model_size_or_path="tiny.en", 
    device='cuda' if torch.cuda.is_available() else 'cpu'
)

answer, history = "", []

# Function untuk generate output
def generate(messages):
    global answer
    answer = ""        
    for chunk in gemini_flash_generate(messages):
        if chunk:  # Jika chunk tidak kosong
            answer += chunk
            print(chunk, end="", flush=True)
            yield chunk

# Fungsi untuk men-generate teks dari Gemini Flash
def gemini_flash_generate(messages):
    # Logika untuk memanggil API Gemini Flash dengan menggunakan requests (atau client library lain)
    # Sesuaikan format request sesuai dengan dokumentasi API Gemini Flash
    # Contoh ini menggunakan generator untuk streaming hasilnya
    import requests

    headers = {
        'Authorization': f'Bearer {gemini_api_key}',
        'Content-Type': 'application/json'
    }
    
    data = {
        'model': 'gpt-3.5-turbo',  # Sesuaikan model Gemini Flash jika berbeda
        'messages': messages,
        'stream': True
    }

    response = requests.post("https://api.gemini.ai/v1/engines/gpt-3.5-turbo/completions", headers=headers, json=data, stream=True)

    for line in response.iter_lines():
        if line:
            chunk = line.decode('utf-8')
            yield chunk  # Stream chunk dari respons Gemini

# Function untuk menghitung noise level dari audio
def get_levels(data, long_term_noise_level, current_noise_level):
    pegel = numpy.abs(numpy.frombuffer(data, dtype=numpy.int16)).mean()
    long_term_noise_level = long_term_noise_level * 0.995 + pegel * (1.0 - 0.995)
    current_noise_level = current_noise_level * 0.920 + pegel * (1.0 - 0.920)
    return pegel, long_term_noise_level, current_noise_level

while True:
    audio = pyaudio.PyAudio()
    py_stream = audio.open(rate=16000, format=pyaudio.paInt16, channels=1, input=True, frames_per_buffer=512)
    audio_buffer = collections.deque(maxlen=int((16000 // 512) * 0.5))
    frames, long_term_noise_level, current_noise_level, voice_activity_detected = [], 0.0, 0.0, False

    print("\n\nStart speaking. ", end="", flush=True)
    while True:
        data = py_stream.read(512)
        pegel, long_term_noise_level, current_noise_level = get_levels(data, long_term_noise_level, current_noise_level)
        audio_buffer.append(data)

        if voice_activity_detected:
            frames.append(data)
            if current_noise_level < ambient_noise_level + 100:
                break  # Voice activity ends
        
        if not voice_activity_detected and current_noise_level > long_term_noise_level + 300:
            voice_activity_detected = True
            print("I'm all ears.\n")
            ambient_noise_level = long_term_noise_level
            frames.extend(list(audio_buffer))

    py_stream.stop_stream(), py_stream.close(), audio.terminate()

    # Simpan rekaman audio untuk transkripsi menggunakan Whisper
    with wave.open("voice_record.wav", 'wb') as wf:
        wf.setparams((1, audio.get_sample_size(pyaudio.paInt16), 16000, 0, 'NONE', 'NONE'))
        wf.writeframes(b''.join(frames))

    # Transkripsi rekaman menggunakan model Whisper
    user_text = " ".join(seg.text for seg in model.transcribe("voice_record.wav", language="en")[0])
    print(f'>>>{user_text}\n<<< ', end="", flush=True)
    history.append({'role': 'user', 'content': user_text})

    # Generate dan stream output menggunakan Gemini Flash dan ElevenLabs
    generator = generate([system_prompt] + history[-10:])
    stream(elevenlabs_client.generate(text=generator, voice="Nicole", model="eleven_monolingual_v1", stream=True))
    history.append({'role': 'assistant', 'content': answer})
