"""
Alexa-style Voice Assistant
----------------------------
Speech Recognition : OpenAI Whisper (PyTorch-based)
Brain / Answers     : Google Gemini API
Text-to-Speech      : pyttsx3
GUI                 : CustomTkinter

Setup:
    pip install -r requirements.txt

Set your Gemini API key as an environment variable before running:
    Windows (PowerShell):  $env:GEMINI_API_KEY="your_key_here"
    macOS/Linux:            export GEMINI_API_KEY="your_key_here"

Run:
    python voice_assistant.py
"""

import os
import threading
import tempfile
import queue
import time

import numpy as np
import sounddevice as sd
import soundfile as sf
import torch
import whisper
import pyttsx3
import customtkinter as ctk
import google.generativeai as genai


# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
ASSISTANT_NAME = "Alexa"
WHISPER_MODEL_SIZE = "base"        # tiny / base / small / medium / large
RECORD_SECONDS = 5                 # length of each recording
SAMPLE_RATE = 16000
GEMINI_MODEL = "gemini-2.0-flash"

SYSTEM_PROMPT = f"""You are {ASSISTANT_NAME}, a voice assistant that behaves exactly like Amazon Alexa.
Rules you must always follow:
- Answer briefly and conversationally, the way a smart speaker would (1-3 sentences max).
- Never use markdown, bullet points, asterisks, or code blocks — you are SPOKEN aloud.
- If you don't know something or it needs real-time data you don't have, say so briefly,
  the way Alexa says "Sorry, I don't know that one."
- Do not say "As an AI language model". Just answer like a helpful voice assistant.
"""


# ---------------------------------------------------------------------------
# BACKEND: Whisper (speech-to-text) + Gemini (brain) + pyttsx3 (speech-out)
# ---------------------------------------------------------------------------
class AssistantBackend:
    def __init__(self, status_callback=None):
        self.status_callback = status_callback or (lambda msg: None)

        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GEMINI_API_KEY environment variable not set. "
                "Set it before running the app."
            )
        genai.configure(api_key=api_key)
        self.chat_model = genai.GenerativeModel(
            model_name=GEMINI_MODEL,
            system_instruction=SYSTEM_PROMPT,
        )
        self.chat_session = self.chat_model.start_chat(history=[])

        self._set_status("Loading Whisper model (PyTorch)...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.whisper_model = whisper.load_model(WHISPER_MODEL_SIZE, device=self.device)

        self.tts_engine = pyttsx3.init()
        self.tts_engine.setProperty("rate", 175)
        # Try to pick a female voice, closer to Alexa's default voice
        for voice in self.tts_engine.getProperty("voices"):
            if "female" in voice.name.lower() or "zira" in voice.name.lower():
                self.tts_engine.setProperty("voice", voice.id)
                break

        self._set_status("Ready")

    def _set_status(self, msg: str):
        self.status_callback(msg)

    # -------------------- Recording --------------------
    def record_audio(self, seconds=RECORD_SECONDS) -> str:
        """Records from the default mic and returns a path to a temp wav file."""
        self._set_status("Listening...")
        audio = sd.rec(
            int(seconds * SAMPLE_RATE),
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
        )
        sd.wait()
        tmp_path = os.path.join(tempfile.gettempdir(), "va_input.wav")
        sf.write(tmp_path, audio, SAMPLE_RATE)
        return tmp_path

    # -------------------- Speech to text --------------------
    def transcribe(self, wav_path: str) -> str:
        self._set_status("Transcribing...")
        result = self.whisper_model.transcribe(wav_path, fp16=(self.device == "cuda"))
        return result.get("text", "").strip()

    # -------------------- Gemini answer --------------------
    def ask_gemini(self, user_text: str) -> str:
        self._set_status("Thinking...")
        response = self.chat_session.send_message(user_text)
        return response.text.strip()

    # -------------------- Speak --------------------
    def speak(self, text: str):
        self._set_status("Speaking...")
        self.tts_engine.say(text)
        self.tts_engine.runAndWait()
        self._set_status("Ready")

    # -------------------- Full pipeline --------------------
    def listen_and_respond(self, seconds=RECORD_SECONDS):
        """Returns (user_text, assistant_text). Speaks the answer aloud too."""
        wav_path = self.record_audio(seconds)
        user_text = self.transcribe(wav_path)

        if not user_text:
            reply = "Sorry, I didn't catch that."
            self.speak(reply)
            return "", reply

        reply = self.ask_gemini(user_text)
        self.speak(reply)
        return user_text, reply


# ---------------------------------------------------------------------------
# GUI: CustomTkinter
# ---------------------------------------------------------------------------
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")


class VoiceAssistantApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title(f"{ASSISTANT_NAME} Voice Assistant")
        self.geometry("480x640")
        self.resizable(False, False)

        self.busy = False
        self.backend = None

        # ---------- Header ----------
        self.title_label = ctk.CTkLabel(
            self, text=ASSISTANT_NAME, font=ctk.CTkFont(size=32, weight="bold")
        )
        self.title_label.pack(pady=(30, 5))

        self.status_label = ctk.CTkLabel(
            self, text="Starting up...", font=ctk.CTkFont(size=14), text_color="#9dd6ff"
        )
        self.status_label.pack(pady=(0, 20))

        # ---------- Glowing mic circle ----------
        self.canvas = ctk.CTkCanvas(self, width=180, height=180, highlightthickness=0, bg="#1a1a1a")
        self.canvas.pack(pady=10)
        self.circle = self.canvas.create_oval(20, 20, 160, 160, fill="#1f6aa5", outline="")
        self.canvas.create_text(90, 90, text="🎙", font=("Arial", 48), fill="white")

        # ---------- Conversation log ----------
        self.log_box = ctk.CTkTextbox(self, width=420, height=280, font=ctk.CTkFont(size=13))
        self.log_box.pack(pady=20)
        self.log_box.configure(state="disabled")

        # ---------- Mic button ----------
        self.mic_button = ctk.CTkButton(
            self,
            text="Tap to Talk",
            command=self.on_mic_pressed,
            width=200,
            height=45,
            font=ctk.CTkFont(size=16, weight="bold"),
        )
        self.mic_button.pack(pady=10)

        # Load backend (Whisper + Gemini) in a background thread so the UI opens instantly
        threading.Thread(target=self.init_backend, daemon=True).start()

    # ------------------------------------------------------------------
    def init_backend(self):
        try:
            self.backend = AssistantBackend(status_callback=self.set_status)
        except Exception as e:
            self.set_status(f"Error: {e}")
            self.append_log(f"[Startup error] {e}")

    def set_status(self, msg: str):
        self.status_label.after(0, lambda: self.status_label.configure(text=msg))
        pulse = msg.lower() in ("listening...", "speaking...")
        self.after(0, lambda: self.animate_circle(pulse))

    def animate_circle(self, active: bool):
        color = "#e04b4b" if active else "#1f6aa5"
        self.canvas.itemconfig(self.circle, fill=color)

    def append_log(self, text: str):
        self.log_box.configure(state="normal")
        self.log_box.insert("end", text + "\n\n")
        self.log_box.see("end")
        self.log_box.configure(state="disabled")

    # ------------------------------------------------------------------
    def on_mic_pressed(self):
        if self.busy:
            return
        if self.backend is None:
            self.append_log("[Assistant is still starting up, please wait]")
            return

        self.busy = True
        self.mic_button.configure(state="disabled", text="Listening...")
        threading.Thread(target=self.run_conversation_turn, daemon=True).start()

    def run_conversation_turn(self):
        try:
            user_text, reply = self.backend.listen_and_respond()
            if user_text:
                self.append_log(f"You: {user_text}")
            self.append_log(f"{ASSISTANT_NAME}: {reply}")
        except Exception as e:
            self.append_log(f"[Error] {e}")
        finally:
            self.busy = False
            self.mic_button.after(
                0, lambda: self.mic_button.configure(state="normal", text="Tap to Talk")
            )


if __name__ == "__main__":
    app = VoiceAssistantApp()
    app.mainloop()
