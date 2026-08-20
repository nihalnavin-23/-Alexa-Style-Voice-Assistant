# 🎤 Alexa-Style Voice Assistant

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Platform](https://img.shields.io/badge/Platform-Windows%20%7C%20macOS%20%7C%20Linux-lightgrey.svg)
![AI](https://img.shields.io/badge/AI-Gemini%20%2B%20Whisper-orange.svg)
![Status](https://img.shields.io/badge/Status-Active-brightgreen.svg)

**A sophisticated voice assistant that mimics Amazon Alexa's behavior, powered by cutting-edge AI technologies**

</div>

---

## ✨ Features

### 🎯 Core Capabilities
- **Real-time Voice Recognition**: Utilizes OpenAI's Whisper model with PyTorch for accurate speech-to-text conversion
- **Intelligent Conversations**: Powered by Google's Gemini API for natural, context-aware responses
- **Natural Text-to-Speech**: Uses pyttsx3 for clear, natural-sounding voice output
- **Alexa-Like Personality**: Behaves exactly like Amazon's Alexa with warm, friendly responses

### 🎨 User Interface
- **Modern Dark Theme**: Sleek, professional design with CustomTkinter
- **Animated Visual Feedback**: Dynamic circular indicator that changes color based on assistant state
- **Real-time Status Updates**: Visual indicators for listening, thinking, and speaking states
- **Conversation History**: Scrollable log of all interactions
- **Quick Command Buttons**: One-click access to common queries

### 🧠 Smart Features
- **Silence Detection**: Automatically stops recording when you finish speaking
- **Built-in Commands**: Handles time, date, jokes, weather queries, and more
- **Mathematical Calculations**: Can perform basic arithmetic
- **Context Memory**: Maintains conversation history for natural follow-up questions
- **Multi-Model Fallback**: Automatically detects and uses available Gemini models
- **Offline Mode**: Works with basic commands even without internet connection

---

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- A working microphone
- Internet connection for Gemini API

### Installation

1. **Clone the repository:**
```bash
git clone https://github.com/YOUR_USERNAME/alexa-voice-assistant.git
cd alexa-voice-assistant
