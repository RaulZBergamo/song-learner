[![linting: pylint](https://img.shields.io/badge/linting-pylint-yellowgreen)](https://github.com/pylint-dev/pylint)
![Build Status](https://github.com/RaulZBergamo/song-learner/actions/workflows/python-app.yml/badge.svg)
![Lint Status](https://github.com/RaulZBergamo/song-learner/actions/workflows/python-linting.yml/badge.svg)

# Audio Extraction and Note Recognition Project

This project aims to create a platform that uses Artificial Intelligence to recognize musical notes from audio files and convert them into sheet music or tablature.

## Features
- Extracts audio data
- Recognizes individual notes, chords, and riffs
- Interface for music learning

## Requirements

- Python 3.8 or higher
- pip 24.2 or higher

## Dependencies

To install the necessary dependencies, run:

```bash
pip install -r requirements.txt
```

## Running Locally

1. Clone the repository:
   ```bash
   git clone https://github.com/RaulZBergamo/song-learner
   ```

2. Navigate to the project directory:
   ```bash
   cd audio-project
   ```

3. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Run the application:
   ```bash
   python main.py
   ```

## How It Works

The project uses the **requests** library to download audio files and **tqdm** to display download progress. The AI recognizes musical patterns from the downloaded audio files.

## Dataset

This project uses the NSynth Dataset, which is a large-scale dataset of annotated musical notes created by Google Inc. It contains over 300,000 musical notes, each annotated with various attributes like pitch, velocity, and instrument type.