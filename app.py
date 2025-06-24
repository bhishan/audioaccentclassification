import streamlit as st
import torch
import torchaudio
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2FeatureExtractor
import librosa
import os
import shutil
import uuid
import subprocess
import imageio_ffmpeg
import logging

# config, not wanting to overcomplicate things, i am therefore not using a .env or config file
DEFAULT_MODEL_ID = "dima806/english_accents_classification"
DEFAULT_SAMPLE_RATE = 16000 # becasue wav2vec2 is trained with audio sampled at 16kHz
DEFAULT_WINDOW_SIZE = 10 # use 10 seconds of content for the features to make classification, i want to emulate the way humans identify accent in sections
LABEL_MAP = {0: 'us', 1: 'england', 2: 'indian', 3: 'australia', 4: 'canada'} # the model is trained on these classes
# create logs directory if it doesn't exist
os.makedirs("logs", exist_ok=True)

# Use a fixed log file name
log_file_path = "logs/accent_classification.log"
# logging to keep record and especially for debugging purposes
# Configure global logging
logging.basicConfig(
    filename=log_file_path,
    level=logging.INFO,
    format="%(asctime)s — %(levelname)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    filemode="a"  # Append to existing log file
)


ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe() # using this because i don't want to offload from python env and install another system level dependency of ffmpeg


@st.cache_resource
def load_model(model_id):
    """
    Loads any compatible pretrained model and its corresponding feature extractor 
    from Hugging Face for audio classification tasks. This is so that the underlying model is pluggable in nature.

    This function is designed to work with models like Wav2Vec2 (or similar) that 
    use a feature extractor and output class logits. As long as the model and 
    extractor are compatible (i.e., from the same checkpoint), they can be used 
    interchangeably.

    Args:
        model_id (str): The Hugging Face model identifier. Must point to a model 
                        that supports audio classification and has an associated 
                        feature extractor (default is 'dima806/english_accents_classification').

    Returns:
        Tuple[PreTrainedModel, PreTrainedFeatureExtractor]:
            A tuple containing the loaded model and its corresponding feature extractor.
    """
    model = Wav2Vec2ForSequenceClassification.from_pretrained(model_id)
    extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_id)
    return model, extractor


def process_video_url(video_url, video_path):
    """
    Downloads a video from a public URL and returns paths.
    Handles errors and optionally cleans up temp files.
    Using yt-dlp for this via subprocess.
    
    Args:
        video_url (str): Public video URL (YouTube, Loom, direct MP4)
        video_path (str): Path to save downloaded video
        audio_path (str): Path to save extracted audio

    Returns:
        str | None: Path to audio file if successful, None otherwise
    """
    try:
        print(f"Downloading video: {video_url}")
        subprocess.run(["yt-dlp", "-o", video_path, video_url], check=True)
        logging.info(f"Downloaded video: {video_url}")
        logging.info(f"Downloaded video path: {video_path}")
        return video_path
    except Exception as e:
        print("Error processing video:", e)
        logging.error(f"Video download failed for {video_url}: {str(e)}")
        return None



def extract_audio_from_video(video_path, audio_path, sample_rate):
    """
    Extracts audio from a video file and saves it as mono 16kHz WAV using imageio_ffmpeg.

    Args:
        video_path (str): Path to the input video file.
        audio_path (str): Path to save the extracted audio file.
        sample_rate (int): Desired audio sample rate (default 16000 Hz).

    Returns:
        str | None: Path to the extracted audio file if successful, None otherwise.
    """
    try:
        ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
        # using imageio_ffmpeg because i don't want to offload and install ffmpeg from apt or some system repo
        command = [
            ffmpeg_path,
            "-i", video_path,
            "-ar", str(sample_rate),
            "-ac", "1",
            "-f", "wav",
            audio_path,
            "-y"
        ]

        print(f"Extracting audio to: {audio_path}")
        subprocess.run(command, check=True)
        logging.info(f"Extracted audio path: {audio_path}")
        return audio_path

    except subprocess.CalledProcessError as e:
        print("FFmpeg failed:", e)
        logging.error(f"Audio extraction failed for video path: {video_path}: {str(e)}")
        return None

    except Exception as e:
        print("Unexpected error:", e)
        return None

def classify_accent_proba(waveform_chunk, model, feature_extractor):
    """
    Computes the softmax probabilities over accent classes or labels 
    for a given chunk of audio waveform using a pretrained audio classification model.

    This function is model-agnostic and works with any Hugging Face-compatible 
    sequence classification model and its corresponding feature extractor.

    Args:
        waveform_chunk (torch.Tensor): A mono waveform tensor of shape (1, samples), 
        typically sampled at 16kHz.
        model (PreTrainedModel): A pretrained Hugging Face model for sequence classification
        feature_extractor (PreTrainedFeatureExtractor): The corresponding feature extractor for the given model.

    Returns:
        torch.Tensor: A 1D tensor containing softmax probabilities for each class 
                      (e.g., accent categories).
    """
    inputs = feature_extractor(waveform_chunk.squeeze().numpy(), sampling_rate=16000, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.nn.functional.softmax(logits, dim=-1)[0]
    return probs


def classify_accent(audio_path, model, feature_extractor, label_map, window_size):
    """
    Classifies the accent in an audio file by splitting it into time-based chunks
    and averaging the model predictions across all chunks. The idea is to emulate the human approach
    to classification of accent by taking smaller chunks of the speech and finally aggregating
    individual probabilities

    This function processes the entire audio file and reports:
      - Predicted accent (based on average probabilities)
      - Confidence of prediction
      - Max confidence reached across any chunk
      - Average of max confidences across all chunks

    Args:
        audio_path (str): Path to the .wav audio file.
        model (PreTrainedModel): Hugging Face audio classification model.
        feature_extractor (PreTrainedFeatureExtractor): Corresponding feature extractor.
        label_map (dict): Mapping from class index to label.
        window_size (int): Duration of each chunk in seconds (default: 10).

    Returns:
        dict: {
            "accent": str,
            "confidence_score": float,
            "max_chunk_confidence": float,
            "avg_chunk_confidence": float,
            "summary": str
        }
    """
    waveform, sr = torchaudio.load(audio_path)

    if sr != 16000:
        waveform = librosa.resample(waveform.numpy()[0], orig_sr=sr, target_sr=16000)
        waveform = torch.tensor(waveform).unsqueeze(0)

    chunk_len = 16000 * window_size
    total_len = waveform.shape[1]
    chunks = (total_len + chunk_len - 1) // chunk_len

    logging.info(f"Total chunks: {chunks} for audio: {audio_path}")

    prob_vectors = []
    max_confidences = []

    for i in range(chunks):
        start = i * chunk_len
        end = min(start + chunk_len, total_len)
        chunk = waveform[:, start:end]

        logging.info(f"Processing chunk {i + 1}/{chunks} | Start: {start} | End: {end} | Samples: {chunk.shape[1]}")

        proba = classify_accent_proba(chunk, model, feature_extractor)
        prob_vectors.append(proba)
        
        max_val, max_idx = torch.max(proba, dim=0)
        max_conf = max_val.item() * 100
        predicted_label = model.config.id2label[max_idx.item()] if hasattr(model.config, "id2label") else label_map[max_idx.item()]
        max_confidences.append(max_conf)
        

        logging.info(f"Chunk {i + 1} predicted accent: {predicted_label} | Confidence: {max_conf:.2f}%")


    if not prob_vectors:
        return {
            "accent": "unknown",
            "confidence_score": 0,
            "max_chunk_confidence": 0,
            "avg_chunk_confidence": 0,
            "summary": "Unable to classify accent due to short or silent audio."
        }

    mean_proba = torch.stack(prob_vectors).mean(dim=0)
    best_idx = torch.argmax(mean_proba).item()
    final_confidence = round(mean_proba[best_idx].item() * 100, 2)
    predicted = label_map[best_idx]

    max_chunk_conf = round(max(max_confidences), 2)
    avg_chunk_conf = round(sum(max_confidences) / len(max_confidences), 2)

    logging.info(f"Final Prediction: {predicted} | Confidence: {final_confidence}%")
    logging.info(f"Max chunk confidence: {max_chunk_conf}% | Avg chunk confidence: {avg_chunk_conf}%")

    return {
        "accent": predicted,
        "confidence_score": final_confidence,
        "max_chunk_confidence": max_chunk_conf,
        "avg_chunk_confidence": avg_chunk_conf,
        "summary": (
            f"Predicted {predicted} with {final_confidence}% confidence "
            f"after analyzing {len(prob_vectors) * window_size} seconds of audio. "
            f"Max chunk confidence was {max_chunk_conf}%. "
            f"Average chunk confidence: {avg_chunk_conf}%."
        )
    }


def generate_unique_paths():
    """
    Generates a unique identifier (UUID) and corresponding file paths for a video and its 
    extracted audio. It also ensures that required directories exist for organizing downloaded, 
    processed, and failed files. Avoid overwriting files and keep them uniquely identifiable

    This function helps to:
    - Assign a unique filename for each video/audio pair
    - Maintain structured folders for downloads, audio files, processed, and failed attempts

    Returns:
        Tuple[str, str, str, dict]:
            - uid (str): A unique identifier for the current processing session.
            - video_path (str): File path for storing the downloaded video (e.g., downloads/<uuid>.mp4).
            - audio_path (str): File path for storing the extracted audio (e.g., audios/<uuid>.wav).
            - dirs (dict): A dictionary containing the absolute paths of all key folders:
                {
                    "download": <download_dir>,
                    "audio": <audio_dir>,
                    "processed": <processed_dir>,
                    "failed": <failed_dir>,
                }
    """
    uid = str(uuid.uuid4())
    base_dir = os.path.abspath(".")
    
    dirs = {
        "download": os.path.join(base_dir, "downloads"),
        "audio": os.path.join(base_dir, "audios"),
        "processed": os.path.join(base_dir, "processed"),
        "failed": os.path.join(base_dir, "failed"),
    }

    # Ensure all directories exist
    for path in dirs.values():
        os.makedirs(path, exist_ok=True)

    video_path = os.path.join(dirs["download"], f"{uid}.mp4")
    audio_path = os.path.join(dirs["audio"], f"{uid}.wav")

    return uid, video_path, audio_path, dirs

# ---------------------- Streamlit UI ----------------------

st.set_page_config(page_title="English Accent Classifier", page_icon="🗣️")
st.title("🗣️ English Accent Classifier")
st.write("Paste a public video URL (e.g. YouTube ) to detect the speaker's English accent.")

video_url = st.text_input("Enter video URL:")
start_button = st.button("Detect")

if start_button and video_url:
    with st.spinner("Downloading and analyzing..."):
        model, feature_extractor = load_model(DEFAULT_MODEL_ID)
        
        uid, video_path, audio_path, dirs = generate_unique_paths()
        logging.info(f"Operation started for uid: {uid}")
        
        video_path = process_video_url(video_url, video_path)
        if not video_path:
            st.stop()

        audio_path = extract_audio_from_video(video_path, audio_path, DEFAULT_SAMPLE_RATE)

        if not audio_path:
            st.stop()
        try:
            classification_response = classify_accent(audio_path, model, feature_extractor, LABEL_MAP, DEFAULT_WINDOW_SIZE)
            print(classification_response)
        
            # Move files to processed after successful processing
            shutil.move(video_path, os.path.join(dirs["processed"], os.path.basename(video_path)))
            shutil.move(audio_path, os.path.join(dirs["processed"], os.path.basename(audio_path)))
            logging.info(f"Successfully processed and moved files for video: {video_url}")
        except Exception as e:
            print(f"Error: {str(e)}")
            logging.error(f"Processing failed: for {video_url} {str(e)}")

            # Move any existing files to the failed folder
            if os.path.exists(video_path):
                shutil.move(video_path, os.path.join(dirs["failed"], os.path.basename(video_path)))
            if os.path.exists(audio_path):
                shutil.move(audio_path, os.path.join(dirs["failed"], os.path.basename(audio_path)))
    

    st.subheader("🔍 Result")
    st.json(classification_response)

  
