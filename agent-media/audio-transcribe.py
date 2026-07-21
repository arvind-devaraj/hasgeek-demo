from dotenv import load_dotenv
from langchain_core.tools import StructuredTool
from openai import OpenAI
from pydantic import BaseModel, Field

load_dotenv()
client = OpenAI()


class TranscribeInput(BaseModel):
    audio_path: str = Field(description="Local path to the audio file to transcribe.")


def transcribe_audio_tool(audio_path: str) -> str:
    """Transcribes an audio file to text using OpenAI's Whisper model."""
    with open(audio_path, "rb") as audio_file:
        transcript = client.audio.transcriptions.create(model="whisper-1", file=audio_file)
    return transcript.text


# Wrap the execution function into a LangChain StructuredTool
transcribe_tool = StructuredTool.from_function(
    func=transcribe_audio_tool,
    name="audio_transcriber",
    description=(
        "Transcribes audio files into text using OpenAI's Whisper model. "
        "Use this tool whenever you receive a local file path pointing to audio "
        "that needs transcription."
    ),
    args_schema=TranscribeInput,
)

if __name__ == "__main__":
    # Example: How an LLM agent invokes this tool internally
    result = transcribe_tool.run({"audio_path": "sample_audio.mp3"})
    print(result)
