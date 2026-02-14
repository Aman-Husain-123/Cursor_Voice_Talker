from dotenv import load_dotenv
import os

# Load environment variables first, before other imports that need them
load_dotenv()

import speech_recognition as sr
from langgraph.checkpoint.mongodb import MongoDBSaver
from .graph import create_chat_graph
import asyncio
from openai.helpers import LocalAudioPlayer
from openai import AsyncOpenAI

# Do not print the actual API key for security reasons
if os.getenv("OPENAI_API_KEY"):
    print("OpenAI API key loaded from environment.")
else:
    print("OpenAI API key is NOT set. Check your .env file.")

openai = AsyncOpenAI()

# Read MongoDB URI from environment, with a safe default for local dev
MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://admin:admin@localhost:27017")
config = {"configurable": {"thread_id": "8"}}


def main():
    with MongoDBSaver.from_conn_string(MONGODB_URI) as checkpointer:
        graph = create_chat_graph(checkpointer=checkpointer)

        r = sr.Recognizer()

        with sr.Microphone() as source:
            r.adjust_for_ambient_noise(source)
            r.pause_threshold = 2

            while True:
                print("Say something!")
                audio = r.listen(source)

                print("Processing audio...")
                try:
                    sst = r.recognize_google(audio)
                except sr.UnknownValueError:
                    print("Sorry, I could not understand the audio. Please try again.")
                    continue
                except sr.RequestError as e:
                    print(f"Could not request results from Google Speech Recognition service; {e}")
                    continue

                print("You Said:", sst)

                # Collect the last assistant message text while streaming
                last_ai_text = None
                for event in graph.stream({"messages": [{"role": "user", "content": sst}]}, config, stream_mode="values"):
                    if "messages" in event:
                        event["messages"][-1].pretty_print()
                        msg = event["messages"][-1]
                        if getattr(msg, "type", None) == "ai" and getattr(msg, "content", None):
                            last_ai_text = msg.content

                # After each response, speak a brief summary of what was done, but keep it short
                if last_ai_text:
                    # Convert to string and truncate to avoid exceeding TTS token limits
                    text_str = last_ai_text if isinstance(last_ai_text, str) else str(last_ai_text)
                    short_text = text_str[:400]  # keep it reasonably short
                    summary_text = "Here is my response: " + short_text
                else:
                    summary_text = "I have completed your request."

                # Run TTS in a separate event loop invocation
                try:
                    asyncio.run(speak(summary_text))
                except Exception as e:
                    print(f"Error during speech synthesis: {e}")


async def speak(text: str):
    async with openai.audio.speech.with_streaming_response.create(
        model="gpt-4o-mini-tts",
        voice="coral",
        input=text,
        instructions="Speak in a cheerful and positive tone.",
        response_format="pcm",
    ) as response:
        await LocalAudioPlayer().play(response)


main()

# if __name__ == "__main__":
#      asyncio.run(speak(text="This is a sample voice. Hi Piyush"))