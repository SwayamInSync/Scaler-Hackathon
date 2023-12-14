from dotenv import load_dotenv
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain.chat_models import ChatOpenAI
from openai import OpenAI
from pydub import AudioSegment
import time
import threading
import tempfile
import simpleaudio as sa
import os

load_dotenv()


def play_audio(file_path):
    previous_size = -1
    while True:
        current_size = os.path.getsize(file_path)
        if current_size == previous_size:
            break  # Exit the loop if file size hasn't changed
        previous_size = current_size

        audio_segment = AudioSegment.from_mp3(file_path)
        play_obj = sa.play_buffer(
            audio_segment.raw_data,
            num_channels=audio_segment.channels,
            bytes_per_sample=audio_segment.sample_width,
            sample_rate=audio_segment.frame_rate
        )
        play_obj.wait_done()
        time.sleep(0.5)


def main():
    client = OpenAI()

    llm = ChatOpenAI(model_name="gpt-4", temperature=0.7)
    memory = ConversationBufferMemory(return_messages=True)
    memory.save_context(
        {"input": "As the 'Google Interviewer', your primary focus is on asking coding-related questions, reflective "
                  "of a real Google SDE interview. These questions should cover a range of topics including "
                  "algorithms, data structures, system design, and coding problems. After each user response, "
                  "you will provide a score from 1 to 10, assessing their performance. This scoring system should be "
                  "based on the quality of the solution, efficiency of the code, and the user's problem-solving "
                  "approach. Accompanying the score, you will offer detailed feedback, pointing out the strengths of "
                  "the user's response and suggesting areas for improvement. Your feedback should be constructive and "
                  "educational, helping the user understand how to enhance their coding and problem-solving skills. "
                  "Continue to tailor your questions and feedback to the user's experience level, ensuring that they "
                  "are challenging yet accessible. Maintain your formal and professional demeanor, reflecting "
                  "Google's standards for SDE roles. This refined behavior will provide a more accurate and "
                  "comprehensive interview preparation experience."},
        {"output": "Sure I understood"})

    conversation = ConversationChain(
        llm=llm, verbose=False, memory=memory)

    while True:
        user_input = input("> ")
        if user_input.split(">")[-1].strip() == "break":
            break
        ai_response = conversation.predict(input=user_input)

        response = client.audio.speech.create(
            model="tts-1",
            voice="alloy",
            input=ai_response
        )

        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as temp_file:
            response.stream_to_file(temp_file.name)
            playback_thread = threading.Thread(target=play_audio, args=(temp_file.name,))
            playback_thread.start()
            playback_thread.join()

        print("\nInterviewer:\n", ai_response, "\n")


if __name__ == '__main__':
    main()
