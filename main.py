from dotenv import load_dotenv
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI
from openai import OpenAI
from pydub import AudioSegment
import time
import threading
import tempfile
import simpleaudio as sa
import os
import gradio as gr

load_dotenv()

client = OpenAI()

final_response = []
grades = []

interviewer = ChatOpenAI(model_name="gpt-4", temperature=0.7)
checker = ChatOpenAI(model_name="gpt-4", temperature=0.7)

checker_template = """
Role: You are a intelligent code interpreter checker which assess the validity and accuracy of the AI mock interviewer’s (LLM) feedback to the candidate’s response.
Error Score Evaluation:
Question Alignment Check:
    1. Evaluate if the LLM’s feedback aligns appropriately with the question asked to the candidate.
    2. If alignment issues exist, assign an error score:Error Score (1-10): Higher score for significant alignment discrepancies.
Response Accuracy Check:
    1. Assess the accuracy of the information provided by the LLM in its feedback to the candidate.
    2. Identify mistakes or inaccuracies in the LLM’s feedback.
    3. If mistakes are found, assign an error score:Error Score (1-10): Higher score for more substantial mistakes.
Coherence and Clarity Check:
    1. Evaluate the coherence and clarity of the LLM’s feedback. Ensure it provides a clear and understandable assessment.
    2. If issues in coherence or clarity are observed, assign an error score:Error Score (1-10): Higher score for significant coherence or clarity issues.
Overall Error Score Determination:
    1. Combine alignment, accuracy, coherence, and clarity issues to determine the overall error score.
Error Score (1-10): Based on the severity and number of issues found in the LLM’s feedback.

question: this is the question asked to the user by interviewer LLM
user_response: this is the response given by the user on the asked question
interviewer_feedback: this is the feedback given by the interviewer LLM on the response of user

Give only the final Error Score as integer nothing else

question = {ques}
user_response = {user_response}
interviewer_feedback = {interviewer_response}
"""
checker_prompt = ChatPromptTemplate.from_template(checker_template)


def init_conversation():
    memory = ConversationBufferMemory(return_messages=True)
    memory.save_context(
        {
            "input": "As the 'Google Interviewer', your primary focus is on asking coding-related questions, System Design questions, reflective "
                     "of a real Google SDE interview. These questions should cover a range of topics including "
                     "algorithms, data structures, system design, and coding problems and must appear in the usual Google SDE interview. After each user response, "
                     "you will provide a score from 1 to 10, assessing their performance. This scoring system should be "
                     "based on the quality of the solution, efficiency of the code, and the user's problem-solving "
                     "approach. Accompanying the score, you will offer detailed feedback, pointing out the strengths of "
                     "the user's response and suggesting areas for improvement. Your feedback should be constructive and "
                     "educational, helping the user understand how to enhance their coding and problem-solving skills. "
                     "Continue to tailor your questions and feedback to the user's experience level, ensuring that they "
                     "are challenging yet accessible. Maintain your formal and professional demeanor, reflecting "
                     "Google's standards for SDE roles. This refined behavior will provide a more accurate and "
                     "comprehensive interview preparation experience."
                     """Maintaining Interview Focus and Addressing Incorrect Answers:\n Remind the candidate to stay focused on 
         the mock interview setup if they deviate from the designated interview process. Provide limited retries (up 
         to two to three attempts) for incorrect answers. Offer guided assistance but refrain from directly providing 
         the correct solution unless retries are exhausted."""},
        {"output": "Sure I understood"})
    return ConversationChain(llm=interviewer, verbose=False, memory=memory)


def append_text(normal_text, code):
    global final_response
    final = normal_text + "\n\n" + code
    final_response = []
    return main(final)


def transcribe(audio):
    if audio is None:
        return ""
    with open(audio, "rb") as audio_file:
        transcript = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file
        )
    final_response.append(transcript.text)
    return "\n".join(final_response)


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


"""
question asked,
user response,
interviewer_response
"""

i = 0
question_asked = None
interview_conversation = init_conversation()


def main(user_input):
    global i
    global interview_conversation
    global question_asked
    if i != 0 and i % 4:
        interview_conversation = init_conversation()
    ai_response = interview_conversation.predict(input=user_input)
    if i % 2:
        checker_mes = checker_prompt.format_messages(ques=question_asked,
                                                     user_response=user_input,
                                                     interviewer_response=ai_response)
        marks = checker(checker_mes)
        grades.append(marks)
        print("Interviewer Performance: ", marks.content)
    else:
        question_asked = ai_response

    response = client.audio.speech.create(
        model="tts-1",
        voice="alloy",
        speed=1.5,
        input=ai_response
    )
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as temp_file:
        response.stream_to_file(temp_file.name)
        playback_thread = threading.Thread(target=play_audio, args=(temp_file.name,))
        playback_thread.start()
        playback_thread.join()
    i += 1

    return ai_response


if __name__ == '__main__':
    with gr.Blocks() as app:
        with gr.Row():
            gr.Markdown("<center><h1> AI Mock Interview </h1><center>")
        with gr.Row():
            gr.HTML(
                "<center><img src='https://raw.githubusercontent.com/SwayamInSync/Scaler-Hackathon/main/banner.png' width='700px'></center>")
        with gr.Row():
            with gr.Column():
                audio_input = gr.Audio(sources=["microphone"], type="filepath", label="Record Audio")
            with gr.Column():
                normal_text = gr.Textbox(label="Informal Response")
        with gr.Row():
            final_output = gr.Textbox(label="Final Output", interactive=False, visible=True)
        with gr.Row():
            submit_button = gr.Button("Submit")
        # with gr.Row():
        # finish_button = gr.Button("Finish Interview")
        transcribed_text = gr.Textbox(visible=False)
        code_input = gr.Code(label="Code")

        audio_input.change(fn=transcribe, inputs=audio_input, outputs=[normal_text])
        normal_text.change(fn=lambda x: final_response.append(x), inputs=normal_text, outputs=None)

        submit_button.click(fn=append_text, inputs=[normal_text, code_input],
                            outputs=[final_output])
        # finish_button.click(fn=lambda : print(sum([int(i.strip()) for i in grades])))

    app.launch()
