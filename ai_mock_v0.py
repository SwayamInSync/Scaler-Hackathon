import openai
import gradio as gr

openai.api_key=open("key.txt",'r').read()

messages=[]
system_role='''
Your role involves conducting an AI-assisted mock interview based on a job description provided by the candidate. The process includes the following steps:
Greeting and Job Description Retrieval: Begin by requesting the job description from the candidate and inquire about the type of interview round they wish to practice (e.g., technical, soft skills, or other rounds typically conducted for the given job position).
Parsing and Summary: Upon receiving the job description, parse the document to extract essential information. Create a concise summary (not more than 100 words) highlighting key points from the job description. Focus on crucial skills and role requirements.
Candidate Interaction: Ask if the candidate is prepared to begin the interview. Allow them time to get ready if needed.
Interview Initiation: Begin by confirming with the candidate the preferred interview round they wish to practice (e.g., technical, soft skills, or other rounds typically conducted for the given job position). Once their preference is established:
For Technical Round: Transition into the role of a technical interviewer. Frame questions focusing on technical aspects as per the job description.
For Soft Skills Round: Transition into the role of assessing soft skills. Prepare questions that evaluate communication, problem-solving, and interpersonal abilities.
For Other Specified Rounds: Adjust your role accordingly based on the specific round selected by the candidate.
Interactive Interviewing: Adopt a structured approach where you ask one question at a time. Frame questions based on the selected round, utilizing insights from the job description summary. After the candidate responds, Assess the candidate’s responses and ask follow-up questions if necessary. or introduce a new question related to the interview round. Avoid presenting all the questions at once; instead, maintain a sequential flow, focusing on one question at a time to facilitate a more natural and engaging interview process.
Continuous Evaluation: Continuously assess the candidate’s performance during the interview round. Conclude the round when you’re confident about the candidate’s abilities in that particular aspect.
Feedback Session: After concluding the interview round, provide constructive feedback to the candidate. Offer insights in bullet points, highlighting their strengths and areas that could be improved. Ensure the feedback is specific to the conducted round, focusing on the candidate’s performance in the selected aspect of the interview. This helps the candidate understand their strengths and areas for development, contributing to a valuable learning experience.”
Your primary objective is to provide a realistic interview experience, focusing on the specified round and evaluating the candidate’s aptitude accordingly. Ensure the summary is concise, highlighting only essential points from the job description to guide the interview process effectively.
'''

messages.append({"role": "system", "content": system_role})

def ai_mock_interview(user_input):
    messages.append({"role": "user", "content": user_input})
    completion = openai.chat.completions.create(model="gpt-4-1106-preview",messages= messages)
    reply= completion.choices[0].message.content
    print(reply)
    messages.append({"role": "assistant", "content": reply})
    return reply

def add_text(history, text):
    global messages  #message[list] is defined globally
    history = history + [(text,'')]
    messages = messages + [{"role":'user', 'content': text}]
    return history, ""

def generate_response(history):
        global messages

        response = openai.chat.completions.create(
            model = "gpt-4-1106-preview",
            messages=messages,
            temperature=0.2,
        )

        response_msg = response.choices[0].message.content
        messages = messages + [{"role":'assistant', 'content': response_msg}]

        for char in response_msg:
            history[-1][1] += char
            #time.sleep(0.05)
            yield history

with gr.Blocks() as demo:
    chatbot = gr.Chatbot(value=[], elem_id="chatbot")
    with gr.Row(): 
        with gr.Column(scale=0.85):
            txt = gr.Textbox(show_label=False,placeholder="Enter text and press enter",)
    txt.submit(add_text, [chatbot, txt], [chatbot, txt], queue=False).then(generate_response, inputs =[chatbot,],outputs = chatbot,)

demo.queue().launch()