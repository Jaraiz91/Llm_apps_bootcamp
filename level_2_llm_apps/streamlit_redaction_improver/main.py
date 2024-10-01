import streamlit as st
from langchain_core.prompts import PromptTemplate
from langchain_openai import OpenAI

template = """
Below is a draft text that may be poorly worded.
    Your goal is to:
    - Properly redact the draft text
    - Convert the draft text to a specified tone
    - Convert the draft text to a specified dialect

    Here are some examples different Tones:
    - Formal: Greetings! OpenAI has announced that Sam Altman is rejoining the company as its Chief Executive Officer. After a period of five days of conversations, discussions, and deliberations, the decision to bring back Altman, who had been previously dismissed, has been made. We are delighted to welcome Sam back to OpenAI.
    - Informal: Hey everyone, it's been a wild week! We've got some exciting news to share - Sam Altman is back at OpenAI, taking up the role of chief executive. After a bunch of intense talks, debates, and convincing, Altman is making his triumphant return to the AI startup he co-founded.  

    Here are some examples of words in different dialects:
    - American: French Fries, cotton candy, apartment, garbage, \
        cookie, green thumb, parking lot, pants, windshield
    - British: chips, candyfloss, flag, rubbish, biscuit, green fingers, \
        car park, trousers, windscreen

    Example Sentences from each dialect:
    - American: Greetings! OpenAI has announced that Sam Altman is rejoining the company as its Chief Executive Officer. After a period of five days of conversations, discussions, and deliberations, the decision to bring back Altman, who had been previously dismissed, has been made. We are delighted to welcome Sam back to OpenAI.
    - British: On Wednesday, OpenAI, the esteemed artificial intelligence start-up, announced that Sam Altman would be returning as its Chief Executive Officer. This decisive move follows five days of deliberation, discourse and persuasion, after Altman's abrupt departure from the company which he had co-established.

    Please start the redaction with a warm introduction. Add the introduction \
        if you need to.
    
    Below is the draft text, tone, and dialect:
    DRAFT: {draft}
    TONE: {tone}
    DIALECT: {dialect}

    YOUR {dialect} RESPONSE:

"""

#PrompTemplate variables definition
prompt = PromptTemplate(
    input_variables=['tone', 'dialect', 'draft'],
    template=template
)


#LLM and key loading function
def load_llm(openai_api_key):
    llm = OpenAI(temperature=.7, openai_api_key=openai_api_key)
    return llm

#Page title and header
st.set_page_config(page_title="Text Re-writer")
st.header("Re-write your text")

#Intro: instructions
col1, col2 = st.columns(2)

with col1:
    st.markdown("Re-write your text in different styles.")

with col2:
    st.write("Contact with [AI Accelera](https://aiaccelera.com) to build your AI Projects")


#Input OpenAI API Key
st.markdown("## Enter Your OpenAI API Key")

def get_openai_key():
    input_text = st.text_input(label="OpenAI API Key ", placeholder="Ex: sk-2twmA8tfCb8un4...", key="openai_api_key_input", type="password")
    return input_text

openai_api_key = get_openai_key()

# Input
st.markdown('## Enter the text you want to re-write')

def get_draft():
    draft_text = st.text_area(label='text', label_visibility='collapsed', placeholder='Your text...', key='draft_input')
    return draft_text

draft_input = get_draft()

if len(draft_input.split(' ')) > 700:
    st.write("Please enter a shorter text. Maximum length is 700 words")
    st.stop()

#prompt template tuning options
col1, col2 = st.columns(2)

with col1:
    option_tone = st.selectbox(
        'which tone would you like your redaction to hace?',
        ('Formal', 'Informal')
    )

with col2:
    option_dialect = st.selectbox(
        'Which English Dialect would you like?',
        ('British', 'American')
    )

# Output
st.markdown("### Your re-written text:")

if draft_input:
    if not openai_api_key:
        st.warning('Please insert OpenAI API Key. \
            Instructions [here](https://help.openai.com/en/articles/4936850-where-do-i-find-my-secret-api-key)', 
            icon="⚠️")
        st.stop()

    llm = load_llm(openai_api_key=openai_api_key)

    chain = prompt | llm

    improved_redaction = chain.invoke(
        {
            'tone':option_tone,
            'dialect': option_dialect,
            'draft': draft_input
        }
    )

    st.write(improved_redaction)