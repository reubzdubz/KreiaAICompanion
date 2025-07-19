import whisper
import keyboard
import sounddevice as sd
from scipy.io.wavfile import write
from elevenlabs.client import ElevenLabs
from elevenlabs import stream
from langchain_ollama import ChatOllama
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

prompt = PromptTemplate(
    template="""Embody the persona of Kreia, the mysterious and complex mentor from Knights of the Old Republic II: The Sith Lords.
    Interact with the user as though they are your pupil, guiding them with a mix of wisdom, subtle manipulation, and challenging questions to provoke introspection.
    Speak in a measured, contemplative tone, weaving deep philosophical insights about the Force, morality, and the nature of choice into your responses.
    Be cryptic yet insightful, often questioning the user's assumptions and offering perspectives that are neither entirely light nor dark, but a fusion of both.
    Maintain an air of intellectual superiority and detachment, but occasionally reveal glimpses of personal conviction and emotional depth.
    Your responses should reflect Kreia's disdain for blind allegiance, her belief in the strength of individual will, and her layered, thought-provoking dialogue.
    Stay true to her multifaceted character to deliver an immersive experience. Yet, only answer in two or three sentences, and skip the foreplay, only answer in dialogue, no actions required.
    Below is your chat history so far: {chat}
    """,
    input_variables=["chat"],
)

elevenlabs_key="insert your 11Labs API key"
client = ElevenLabs(
  api_key=elevenlabs_key,
)


llm = ChatOllama(
    model="llama3.2",
    temperature=0.8,
)

chat=''

rag_chain = prompt | llm |StrOutputParser()


def run(chat,question):
    chat+='\nUser: ' + question
    answer = rag_chain.invoke({"chat": chat})
    chat+='\nYou: '+ answer
    return chat , answer


fs = 22050  # Sample rate
seconds = 7  # Duration of recording
model = whisper.load_model("tiny.en")

while True:
    print("Press ` to talk to the AI")
    keyboard.wait('`')
    print("Recording user response\n")
    myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=1)
    sd.wait()  # Wait until recording is finished
    write("output.wav", fs, myrecording)  # Save as WAV file 
    result = model.transcribe("output.wav")
    print(result["text"])
    chat , answer=run(chat, result["text"])
    print(answer)
    audio_stream = client.text_to_speech.convert_as_stream(
        text=answer,
        voice_id="choose your voice id",
        model_id="eleven_multilingual_v2"
    )
    a=stream(audio_stream)