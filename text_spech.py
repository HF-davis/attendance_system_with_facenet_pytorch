import pyttsx3


def text_to_speech(msg):
    engine=pyttsx3.init()
    txt=msg
    engine.setProperty("rate", 150)
    engine.say(txt)
    engine.runAndWait()
