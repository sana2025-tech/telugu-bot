# telugu_bot_simple.py
"""
Simple Telugu-learning chatbot (single-file).
- Conversation model: microsoft/DialoGPT-small (small, fast)
- Translation: Helsinki-NLP/opus-mt-en-te  and Helsinki-NLP/opus-mt-te-en
Run: python telugu_bot_simple.py
"""

from transformers import pipeline, Conversation
import sys

# Optional: detect GPU if available (use CPU otherwise)
try:
    import torch
    device = 0 if torch.cuda.is_available() else -1
except Exception:
    device = -1

def contains_telugu(text: str) -> bool:
    # Telugu Unicode block: U+0C00 – U+0C7F
    return any("\u0C00" <= ch <= "\u0C7F" for ch in text)

def safe_get_translation_text(pipe_out):
    # pipeline returns list of dicts like [{'translation_text': '...'}]
    if isinstance(pipe_out, list) and len(pipe_out) > 0:
        d = pipe_out[0]
        return d.get("translation_text") or d.get("generated_text") or str(d)
    return str(pipe_out)

def main():
    print("Loading models (first run will download models). Please wait...")
    try:
        # conversational pipeline (small & simple)
        conv_pipe = pipeline("conversational", model="microsoft/DialoGPT-small", device=device)

        # translation pipelines
        en_to_te = pipeline("translation_en_to_te", model="Helsinki-NLP/opus-mt-en-te", device=device)
        te_to_en = pipeline("translation_te_to_en", model="Helsinki-NLP/opus-mt-te-en", device=device)
    except Exception as e:
        print("Error loading pipelines/models:", e)
        print("Make sure you have internet for the first run and you've installed required packages.")
        sys.exit(1)

    print("Models loaded. Type messages (English or Telugu). Type 'exit' or 'quit' to stop.")
    # maintain conversation state across turns so replies have context
    conversation = Conversation()

    while True:
        try:
            user = input("\nYou: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break

        if not user:
            continue
        if user.lower() in ("exit", "quit"):
            print("Bot: శుభాకాంక్షలు! (Goodbye!)")
            break

        try:
            if contains_telugu(user):
                # user typed Telugu -> translate to English, send to model, translate reply back
                eng_user = safe_get_translation_text(te_to_en(user))
                conversation.add_user_input(eng_user)
                conv_pipe(conversation)
                eng_reply = conversation.generated_responses[-1]  # last generated reply in English
                tel_reply = safe_get_translation_text(en_to_te(eng_reply))
                print("\nBot (Telugu):", tel_reply)
                print("Bot (English):", eng_reply)
            else:
                # user typed English -> send to model -> translate reply to Telugu
                conversation.add_user_input(user)
                conv_pipe(conversation)
                eng_reply = conversation.generated_responses[-1]
                tel_reply = safe_get_translation_text(en_to_te(eng_reply))
                print("\nBot (Telugu):", tel_reply)
                print("Bot (English):", eng_reply)
        except Exception as e:
            print("Generation/translation error:", e)
            print("Try a shorter message or check your internet/installation.")

if __name__ == "__main__":
    main()
