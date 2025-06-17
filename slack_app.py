import os
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler
from dotenv import load_dotenv
from src.main import ask_for_help
load_dotenv()


app = App(token=os.environ.get("SLACK_BOT_TOKEN"))


# Message handler for Slack
@app.event({"type": "message", "subtype": None})
def handle_message_events(message, say):
    print(message)
    output = ask_for_help(message['text'])
    print(output)

    say(output)

@app.event("app_mention")
def handle_mention(message, say):
    print(message)
    output = ask_for_help(message['text'])
    print(output)

    say(output)


if __name__ == "__main__":

    SocketModeHandler(app, os.environ.get("SLACK_APP_TOKEN")).start()
