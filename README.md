# Shapes.inc Discord Bot Revival
**Bring shape bots back to discord with a cursed method. *(If you know anything better feel free to open a PR)***
yes, it uses email.

## Before starting, make sure you have the following
- An email SMTP and IMAP server that the bot can use. (Thats how it communicates without breaking shapes TOS)
- A shapes.inc bot (duh) _note down its username_
- A discord bot

## Installation
Clone the repository and enter the directory:
```
git clone https://github.com/Ikelene/discord-shapes-ai.git
cd discord-shapes-ai/
```

Install all required dependancies
```
pip install -r requirements.txt
```

Edit the `.env` file with terminal or a text editor
```
nano .env
```

Enter in your bot token in `DISCORD_TOKEN`, Enter the **email** of the shapes bot in `TO_EMAIL`, its usually `<bot-username>@shapes.inc`, Enter the bots name (it will respond to this) in `BOT_NAME`, Then enter your email server credentials in the `SMTP` and `IMAP` sections. The bot will use this email to send messages to the AI bot, so dont use your main email. *(Unless you love spam)* No, The bot does not delete emails... yet.

Add the bot to your discord server and run it.
```
python3 shapesBot.py
```

Run `/enable [channel]` to enable the bot to respond to its configured name or a ping. You can reply to the bot as well to get it to respond.
uhh if theres a bug then open a bug report or something idk
if you want to make this better so it wont use email, or fix it up, feel free to open a PR.
