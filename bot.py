import os
import discord
from discord.ext import commands
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
from io import BytesIO

# Get token
TOKEN = os.environ.get("DISCORD_BOT_TOKEN")
if not TOKEN:
    raise RuntimeError("Set DISCORD_BOT_TOKEN environment variable")

# Intents
intents = discord.Intents.default()
intents.message_content = True

# Disable built-in help
bot = commands.Bot(command_prefix="!", intents=intents, help_command=None)

# Load BLIP model
print("Loading BLIP model...this may take a bit on first run.")
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
print("BLIP model loaded.")

STOP = {"a","an","the","and","is","in","on","of","to","with","for","by"}
waiting_for_image = {}

@bot.event
async def on_ready():
    print(f"Bot logged in as: {bot.user}")

@bot.command(name="help")
async def help_cmd(ctx):
    await ctx.send(
        "Commands:\n"
        "`!image` → bot asks you to upload a picture\n"
        "`!help`  → show this message\n"
    )

@bot.command()
async def image(ctx):
    waiting_for_image[ctx.author.id] = True
    print(f"DEBUG: waiting_for_image set for user {ctx.author.id}")
    await ctx.send("Okay! Please upload the image now.")

@bot.event
async def on_message(message):
   
    if message.author == bot.user:
        return

    
    try:
        print("DEBUG: on_message from:", message.author)
        print("DEBUG: content:", repr(message.content))
        if message.attachments:
            print("DEBUG: attachment filename:", message.attachments[0].filename)
    except Exception as _e:
        print("DEBUG: on_message logging failed", _e)

    #
    await bot.process_commands(message)

    
    if message.attachments:
        # take the first attachment
        att = message.attachments[0]

        # validate file type
        if not att.filename.lower().endswith((".jpg", ".jpeg", ".png", ".webp", ".bmp")):
            await message.channel.send("Please send a valid image file (jpg/png/webp/bmp).")
            
            waiting_for_image[message.author.id] = False
            return

       
        print("DEBUG: Starting image processing for", message.author)
        processing_msg = await message.channel.send("Processing image... (this may take 5–30s on CPU)")

        try:
            img_bytes = await att.read()
            print("DEBUG: downloaded bytes:", len(img_bytes))
            pil_img = Image.open(BytesIO(img_bytes)).convert("RGB")

           
            pil_img.thumbnail((800, 800))

            print("DEBUG: Calling processor()")
            inputs = processor(images=pil_img, return_tensors="pt")
            print("DEBUG: Calling model.generate()")
            out = model.generate(**inputs, max_new_tokens=40)
            print("DEBUG: model.generate() finished")
            caption = processor.decode(out[0], skip_special_tokens=True).strip()
            print("DEBUG: caption:", caption)

            
            text = caption.lower()
            for ch in ".,!?;:\"'()[]{}":
                text = text.replace(ch, " ")
            words = [w for w in text.split() if w not in STOP and len(w) > 2]

            tags = []
            for w in words:
                if w not in tags:
                    tags.append(w)
                if len(tags) == 3:
                    break

            tag_str = ", ".join(tags) if tags else "none"

            # edit processing message with result
            await processing_msg.edit(content=f"**Caption:** {caption}\n\n**Tags:** {tag_str}")
            print("DEBUG: Done processing, replied")

        except Exception as exc:
            import traceback
            traceback.print_exc()
            await processing_msg.edit(content="❌ Sorry, an error occurred while processing the image.")
            print("ERROR: Exception while processing image:", exc)

        finally:
            
            waiting_for_image[message.author.id] = False
            return

    
    if waiting_for_image.get(message.author.id, False) and not message.attachments:
        await message.channel.send("Waiting for an image — please upload a photo in this channel.")# Run bot
bot.run(TOKEN)