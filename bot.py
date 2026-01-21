"""
Discord Channel Summarizer Bot (Deepseek Version)
==================================================
A bot that lets users DM it to get summaries of server channels.
Uses Deepseek AI for summarization (way cheaper than OpenAI!)

"""

import discord
from discord.ext import commands
import openai  # Deepseek uses OpenAI-compatible API
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv
import re

# Load environment variables from .env file
load_dotenv()

# =============================================================================
# CONFIGURATION
# =============================================================================

DISCORD_TOKEN = os.getenv('DISCORD_TOKEN')
DEEPSEEK_API_KEY = os.getenv('DEEPSEEK_API_KEY')

# Bot settings
MAX_MESSAGES = 500
DEFAULT_MESSAGES = 50

# =============================================================================
# BOT SETUP
# =============================================================================

intents = discord.Intents.default()
intents.message_content = True
intents.messages = True
intents.guilds = True
intents.dm_messages = True
intents.members = True

bot = commands.Bot(command_prefix='!', intents=intents)

# Set up Deepseek client (OpenAI-compatible API)
deepseek_client = openai.OpenAI(
    api_key=DEEPSEEK_API_KEY,
    base_url="https://api.deepseek.com"  # This is the key difference!
)

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def find_channel_in_guilds(bot, channel_name: str, user_id: int):
    """Find a channel by name across all servers the bot and user share."""
    channel_name = channel_name.strip().lstrip('#').lower()
    
    for guild in bot.guilds:
        member = guild.get_member(user_id)
        if member is None:
            continue
            
        for channel in guild.text_channels:
            if channel.name.lower() == channel_name:
                if channel.permissions_for(member).read_messages:
                    return channel
    
    return None


async def fetch_messages(channel, limit: int = None, hours: int = None):
    """Fetch messages from a channel."""
    messages = []
    after_time = None
    
    if hours:
        after_time = datetime.utcnow() - timedelta(hours=hours)
    
    fetch_limit = min(limit or DEFAULT_MESSAGES, MAX_MESSAGES)
    
    async for message in channel.history(limit=fetch_limit, after=after_time, oldest_first=True):
        if message.author.bot:
            continue
            
        messages.append({
            'author': message.author.display_name,
            'content': message.content,
            'timestamp': message.created_at.strftime('%Y-%m-%d %H:%M'),
        })
    
    return messages


async def summarize_with_deepseek(messages: list, channel_name: str) -> str:
    """Send messages to Deepseek for summarization."""
    if not messages:
        return "No messages found in the specified time range."
    
    formatted_messages = "\n".join([
        f"[{msg['timestamp']}] {msg['author']}: {msg['content']}"
        for msg in messages
        if msg['content']
    ])
    
    prompt = f"""Please summarize the following Discord conversation from the #{channel_name} channel. 

Provide a concise but comprehensive summary that includes:
- Main topics discussed
- Key decisions or conclusions reached
- Important announcements or information shared
- Any action items or things people should know about

Keep the summary organized with bullet points for easy reading.

Here are the messages:

{formatted_messages}"""

    try:
        response = deepseek_client.chat.completions.create(
            model="deepseek-chat",  # Deepseek's main model
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant that summarizes Discord conversations. Be concise but don't miss important details. Use bullet points for clarity."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            max_tokens=1000,
            temperature=0.3
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        return f"❌ Error during summarization: {str(e)}"


def parse_user_request(content: str):
    """Parse the user's DM to extract channel name and parameters."""
    content = content.lower().strip()
    
    result = {
        'channel': None,
        'limit': None,
        'hours': None
    }
    
    # Find channel mention
    channel_match = re.search(r'#([\w-]+)', content)
    if channel_match:
        result['channel'] = channel_match.group(1)
    else:
        keywords = ['summarize', 'summary', 'recap', 'what happened in', 'catch me up on', 'in']
        for keyword in keywords:
            if keyword in content:
                after_keyword = content.split(keyword)[-1].strip()
                words = after_keyword.split()
                if words:
                    result['channel'] = words[0].lstrip('#')
                break
    
    # Message count
    count_match = re.search(r'(\d+)\s*(messages?|msgs?)?(?!\s*h)', content)
    if count_match:
        result['limit'] = min(int(count_match.group(1)), MAX_MESSAGES)
    
    # Hours
    hours_match = re.search(r'(\d+)\s*h(?:ours?)?', content)
    if hours_match:
        result['hours'] = int(hours_match.group(1))
    
    # Common time phrases
    if 'last hour' in content:
        result['hours'] = 1
    elif 'last day' in content or 'past day' in content or '24 hours' in content:
        result['hours'] = 24
    elif 'last week' in content or 'past week' in content:
        result['hours'] = 168
    
    return result


# =============================================================================
# BOT EVENTS
# =============================================================================

@bot.event
async def on_ready():
    print(f'{"="*50}')
    print(f'Bot is ready!')
    print(f'Logged in as: {bot.user.name}')
    print(f'Bot ID: {bot.user.id}')
    print(f'Connected to {len(bot.guilds)} server(s):')
    for guild in bot.guilds:
        print(f'  - {guild.name}')
    print(f'{"="*50}')
    print('Waiting for DMs...')


@bot.event
async def on_message(message):
    if message.author == bot.user:
        return
    
    # Only respond to DMs
    if not isinstance(message.channel, discord.DMChannel):
        return
    
    content = message.content.strip()
    
    # Help command
    if content.lower() in ['help', '!help', 'commands', '?']:
        await send_help_message(message.channel)
        return
    
    # List servers
    if content.lower() in ['servers', 'list servers', 'my servers']:
        await list_user_servers(message)
        return
    
    # List channels
    if content.lower().startswith('channels') or content.lower().startswith('list channels'):
        await list_server_channels(message)
        return
    
    # Parse summary request
    parsed = parse_user_request(content)
    
    if not parsed['channel']:
        await message.channel.send(
            "🤔 I couldn't understand which channel you want summarized.\n\n"
            "**Try something like:**\n"
            "• `summarize #general`\n"
            "• `summarize #announcements last 100 messages`\n"
            "• `summarize #chat last 24h`\n\n"
            "Type `help` for more options!"
        )
        return
    
    # Find the channel
    channel = find_channel_in_guilds(bot, parsed['channel'], message.author.id)
    
    if not channel:
        await message.channel.send(
            f"❌ I couldn't find a channel called **#{parsed['channel']}** in any server we share.\n\n"
            "Type `servers` to see shared servers, or `channels` to list available channels."
        )
        return
    
    # Check bot permissions
    bot_member = channel.guild.get_member(bot.user.id)
    if not channel.permissions_for(bot_member).read_message_history:
        await message.channel.send(
            f"❌ I don't have permission to read message history in **#{channel.name}**.\n"
            "Please ask a server admin to give me the 'Read Message History' permission."
        )
        return
    
    # Working message
    status_msg = await message.channel.send(
        f"📖 Reading messages from **#{channel.name}** in **{channel.guild.name}**..."
    )
    
    try:
        # Fetch messages
        messages = await fetch_messages(
            channel, 
            limit=parsed['limit'], 
            hours=parsed['hours']
        )
        
        if not messages:
            await status_msg.edit(content=
                f"📭 No messages found in **#{channel.name}**"
                + (f" from the last {parsed['hours']} hours." if parsed['hours'] else ".")
            )
            return
        
        await status_msg.edit(content=
            f"🤖 Summarizing {len(messages)} messages from **#{channel.name}**..."
        )
        
        # Get summary
        summary = await summarize_with_deepseek(messages, channel.name)
        
        # Create embed
        embed = discord.Embed(
            title=f"📋 Summary of #{channel.name}",
            description=summary,
            color=discord.Color.blue(),
            timestamp=datetime.utcnow()
        )
        embed.set_footer(text=f"Server: {channel.guild.name} • {len(messages)} messages analyzed")
        
        if parsed['hours']:
            embed.add_field(name="Time Range", value=f"Last {parsed['hours']} hour(s)", inline=True)
        if parsed['limit']:
            embed.add_field(name="Message Limit", value=str(parsed['limit']), inline=True)
        
        await status_msg.delete()
        await message.channel.send(embed=embed)
        
    except discord.Forbidden:
        await status_msg.edit(content="❌ I don't have permission to read that channel.")
    except Exception as e:
        await status_msg.edit(content=f"❌ An error occurred: {str(e)}")
        print(f"Error: {e}")


async def send_help_message(channel):
    embed = discord.Embed(
        title="📚 Discord Summarizer Bot - Help",
        description="I help you catch up on Discord conversations without the clutter!",
        color=discord.Color.green()
    )
    
    embed.add_field(
        name="🔹 Basic Summary",
        value="```summarize #channel-name```",
        inline=False
    )
    
    embed.add_field(
        name="🔹 Specific Message Count",
        value="```summarize #channel-name 100 messages```",
        inline=False
    )
    
    embed.add_field(
        name="🔹 Time-Based Summary",
        value="```summarize #channel-name last 24h```",
        inline=False
    )
    
    embed.add_field(
        name="🔹 Other Commands",
        value="`servers` - List servers we share\n`channels` - List available channels\n`help` - Show this message",
        inline=False
    )
    
    await channel.send(embed=embed)


async def list_user_servers(message):
    shared_servers = []
    for guild in bot.guilds:
        if guild.get_member(message.author.id):
            shared_servers.append(guild.name)
    
    if not shared_servers:
        await message.channel.send("❌ We don't share any servers!")
        return
    
    embed = discord.Embed(title="🌐 Shared Servers", color=discord.Color.blue())
    embed.description = "\n".join([f"• {name}" for name in shared_servers])
    await message.channel.send(embed=embed)


async def list_server_channels(message):
    embed = discord.Embed(title="📑 Available Channels", color=discord.Color.blue())
    
    for guild in bot.guilds:
        member = guild.get_member(message.author.id)
        if not member:
            continue
        
        accessible = [f"#{c.name}" for c in guild.text_channels if c.permissions_for(member).read_messages]
        if accessible:
            embed.add_field(
                name=f"📁 {guild.name}",
                value=", ".join(accessible[:20]) + ("..." if len(accessible) > 20 else ""),
                inline=False
            )
    
    await message.channel.send(embed=embed)


# =============================================================================
# RUN
# =============================================================================

if __name__ == "__main__":
    if not DISCORD_TOKEN:
        print("❌ ERROR: DISCORD_TOKEN not found!")
        exit(1)
    
    if not DEEPSEEK_API_KEY:
        print("❌ ERROR: DEEPSEEK_API_KEY not found!")
        exit(1)
    
    print("Starting bot...")
    bot.run(DISCORD_TOKEN)
