"""
Discord Channel Summarizer Bot
==============================
A bot that lets users DM it to get summaries of server channels.
No messages are ever posted in the server - everything stays in DMs.

Author: Your Name
"""

import discord
from discord.ext import commands
from discord import app_commands
import openai
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv
import re

# Load environment variables from .env file
load_dotenv()

# =============================================================================
# CONFIGURATION
# =============================================================================

# Get tokens from environment variables (NEVER hardcode these!)
DISCORD_TOKEN = os.getenv('DISCORD_TOKEN')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# Bot settings
MAX_MESSAGES = 500  # Maximum messages to fetch at once (Discord limit is 100 per request)
DEFAULT_MESSAGES = 50  # Default number of messages if user doesn't specify

# =============================================================================
# BOT SETUP
# =============================================================================

# Set up bot with required permissions/intents
intents = discord.Intents.default()
intents.message_content = True  # Required to read message content
intents.messages = True
intents.guilds = True
intents.dm_messages = True  # Required to receive DMs

# Create bot instance
bot = commands.Bot(command_prefix='!', intents=intents)

# Set up OpenAI client
openai_client = openai.OpenAI(api_key=OPENAI_API_KEY)

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def find_channel_in_guilds(bot, channel_name: str, user_id: int):
    """
    Find a channel by name across all servers the bot and user share.
    Returns the channel object or None if not found.
    """
    # Clean up channel name (remove # if present)
    channel_name = channel_name.strip().lstrip('#').lower()
    
    for guild in bot.guilds:
        # Check if user is in this guild
        member = guild.get_member(user_id)
        if member is None:
            continue
            
        # Search for the channel
        for channel in guild.text_channels:
            if channel.name.lower() == channel_name:
                # Verify user can actually see this channel
                if channel.permissions_for(member).read_messages:
                    return channel
    
    return None


def find_channel_by_id(bot, channel_id: int, user_id: int):
    """
    Find a channel by ID and verify user has access.
    """
    channel = bot.get_channel(channel_id)
    if channel is None:
        return None
    
    # Check if user is in the guild and can see the channel
    member = channel.guild.get_member(user_id)
    if member is None:
        return None
    
    if not channel.permissions_for(member).read_messages:
        return None
    
    return channel


async def fetch_messages(channel, limit: int = None, hours: int = None):
    """
    Fetch messages from a channel.
    
    Args:
        channel: The Discord channel to fetch from
        limit: Maximum number of messages to fetch
        hours: Only fetch messages from the last X hours
    
    Returns:
        List of message dictionaries
    """
    messages = []
    after_time = None
    
    if hours:
        after_time = datetime.utcnow() - timedelta(hours=hours)
    
    # Use limit or default
    fetch_limit = min(limit or DEFAULT_MESSAGES, MAX_MESSAGES)
    
    async for message in channel.history(limit=fetch_limit, after=after_time, oldest_first=True):
        # Skip bot messages (optional - you might want to include them)
        if message.author.bot:
            continue
            
        messages.append({
            'author': message.author.display_name,
            'content': message.content,
            'timestamp': message.created_at.strftime('%Y-%m-%d %H:%M'),
            'attachments': len(message.attachments),
            'reactions': len(message.reactions)
        })
    
    return messages


async def summarize_with_ai(messages: list, channel_name: str) -> str:
    """
    Send messages to OpenAI for summarization.
    
    Args:
        messages: List of message dictionaries
        channel_name: Name of the channel (for context)
    
    Returns:
        Summary string
    """
    if not messages:
        return "No messages found in the specified time range."
    
    # Format messages for the AI
    formatted_messages = "\n".join([
        f"[{msg['timestamp']}] {msg['author']}: {msg['content']}"
        for msg in messages
        if msg['content']  # Skip empty messages
    ])
    
    # Create the prompt
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
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",  # Cost-effective and good quality
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
            temperature=0.3  # Lower temperature for more consistent summaries
        )
        
        return response.choices[0].message.content
        
    except openai.APIError as e:
        return f"❌ Error communicating with AI service: {str(e)}"
    except Exception as e:
        return f"❌ Unexpected error during summarization: {str(e)}"


def parse_user_request(content: str):
    """
    Parse the user's DM to extract channel name and parameters.
    
    Supported formats:
    - "summarize #general"
    - "summarize #general 100" (last 100 messages)
    - "summarize #general 24h" (last 24 hours)
    - "#general last 50 messages"
    - "what happened in #announcements"
    
    Returns:
        dict with 'channel', 'limit', 'hours' keys
    """
    content = content.lower().strip()
    
    result = {
        'channel': None,
        'limit': None,
        'hours': None
    }
    
    # Try to find a channel mention (with #)
    channel_match = re.search(r'#([\w-]+)', content)
    if channel_match:
        result['channel'] = channel_match.group(1)
    else:
        # Try to find channel name without # (after keywords)
        keywords = ['summarize', 'summary', 'recap', 'what happened in', 'catch me up on', 'in']
        for keyword in keywords:
            if keyword in content:
                # Get text after the keyword
                after_keyword = content.split(keyword)[-1].strip()
                # First word is probably the channel
                words = after_keyword.split()
                if words:
                    result['channel'] = words[0].lstrip('#')
                break
    
    # Look for message count
    count_match = re.search(r'(\d+)\s*(messages?|msgs?)?(?!\s*h)', content)
    if count_match:
        result['limit'] = min(int(count_match.group(1)), MAX_MESSAGES)
    
    # Look for hours
    hours_match = re.search(r'(\d+)\s*h(?:ours?)?', content)
    if hours_match:
        result['hours'] = int(hours_match.group(1))
    
    # Look for common time phrases
    if 'last hour' in content:
        result['hours'] = 1
    elif 'last day' in content or 'past day' in content or '24 hours' in content:
        result['hours'] = 24
    elif 'last week' in content or 'past week' in content:
        result['hours'] = 168  # 7 days
    
    return result


# =============================================================================
# BOT EVENTS
# =============================================================================

@bot.event
async def on_ready():
    """Called when the bot successfully connects to Discord."""
    print(f'{"="*50}')
    print(f'Bot is ready!')
    print(f'Logged in as: {bot.user.name}')
    print(f'Bot ID: {bot.user.id}')
    print(f'Connected to {len(bot.guilds)} server(s):')
    for guild in bot.guilds:
        print(f'  - {guild.name} (ID: {guild.id})')
    print(f'{"="*50}')
    print('Waiting for DMs...')


@bot.event
async def on_message(message):
    """Handle incoming messages."""
    
    # Ignore messages from the bot itself
    if message.author == bot.user:
        return
    
    # Only respond to DMs
    if not isinstance(message.channel, discord.DMChannel):
        return
    
    # Get the message content
    content = message.content.strip()
    
    # Handle help command
    if content.lower() in ['help', '!help', 'commands', '?']:
        await send_help_message(message.channel)
        return
    
    # Handle list servers command
    if content.lower() in ['servers', 'list servers', 'my servers']:
        await list_user_servers(message)
        return
    
    # Handle list channels command
    if content.lower().startswith('channels') or content.lower().startswith('list channels'):
        await list_server_channels(message)
        return
    
    # Try to parse as a summary request
    parsed = parse_user_request(content)
    
    if not parsed['channel']:
        # Couldn't understand the request
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
            "**Possible reasons:**\n"
            "• The channel name might be spelled differently\n"
            "• You might not have access to that channel\n"
            "• I might not have access to that channel\n\n"
            "Type `servers` to see shared servers, or `channels [server name]` to list available channels."
        )
        return
    
    # Check if bot has permission to read the channel
    bot_member = channel.guild.get_member(bot.user.id)
    if not channel.permissions_for(bot_member).read_message_history:
        await message.channel.send(
            f"❌ I don't have permission to read message history in **#{channel.name}**.\n"
            "Please ask a server admin to give me the 'Read Message History' permission."
        )
        return
    
    # Send a "working on it" message
    status_msg = await message.channel.send(
        f"📖 Reading messages from **#{channel.name}** in **{channel.guild.name}**...\n"
        "This might take a moment."
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
        
        # Update status
        await status_msg.edit(content=
            f"🤖 Summarizing {len(messages)} messages from **#{channel.name}**..."
        )
        
        # Get summary from AI
        summary = await summarize_with_ai(messages, channel.name)
        
        # Create the response embed for a nicer look
        embed = discord.Embed(
            title=f"📋 Summary of #{channel.name}",
            description=summary,
            color=discord.Color.blue(),
            timestamp=datetime.utcnow()
        )
        embed.set_footer(text=f"Server: {channel.guild.name} • {len(messages)} messages analyzed")
        
        # Add time range info if applicable
        if parsed['hours']:
            embed.add_field(name="Time Range", value=f"Last {parsed['hours']} hour(s)", inline=True)
        if parsed['limit']:
            embed.add_field(name="Message Limit", value=str(parsed['limit']), inline=True)
        
        # Delete the status message and send the summary
        await status_msg.delete()
        await message.channel.send(embed=embed)
        
    except discord.Forbidden:
        await status_msg.edit(content=
            "❌ I don't have permission to read that channel. "
            "Please ask a server admin to check my permissions."
        )
    except Exception as e:
        await status_msg.edit(content=f"❌ An error occurred: {str(e)}")
        print(f"Error: {e}")


async def send_help_message(channel):
    """Send the help message."""
    embed = discord.Embed(
        title="📚 Discord Summarizer Bot - Help",
        description="I help you catch up on Discord conversations without the clutter!",
        color=discord.Color.green()
    )
    
    embed.add_field(
        name="🔹 Basic Summary",
        value="```summarize #channel-name```\nGets a summary of the last 50 messages.",
        inline=False
    )
    
    embed.add_field(
        name="🔹 Specific Message Count",
        value="```summarize #channel-name 100 messages```\nSummarize the last 100 messages.",
        inline=False
    )
    
    embed.add_field(
        name="🔹 Time-Based Summary",
        value="```summarize #channel-name last 24h```\nSummarize messages from the last 24 hours.",
        inline=False
    )
    
    embed.add_field(
        name="🔹 Other Commands",
        value=(
            "`servers` - List servers we share\n"
            "`channels [server]` - List channels in a server\n"
            "`help` - Show this message"
        ),
        inline=False
    )
    
    embed.add_field(
        name="💡 Tips",
        value=(
            "• You can use natural language: 'what happened in #general today'\n"
            "• Maximum 500 messages per request\n"
            "• I can only see channels you have access to"
        ),
        inline=False
    )
    
    await channel.send(embed=embed)


async def list_user_servers(message):
    """List servers the user shares with the bot."""
    shared_servers = []
    
    for guild in bot.guilds:
        member = guild.get_member(message.author.id)
        if member:
            shared_servers.append(guild.name)
    
    if not shared_servers:
        await message.channel.send(
            "❌ We don't seem to share any servers!\n"
            "Make sure the bot is added to your server."
        )
        return
    
    embed = discord.Embed(
        title="🌐 Shared Servers",
        description="Here are the servers we both have access to:",
        color=discord.Color.blue()
    )
    
    for server_name in shared_servers:
        embed.add_field(name="", value=f"• {server_name}", inline=False)
    
    embed.set_footer(text="Use 'channels [server name]' to see available channels")
    await message.channel.send(embed=embed)


async def list_server_channels(message):
    """List channels in a specific server."""
    content = message.content.lower().replace('channels', '').replace('list', '').strip()
    
    # If no server specified, list all accessible channels across all servers
    target_guild = None
    
    if content:
        # Find the server by name
        for guild in bot.guilds:
            if guild.name.lower() == content.lower():
                member = guild.get_member(message.author.id)
                if member:
                    target_guild = guild
                    break
    
    if content and not target_guild:
        await message.channel.send(f"❌ Couldn't find a server named '{content}' that we share.")
        return
    
    guilds_to_check = [target_guild] if target_guild else bot.guilds
    
    embed = discord.Embed(
        title="📑 Available Channels",
        color=discord.Color.blue()
    )
    
    for guild in guilds_to_check:
        member = guild.get_member(message.author.id)
        if not member:
            continue
        
        accessible_channels = []
        for channel in guild.text_channels:
            if channel.permissions_for(member).read_messages:
                accessible_channels.append(f"#{channel.name}")
        
        if accessible_channels:
            # Discord has a 1024 character limit per field
            channel_list = ", ".join(accessible_channels[:25])  # Limit to 25 channels
            if len(accessible_channels) > 25:
                channel_list += f" ... and {len(accessible_channels) - 25} more"
            
            embed.add_field(
                name=f"📁 {guild.name}",
                value=channel_list,
                inline=False
            )
    
    await message.channel.send(embed=embed)


# =============================================================================
# RUN THE BOT
# =============================================================================

if __name__ == "__main__":
    # Validate that we have the required tokens
    if not DISCORD_TOKEN:
        print("❌ ERROR: DISCORD_TOKEN not found in .env file!")
        print("Please create a .env file with your Discord bot token.")
        exit(1)
    
    if not OPENAI_API_KEY:
        print("❌ ERROR: OPENAI_API_KEY not found in .env file!")
        print("Please add your OpenAI API key to the .env file.")
        exit(1)
    
    print("Starting bot...")
    bot.run(DISCORD_TOKEN)
