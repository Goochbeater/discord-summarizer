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
intents.members = True  # Required to see server members

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
    
    # Determine fetch limit:
    # - If user specified a limit, use it
    # - If user specified hours (time-based), get ALL messages in that window (up to MAX)
    # - If neither, use default
    if limit:
        fetch_limit = min(limit, MAX_MESSAGES)
    elif hours:
        fetch_limit = MAX_MESSAGES  # Get everything in the time window
    else:
        fetch_limit = DEFAULT_MESSAGES
    
    async for message in channel.history(limit=fetch_limit, after=after_time, oldest_first=False):
        if message.author.bot:
            continue
            
        messages.append({
            'author': message.author.display_name,
            'content': message.content,
            'timestamp': message.created_at.strftime('%Y-%m-%d %H:%M'),
        })
    
    return messages


async def summarize_with_deepseek(messages: list, channel_name: str, deep: bool = False) -> str:
    """Send messages to Deepseek for summarization."""
    if not messages:
        return "No messages found in the specified time range."
    
    formatted_messages = "\n".join([
        f"[{msg['timestamp']}] {msg['author']}: {msg['content']}"
        for msg in messages
        if msg['content']
    ])
    
    if deep:
        prompt = f"""Analyze this Discord conversation from #{channel_name} in DETAIL.

I want to know:

**1. CONVERSATION THREADS** - Break down the different conversations that happened. Who started talking about what? When did topics shift?

**2. WHO TALKED TO WHO** - Map out the interactions. Who was replying to who? Were there side conversations? Did some people only talk to certain others?

**3. DEBATES & DISAGREEMENTS** - Were there any arguments, debates, or differing opinions? Who was on which side? How did it resolve (or not)?

**4. KEY PLAYERS** - Who were the most active participants? What was each person's main contribution or stance?

**5. VIBES & DYNAMICS** - What was the overall mood? Any tension? Jokes? Was it casual or serious?

**6. IMPORTANT STUFF** - Decisions made, announcements, action items, things people should definitely know.

Format this in a readable way with clear sections. Be specific about WHO said WHAT.

Here are the messages:

{formatted_messages}"""
    else:
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
        # Use deepseek-chat for both modes (more reliable)
        # Deep mode just gets a more detailed prompt
        response = deepseek_client.chat.completions.create(
            model="deepseek-chat",  # V3.2 - works better than reasoner for this use case
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant that summarizes Discord conversations. Be concise but don't miss important details."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            max_tokens=3000 if deep else 1000,
            temperature=0.3
        )
        
        summary = response.choices[0].message.content
        
        # Handle long responses (Discord embed limit is 4096 chars)
        if len(summary) > 4000:
            summary = summary[:3950] + "\n\n... *(summary truncated due to length)*"
        
        return summary
        
    except Exception as e:
        return f"❌ Error during summarization: {str(e)}"


def parse_user_request(content: str):
    """Parse the user's DM to extract channel name and parameters."""
    content_lower = content.lower().strip()
    
    result = {
        'channel': None,
        'limit': None,
        'hours': None,
        'deep': False
    }
    
    # Check for deep summary mode
    deep_triggers = ['deep', 'detailed', 'detail', 'drama', 'who said', 'breakdown', 'full', 'everything']
    if any(word in content_lower for word in deep_triggers):
        result['deep'] = True
    
    # Find channel mention
    channel_match = re.search(r'#([\w-]+)', content_lower)
    if channel_match:
        result['channel'] = channel_match.group(1)
    else:
        keywords = ['summarize', 'summary', 'recap', 'what happened in', 'catch me up on', 'in']
        for keyword in keywords:
            if keyword in content_lower:
                after_keyword = content_lower.split(keyword)[-1].strip()
                words = after_keyword.split()
                if words:
                    # Skip trigger words to find actual channel name
                    skip_words = ['deep', 'detailed', 'detail', 'drama', 'full', 'the', 'a']
                    for word in words:
                        if word not in skip_words:
                            result['channel'] = word.lstrip('#')
                            break
                break
    
    # Parse HOURS FIRST (before message count to avoid "24h" -> limit=2 bug)
    hours_match = re.search(r'(\d+)\s*h(?:ours?)?', content_lower)
    if hours_match:
        result['hours'] = int(hours_match.group(1))
        # Remove the hours part so it doesn't interfere with message count parsing
        content_for_count = content_lower.replace(hours_match.group(0), ' ')
    else:
        content_for_count = content_lower
    
    # Common time phrases (check these too)
    if 'last hour' in content_lower and not result['hours']:
        result['hours'] = 1
    elif ('last day' in content_lower or 'past day' in content_lower or '24 hours' in content_lower) and not result['hours']:
        result['hours'] = 24
    elif ('last week' in content_lower or 'past week' in content_lower) and not result['hours']:
        result['hours'] = 168
    
    # Message count (from content with hours removed)
    count_match = re.search(r'(\d+)\s*(messages?|msgs?)?', content_for_count)
    if count_match:
        count = int(count_match.group(1))
        # Only accept as message count if it's a reasonable number (not like "1" from random text)
        if count >= 5:  # Minimum 5 to be considered a message count
            result['limit'] = min(count, MAX_MESSAGES)
    
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
    mode_text = "🔍 **DEEP ANALYSIS** of" if parsed['deep'] else "📖 Reading messages from"
    status_msg = await message.channel.send(
        f"{mode_text} **#{channel.name}** in **{channel.guild.name}**..."
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
        
        summarize_text = "🕵️ Deep analyzing" if parsed['deep'] else "🤖 Summarizing"
        await status_msg.edit(content=
            f"{summarize_text} {len(messages)} messages from **#{channel.name}**...\n*(this may take a moment)*"
        )
        
        # Get summary
        summary = await summarize_with_deepseek(messages, channel.name, deep=parsed['deep'])
        
        # Check if we got an error response
        if summary.startswith("❌"):
            await status_msg.edit(content=summary)
            return
        
        # Create embed(s)
        title = f"🕵️ Deep Analysis of #{channel.name}" if parsed['deep'] else f"📋 Summary of #{channel.name}"
        mode_label = "Deep Analysis" if parsed['deep'] else "Summary"
        color = discord.Color.purple() if parsed['deep'] else discord.Color.blue()
        
        await status_msg.delete()
        
        # Split long summaries into chunks (Discord embed limit is 4096)
        MAX_EMBED_LENGTH = 4000  # Leave some room for safety
        
        if len(summary) <= MAX_EMBED_LENGTH:
            # Single embed - fits fine
            embed = discord.Embed(
                title=title,
                description=summary,
                color=color,
                timestamp=datetime.utcnow()
            )
            embed.set_footer(text=f"Server: {channel.guild.name} • {len(messages)} messages • {mode_label}")
            
            if parsed['hours']:
                embed.add_field(name="Time Range", value=f"Last {parsed['hours']} hour(s)", inline=True)
            if parsed['limit']:
                embed.add_field(name="Message Limit", value=str(parsed['limit']), inline=True)
            
            await message.channel.send(embed=embed)
        else:
            # Split into multiple embeds
            chunks = []
            current_chunk = ""
            
            # Split by paragraphs/sections to keep it readable
            paragraphs = summary.split('\n\n')
            for para in paragraphs:
                if len(current_chunk) + len(para) + 2 < MAX_EMBED_LENGTH:
                    current_chunk += para + '\n\n'
                else:
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                    current_chunk = para + '\n\n'
            if current_chunk:
                chunks.append(current_chunk.strip())
            
            # Send first chunk with title
            embed = discord.Embed(
                title=title,
                description=chunks[0],
                color=color,
                timestamp=datetime.utcnow()
            )
            if parsed['hours']:
                embed.add_field(name="Time Range", value=f"Last {parsed['hours']} hour(s)", inline=True)
            embed.set_footer(text=f"Part 1/{len(chunks)}")
            await message.channel.send(embed=embed)
            
            # Send remaining chunks
            for i, chunk in enumerate(chunks[1:], start=2):
                embed = discord.Embed(
                    title=f"{title} (continued)",
                    description=chunk,
                    color=color
                )
                embed.set_footer(text=f"Part {i}/{len(chunks)}" + (f" • {len(messages)} messages • {mode_label}" if i == len(chunks) else ""))
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
        name="🕵️ DEEP Summary (The Good Stuff)",
        value="```deep summary #channel-name```\nShows WHO talked to WHO, debates, side convos, drama, vibes — the full breakdown!",
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
