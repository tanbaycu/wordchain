import os
import telebot
import sqlite3
import google.generativeai as genai
import asyncio
from telebot.async_telebot import AsyncTeleBot
import random
import logging
from tenacity import retry, stop_after_attempt, wait_exponential
import uuid
from collections import Counter, defaultdict
from pybloom_live import ScalableBloomFilter
from functools import lru_cache
import nltk
from nltk.util import ngrams
import numpy as np

# Download necessary NLTK data
nltk.download('punkt', quiet=True)

# Configuration
TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
GEMINI_API_KEY = os.getenv('GOOGLE_API_KEY')

# Logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Gemini Configuration
os.environ['GOOGLE_API_KEY'] = GEMINI_API_KEY
genai.configure(api_key=GEMINI_API_KEY)

model = genai.GenerativeModel(model_name="gemini-2.0-flash-exp")

# Initialize Telegram bot
bot = AsyncTeleBot(TELEGRAM_BOT_TOKEN)

# Connect to SQLite database
conn = sqlite3.connect('word_game.db', check_same_thread=False)
cursor = conn.cursor()

# Create tables
cursor.execute('''
CREATE TABLE IF NOT EXISTS game_sessions
(id TEXT PRIMARY KEY, player_id INTEGER, status TEXT, 
start_time DATETIME DEFAULT CURRENT_TIMESTAMP)
''')

cursor.execute('''
CREATE TABLE IF NOT EXISTS used_words
(id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT, word TEXT, user TEXT, 
timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)
''')

conn.commit()

# Game state dictionary
game_states = {}

# Global word cache
word_cache = Counter()

# Bloom filter for quick membership testing
bloom_filter = ScalableBloomFilter(mode=ScalableBloomFilter.SMALL_SET_GROWTH)

# Trie for efficient prefix matching
class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end = False

class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word):
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.is_end = True

    def search_prefix(self, prefix):
        node = self.root
        for char in prefix:
            if char not in node.children:
                return []
            node = node.children[char]
        return self._dfs(node, prefix)

    def _dfs(self, node, prefix):
        result = []
        if node.is_end:
            result.append(prefix)
        for char, child in node.children.items():
            result.extend(self._dfs(child, prefix + char))
        return result

word_trie = Trie()

# Load initial words into trie and bloom filter
def load_initial_words():
    with open('vietnamese_words.txt', 'r', encoding='utf-8') as f:
        for word in f:
            word = word.strip().lower()
            if len(word.split()) == 2:
                word_trie.insert(word)
                bloom_filter.add(word)

load_initial_words()

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
async def get_gemini_response(last_word, used_words, session_id):
    # Get word usage statistics
    word_stats = get_word_usage_stats(session_id)

    # Generate context based on recent words
    context = generate_context(used_words)

    prompt = f"""
    Bạn là một chuyên gia ngôn ngữ học tiếng Việt bậc thầy với kiến thức sâu rộng về văn hóa, lịch sử, và xã hội Việt Nam. Bạn đang tham gia một trò chơi nối từ tiếng Việt cao cấp.

    Nhiệm vụ: Đưa ra một từ tiếng Việt hợp lệ và sáng tạo dựa trên các quy tắc sau:

    1. Cấu trúc: Từ PHẢI có đúng 2 âm tiết (2 chữ).
    2. Quy tắc nối: Từ PHẢI bắt đầu bằng âm tiết cuối cùng của từ "{last_word.lower()}".
    3. Ngữ nghĩa: Từ PHẢI có nghĩa xác định, phổ biến trong tiếng Việt hiện đại và có độ sâu về mặt ý nghĩa.
    4. Phạm vi sử dụng: 
       - Danh từ: Có thể chỉ sự vật, hiện tượng, khái niệm trừu tượng (ví dụ: tâm hồn, quê hương)
       - Động từ: Diễn tả hành động hoặc trạng thái (ví dụ: suy ngẫm, phát triển)
       - Tính từ: Miêu tả tính chất, đặc điểm (ví dụ: tinh tế, sâu sắc)
       - Trạng từ: Bổ sung ý nghĩa cho động từ hoặc tính từ (ví dụ: mạnh mẽ, nhẹ nhàng)
    5. Hạn chế:
       - KHÔNG sử dụng tên riêng, địa danh, tổ chức
       - KHÔNG dùng từ viết tắt, từ mượn chưa Việt hóa hoàn toàn
       - KHÔNG sử dụng biệt ngữ, từ lóng, hay từ địa phương
    6. Đa dạng: Ưu tiên từ đa nghĩa, có thể sử dụng trong nhiều ngữ cảnh, lĩnh vực khác nhau.
    7. Độc đáo: Tránh lặp lại các từ đã xuất hiện. Thống kê sử dụng từ: {word_stats}
    8. Văn hóa: Ưu tiên các từ phản ánh văn hóa, tư duy, hoặc đặc trưng của người Việt.
    9. Tính ứng dụng: Từ có thể sử dụng trong văn nói, văn viết, hoặc trong các tác phẩm văn học.
    10. Độ khó: Thử thách người chơi bằng cách sử dụng các từ có độ khó vừa phải đến cao, nhưng vẫn đảm bảo có thể nối tiếp được.
    11. Ý nghĩa: PHẢI đảm bảo từ được chọn có ý nghĩa rõ ràng và phù hợp trong tiếng Việt.
    12. Ngữ cảnh: Cân nhắc ngữ cảnh hiện tại của trò chơi: {context}

    Hãy suy nghĩ sâu sắc và đưa ra một từ độc đáo, có ý nghĩa sâu rộng, thể hiện sự tinh tế của tiếng Việt.

    Chỉ trả lời bằng một từ 2 âm tiết duy nhất, không kèm theo bất kỳ giải thích nào.

    Từ nối tiếp cho "{last_word.lower()}":
    """

    try:
        response = await asyncio.to_thread(model.generate_content, prompt)
        return response.text.strip().lower()
    except Exception as e:
        logger.error(f"Error generating Gemini response: {e}")
        raise

@lru_cache(maxsize=1000)
def get_word_usage_stats(session_id):
    cursor.execute("""
    SELECT word, COUNT(*) as count
    FROM used_words
    WHERE session_id = ?
    GROUP BY word
    ORDER BY count DESC
    LIMIT 20
    """, (session_id,))
    stats = cursor.fetchall()
    return ", ".join([f"{word}: {count}" for word, count in stats])

def generate_context(used_words):
    # Generate bigrams from the last 10 words
    recent_words = used_words[-10:]
    bigrams = list(ngrams(recent_words, 2))
    
    # Count bigram frequencies
    bigram_freq = Counter(bigrams)
    
    # Find the most common bigram
    if bigram_freq:
        most_common = bigram_freq.most_common(1)[0][0]
        return f"Recent context: {' '.join(most_common)}"
    else:
        return "No strong context detected"

def weighted_random_selection(words, weights):
    return random.choices(words, weights=weights, k=1)[0]

def suggest_word(prefix):
    # First, check the trie for words with the given prefix
    suggestions = word_trie.search_prefix(prefix)
    
    if suggestions:
        # Filter out words that are in the bloom filter (likely used before)
        new_suggestions = [word for word in suggestions if word not in bloom_filter]
        
        if new_suggestions:
            return random.choice(new_suggestions)
    
    # If no suitable word found in trie, fall back to Gemini
    return None

def is_valid_word(word, last_word, used_words):
    word = word.lower()
    last_word = last_word.lower()
    
    if len(word.split()) != 2:
        return False, "Từ phải có đúng 2 âm tiết."
    
    if word.split()[0] != last_word.split()[-1]:
        return False, f"Từ phải bắt đầu bằng '{last_word.split()[-1]}'."
    
    if word.lower() in [w.lower() for w in used_words]:
        return False, "Từ này đã được sử dụng trước đó."
    
    return True, ""

async def bot_turn(message):
    user_id = message.from_user.id
    game_state = game_states.get(user_id)
    if not game_state:
        await bot.reply_to(message, "❌ Không có trò chơi đang diễn ra. Hãy bắt đầu một trò chơi mới với /noitu.")
        return

    thinking_message = await bot.reply_to(message, "🤔 Bot đang suy nghĩ...")
    await asyncio.sleep(2)  # Reduced waiting time

    last_word = game_state['last_word']
    session_id = game_state['session_id']
    used_words = game_state['used_words']

    prefix = last_word.split()[-1]
    
    # Try to suggest a word from the trie first
    bot_word = suggest_word(prefix)
    
    if not bot_word:
        # If no suitable word found in trie, use Gemini
        bot_word = await get_gemini_response(last_word, used_words, session_id)
    
    valid, error_message = is_valid_word(bot_word, last_word, used_words)
    if valid:
        game_state['last_word'] = bot_word
        game_state['turn'] = 'player'
        game_state['used_words'].append(bot_word)

        # Update global word cache and bloom filter
        word_cache[bot_word] += 1
        bloom_filter.add(bot_word)

        # Log the bot's word
        await log_used_word(session_id, bot_word, 'bot')

        # Delete the "thinking" message
        await bot.delete_message(message.chat.id, thinking_message.message_id)

        # Create response message with random emoji
        emoji_list = ["🌟", "💡", "🎨", "🔍", "🧠", "📚", "🌈", "🌺", "🍀", "🌙"]
        random_emoji = random.choice(emoji_list)
        response_message = (
            f"{random_emoji} Bot đã tìm ra từ: <b>{bot_word}</b>\n\n"
            f"🔄 Đến lượt bạn! Hãy đưa ra một từ bắt đầu bằng '<b>{bot_word.split()[-1]}</b>'."
        )
        await bot.reply_to(message, response_message, parse_mode="HTML")
        logger.info(f"Bot used word: {bot_word}")
    else:
        await bot.delete_message(message.chat.id, thinking_message.message_id)
        await bot.reply_to(message, f"❌ Bot không thể tìm ra từ hợp lệ bắt đầu bằng '{last_word.split()[-1]}'. Bạn đã chiến thắng! 🎉🏆")
        await end_game(user_id, 'player')

async def log_used_word(session_id, word, user):
    cursor.execute("INSERT INTO used_words (session_id, word, user) VALUES (?, ?, ?)", (session_id, word.lower(), user))
    conn.commit()
    logger.info(f"Word used: {word.lower()} by {user} in session {session_id}")

@bot.message_handler(commands=['start'])
async def send_welcome(message):
    welcome_message = "🎉 Chào mừng đến với trò chơi Nối Từ Tiếng Việt!\n\n"
    welcome_message += "Để bắt đầu, hãy sử dụng lệnh: `/noitu <từ khởi đầu>`\n"
    welcome_message += "Ví dụ: `/noitu con cò`\n\n"
    welcome_message += "Để dừng trò chơi, sử dụng lệnh: `/stop`"
    await bot.reply_to(message, welcome_message, parse_mode='Markdown')

def is_valid_start_word(word):
    if len(word.split()) != 2:
        return False, "Từ khởi đầu phải có đúng 2 âm tiết."
    return True, ""

@bot.message_handler(commands=['noitu'])
async def start_game(message):
    user_id = message.from_user.id
    if len(message.text.split()) < 2:
        await bot.reply_to(message, "❌ Vui lòng nhập một từ để bắt đầu trò chơi. Ví dụ: `/noitu con cò`", parse_mode='Markdown')
        return

    word = ' '.join(message.text.split()[1:]).strip().lower()
    valid, error_message = is_valid_start_word(word)
    if not valid:
        await bot.reply_to(message, f"❌ {error_message} Vui lòng thử lại với một từ hợp lệ.")
        return

    # Create new game session
    session_id = str(uuid.uuid4())
    cursor.execute("INSERT INTO game_sessions (id, player_id, status) VALUES (?, ?, 'active')", (session_id, user_id))
    conn.commit()

    # Save game state
    game_states[user_id] = {
        'session_id': session_id,
        'last_word': word,
        'turn': 'bot',
        'used_words': [word]
    }

    # Log the starting word
    await log_used_word(session_id, word, 'player')

    await bot.reply_to(message, f"🎮 Trò chơi bắt đầu với từ '{word}'.\n🤖 Đến lượt bot.")
    await bot_turn(message)

@bot.message_handler(func=lambda message: True)
async def player_turn(message):
    user_id = message.from_user.id
    game_state = game_states.get(user_id)
    if not game_state or game_state['turn'] != 'player':
        return

    word = message.text.strip().lower()
    if word == '/stop':
        await stop_game(message)
        return
    last_word = game_state['last_word']
    session_id = game_state['session_id']
    used_words = game_state['used_words']

    valid, error_message = is_valid_word(word, last_word, used_words)
    if valid:
        # Log the valid word
        await log_used_word(session_id, word, 'player')

        # Update game state
        game_state['last_word'] = word
        game_state['turn'] = 'bot'
        game_state['used_words'].append(word)

        response_message = (
            f"✅ Tuyệt vời! Từ '<b>{word}</b>' của bạn hợp lệ.\n\n"
            f"🤖 Đến lượt Bot..."
        )
        await bot.reply_to(message, response_message, parse_mode="HTML")
        await bot_turn(message)
    else:
        error_message = (
            f"❌ Rất tiếc, từ '<b>{word}</b>' không hợp lệ.\n"
            f"Lý do: {error_message}\n"
            f"Hãy thử lại với một từ 2 âm tiết bắt đầu bằng '<b>{last_word.split()[-1]}</b>'."
        )
        await bot.reply_to(message, error_message, parse_mode="HTML")

async def end_game(user_id, winner):
    game_state = game_states.get(user_id)
    if game_state:
        session_id = game_state['session_id']
        cursor.execute("UPDATE game_sessions SET status = 'finished' WHERE id = ?", (session_id,))
        conn.commit()
        
        result_message = f"🏁 Trò chơi kết thúc!\n\n"
        result_message += f"🏆 Người thắng: {winner.capitalize()}\n\n"
        result_message += "🔄 Bạn có muốn chơi lại không? Sử dụng `/noitu` để bắt đầu trò chơi mới."
        await bot.send_message(user_id, result_message, parse_mode='Markdown')
        
        del game_states[user_id]

@bot.message_handler(commands=['stop'])
async def stop_game(message):
    user_id = message.from_user.id
    if user_id in game_states:
        await end_game(user_id, 'Hòa')
        await bot.reply_to(message, "🛑 Trò chơi đã được dừng lại.")
    else:
        await bot.reply_to(message, "❌ Không có trò chơi đang diễn ra để dừng.")

# Background task to analyze word usage patterns
async def analyze_word_patterns():
    while True:
        try:
            cursor.execute("""
            SELECT word, COUNT(*) as count
            FROM used_words
            GROUP BY word
            ORDER BY count DESC
            LIMIT 100
            """)
            common_words = cursor.fetchall()
            
            # Update word_cache with the most common words
            word_cache.clear()
            word_cache.update(dict(common_words))
            
            # Sleep for an hour before the next analysis
            await asyncio.sleep(3600)
        except Exception as e:
            logger.error(f"Error in word pattern analysis: {e}")
            await asyncio.sleep(600)  # Sleep for 10 minutes before retrying

# Main function to run the bot
async def main():
    try:
        # Start the background task
        asyncio.create_task(analyze_word_patterns())
        
        # Start polling
        await bot.polling(non_stop=True)
    except Exception as e:
        logger.error(f"Error in main polling loop: {e}")
        await asyncio.sleep(10)  # Wait before attempting to reconnect

if __name__ == '__main__':
    while True:
        try:
            asyncio.run(main())
        except Exception as e:
            logger.critical(f"Critical error: {e}")
            logger.info("Restarting the bot in 60 seconds...")
            asyncio.sleep(60)
            