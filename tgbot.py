import json
import telebot
from telebot import types
from telebot.types import ReplyKeyboardMarkup, KeyboardButton
from rag_telegram import ask_question

with open('config.json', 'r') as f:
    config = json.load(f)

# Параметры из конфига
TOKEN = config['telegram_token']
bot = telebot.TeleBot(TOKEN)

# Создает кнопку 'Привет' и приветствует пользователя
@bot.message_handler(commands=['start'])
def start_handler(message):
    user_id = message.chat.id
    markup = ReplyKeyboardMarkup(resize_keyboard=True, one_time_keyboard=True)
    markup.add(KeyboardButton("Привет"))
    bot.send_message(user_id, "Привет! ", reply_markup=markup)

# Словарь для хранения состояний пользователей
state = {}

# Переводит пользователя в режим вопросов
@bot.message_handler(func=lambda message: message.text == "Привет")
def ask_no_program(message):
  user_id = message.chat.id
  state[user_id] = {'step': 'waiting_for_question'}
  bot.send_message(user_id, "Я твой помощник, задавай вопрос", parse_mode="HTML")

# Получает ответ через функцию ask_question из  и отправляет пользователю
@bot.message_handler(func=lambda message: (state.get(message.chat.id, {}).get("step") == "waiting_for_question"))
def handle_question_for_program(message):
  user_id = message.chat.id
  state = state.get(user_id)
  question = message.text
  try:
    answer = ask_question(question)
    bot.send_message(user_id, answer)
  except Exception as e:
    bot.send_message(user_id, f"Произошла ошибка при получении ответа: {str(e)}")

bot.polling()