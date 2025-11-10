import logging
import numpy as np
import json
import os
import re
import math
from datetime import datetime
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes

# Настройка логирования
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

# Файлы для хранения данных
DATA_FILE = "bot_data.json"
USERS_FILE = "users.txt"
SENDS_FILE = "sends.txt"

# 12 промтов для классификации
PROMPTS = [
    "техническая поддержка",
    "продажи и маркетинг",
    "финансовые вопросы",
    "жалобы и предложения",
    "информационные запросы",
    "сотрудничество",
    "отзывы о продукте",
    "трудоустройство",
    "обучение и инструкции",
    "безопасность и конфиденциальность",
    "обновления и новости",
    "другое"
]


class AdvancedNeuralNetwork:
    def __init__(self):
        self.weights = np.random.randn(len(PROMPTS), 100) * 0.1
        self.bias = np.zeros(len(PROMPTS))
        self.knowledge_base = self.load_knowledge()
        self.conversation_history = []
        self.training_suggestions = []
        self.pre_train()

    def load_knowledge(self):
        if os.path.exists(DATA_FILE):
            with open(DATA_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        else:
            return {}

    def save_knowledge(self, knowledge=None):
        if knowledge is None:
            knowledge = self.knowledge_base
        with open(DATA_FILE, 'w', encoding='utf-8') as f:
            json.dump(knowledge, f, ensure_ascii=False, indent=2)

    def add_knowledge(self, question, answer):
        self.knowledge_base[question.lower()] = answer
        self.save_knowledge()

    def pre_train(self):
        training_examples = [
            ("привет", "другое"),
            ("здравствуйте", "другое"),
            ("добрый день", "другое"),
            ("hi", "другое"),
            ("hello", "другое"),
            ("как дела", "другое"),
            ("что делаешь", "другое"),
            ("не работает", "техническая поддержка"),
            ("ошибка", "техническая поддержка"),
            ("помогите настроить", "техническая поддержка"),
            ("сломалось", "техническая поддержка"),
            ("купить", "продажи и маркетинг"),
            ("цена", "продажи и маркетинг"),
            ("стоимость", "продажи и маркетинг"),
            ("заказать", "продажи и маркетинг"),
            ("как работает", "информационные запросы"),
            ("что это", "информационные запросы"),
            ("расскажите о", "информационные запросы"),
            ("жалоба", "жалобы и предложения"),
            ("недоволен", "жалобы и предложения"),
            ("предложение", "жалобы и предложения"),
        ]

        for text, category in training_examples:
            for _ in range(5):
                self.train_on_example(text, category, learning_rate=0.3)

    def preprocess_text(self, text):
        text = text.lower()
        vector = np.zeros(100)
        words = text.split()

        for i, word in enumerate(words[:100]):
            hash_val = hash(word) % 100
            vector[hash_val] += 1

        if np.linalg.norm(vector) > 0:
            vector = vector / np.linalg.norm(vector)

        return vector

    def predict(self, text):
        vector = self.preprocess_text(text)
        scores = np.dot(self.weights, vector) + self.bias

        exp_scores = np.exp(scores - np.max(scores))
        probabilities = exp_scores / np.sum(exp_scores)

        top_3_indices = np.argsort(probabilities)[-3:][::-1]

        results = []
        for idx in top_3_indices:
            results.append({
                'category': PROMPTS[idx],
                'probability': float(probabilities[idx])
            })

        return results

    def train_on_example(self, text, correct_category, learning_rate=0.1):
        vector = self.preprocess_text(text)
        scores = np.dot(self.weights, vector) + self.bias

        correct_idx = PROMPTS.index(correct_category)

        for i in range(len(PROMPTS)):
            if i == correct_idx:
                self.weights[i] += learning_rate * vector
                self.bias[i] += learning_rate
            else:
                self.weights[i] -= learning_rate * vector * 0.1
                self.bias[i] -= learning_rate * 0.1

    def find_answer(self, question):
        question_lower = question.lower()

        if question_lower in self.knowledge_base:
            return self.knowledge_base[question_lower]

        for key, answer in self.knowledge_base.items():
            if key in question_lower or question_lower in key:
                return answer

        return None

    def calculate_math_expression(self, expression):
        """Вычисляет математические выражения"""
        try:
            # Безопасное вычисление математических выражений
            expression = expression.replace(' ', '').replace('×', '*').replace('÷', '/')

            # Проверяем на наличие только разрешенных символов
            if not re.match(r'^[0-9+\-*/().\s]+$', expression):
                return None

            # Заменяем ** на ^ для степени
            expression = expression.replace('^', '**')

            # Вычисляем выражение
            result = eval(expression, {"__builtins__": None}, {
                'sin': math.sin, 'cos': math.cos, 'tan': math.tan,
                'sqrt': math.sqrt, 'log': math.log, 'log10': math.log10,
                'pi': math.pi, 'e': math.e
            })

            return f"{expression} = {result}"

        except Exception as e:
            return None

    def split_into_phrases(self, text):
        """Разделяет текст на отдельные фразы - пробел как разделитель"""
        # Просто разбиваем по пробелам
        phrases = text.split()

        # Очищаем фразы от лишних пробелов и фильтруем пустые
        cleaned_phrases = []
        for phrase in phrases:
            phrase = phrase.strip()
            if len(phrase) > 1:  # Минимум 2 символа
                cleaned_phrases.append(phrase)

        return cleaned_phrases

    def generate_response(self, user_message, user_id, username):
        # Сохраняем в историю
        self.conversation_history.append({
            'user_id': user_id,
            'username': username,
            'message': user_message,
            'timestamp': datetime.now().isoformat()
        })

        # Ограничиваем историю
        if len(self.conversation_history) > 1000:
            self.conversation_history = self.conversation_history[-1000:]

        # Разделяем сообщение на отдельные слова по пробелам
        phrases = self.split_into_phrases(user_message)
        responses = []

        # Обрабатываем КАЖДОЕ слово отдельно
        for phrase in phrases:
            # Пропускаем слишком короткие фразы
            if len(phrase) < 2:
                continue

            # Сначала проверяем математическое выражение
            math_result = self.calculate_math_expression(phrase)
            if math_result:
                responses.append(f"🔢 {math_result}")
                continue

            # Ищем ответ в базе знаний для ЭТОГО слова
            answer = self.find_answer(phrase)
            if answer:
                responses.append(f"{answer}")
                continue

            # Если это вопрос, на который нет ответа
            if any(word in phrase.lower() for word in
                   ['что такое', 'кто такой', 'как', 'почему', 'зачем', 'расскажи о', 'что значит']):
                responses.append(
                    f"❓ '{phrase}' - интересный вопрос! Пока у меня нет информации об этом. Используйте /suggest чтобы предложить тему для обучения!")
                continue

            # Классифицируем и отвечаем
            predictions = self.predict(phrase)
            main_category = predictions[0]['category']

            category_responses = {
                "техническая поддержка": f"🔧 По техническому вопросу '{phrase}' - опишите проблему подробнее!",
                "продажи и маркетинг": f"💰 По вопросу покупки '{phrase}' - расскажите, что именно интересует!",
                "финансовые вопросы": f"💳 Финансовый вопрос '{phrase}' - что конкретно вас интересует?",
                "жалобы и предложения": f"📝 По вашему предложению '{phrase}' - расскажите подробнее!",
                "информационные запросы": f"📚 Информационный запрос '{phrase}' - пока нет информации, но админ может её добавить!",
                "сотрудничество": f"🤝 Вопрос сотрудничества '{phrase}' - в какой области хотите сотрудничать?",
                "отзывы о продукте": f"⭐ Отзыв о продукте '{phrase}' - поделитесь впечатлениями!",
                "трудоустройство": f"👔 Вопрос трудоустройства '{phrase}' - расскажите подробнее!",
                "обучение и инструкции": f"🎓 Обучение по вопросу '{phrase}' - с чем нужна помощь?",
                "безопасность и конфиденциальность": f"🔐 Вопрос безопасности '{phrase}' - что именно интересует?",
                "обновления и новости": f"📢 Вопрос обновлений '{phrase}' - пока нет информации!",
                "другое": f"💭 '{phrase}' - интересный вопрос! Пока нет информации, но админ может её добавить!"
            }

            responses.append(category_responses.get(main_category,
                                                    f"❓ '{phrase}' - интересный вопрос! Пока у меня нет информации об этом."))

        # Объединяем все ответы через разделитель
        if responses:
            if len(responses) == 1:
                return responses[0]
            else:
                # Объединяем все ответы через разделитель " | "
                combined_response = " | ".join(responses)
                return combined_response
        else:
            return "🤔 Не совсем понял ваш вопрос. Можете переформулировать?"


# Функции для работы с пользователями и логами
def save_user(user_id, username):
    """Сохраняет пользователя в файл users.txt"""
    try:
        # Читаем существующих пользователей
        if os.path.exists(USERS_FILE):
            with open(USERS_FILE, 'r', encoding='utf-8') as f:
                existing_users = set(line.strip() for line in f.readlines())
        else:
            existing_users = set()

        # Формируем запись
        user_record = f"{user_id}:{username}"

        # Добавляем нового пользователя если его нет
        if user_record not in existing_users:
            with open(USERS_FILE, 'a', encoding='utf-8') as f:
                f.write(user_record + '\n')
    except Exception as e:
        logging.error(f"Error saving user {username}: {e}")


def get_all_users():
    """Возвращает список всех пользователей (user_id)"""
    try:
        if os.path.exists(USERS_FILE):
            with open(USERS_FILE, 'r', encoding='utf-8') as f:
                users = []
                for line in f.readlines():
                    if ':' in line:
                        user_id = line.strip().split(':')[0]
                        users.append(user_id)
                return users
        return []
    except Exception as e:
        logging.error(f"Error reading users: {e}")
        return []


def save_message_log(username, user_message, bot_response):
    """Сохраняет переписку в sends.txt"""
    try:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(SENDS_FILE, 'a', encoding='utf-8') as f:
            f.write(f"[{timestamp}] {username}: {user_message}\n")
            f.write(f"[{timestamp}] Бот: {bot_response}\n")
            f.write("-" * 50 + "\n")
    except Exception as e:
        logging.error(f"Error saving message log: {e}")


def get_message_history(limit=10):
    """Возвращает историю сообщений"""
    try:
        if os.path.exists(SENDS_FILE):
            with open(SENDS_FILE, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                # Каждое сообщение занимает 3 строки
                start_index = max(0, len(lines) - limit * 3)
                return lines[start_index:]
        return []
    except Exception as e:
        logging.error(f"Error reading message history: {e}")
        return []


# Создаем нейросеть
nn = AdvancedNeuralNetwork()
admin_users = set()


# Команда /start
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    welcome_text = """🤖 Привет! Я улучшенная нейросеть с искусственным интеллектом!
    Хочу предупредить!
Внимание при использовании нейросети вы соглашаетесь что мы будем собирать ваши данные из переписок для обучение модели! Но также предупреждаем что сбор данных начн1тся только если вы напишете следущее сообщение!

✨ Новые возможности:
• 🧮 Решаю математические примеры (2+2, 5*3, 10/2)
• 📝 Понимаю составные вопросы (привет а потом спроси про погоду)
• 🔍 Отвечаю на несколько вопросов в одном сообщении
• 💭 Стал более умным и понятливым

📋 Доступные команды:

/start - показать это сообщение
/suggest - предложить улучшение для нейросети

💡 Примеры использования:
• "2+2" - получите ответ 4
• "привет расскажи о Python а потом о Java" - ответит на оба вопроса через разделитель
• "сколько будет 5*8 и что такое ИИ" - математика + информация
• "привет пока" - ответит на обе фразы из базы знаний через разделитель

Удачи в использовании! 🚀"""

    await update.message.reply_text(welcome_text)


# Обработка текстовых сообщений
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_text = update.message.text
    user_id = update.message.from_user.id
    username = update.message.from_user.username or f"user_{user_id}"

    # Сохраняем пользователя
    save_user(user_id, username)

    # Генерируем ответ
    response = nn.generate_response(user_text, user_id, username)

    # Сохраняем переписку
    save_message_log(username, user_text, response)

    await update.message.reply_text(response)


# Команда для предложений по улучшению
async def suggest_improvement(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text(
            "💡 Предложение по улучшению\n\n"
            "Использование: /suggest ваш текст\n\n"
            "Примеры:\n"
            "/suggest научись отвечать на вопросы о программировании\n"
            "/suggest добавь информацию об искусственном интеллекте\n"
            "/suggest научись распознавать технические термины\n\n"
            "Админ рассмотрит ваши предложения и улучшит бота!"
        )
        return

    suggestion = ' '.join(context.args)
    user_id = update.message.from_user.id
    username = update.message.from_user.username or f"user_{user_id}"

    # Сохраняем предложение в список
    nn.training_suggestions.append({
        'user_id': user_id,
        'username': username,
        'suggestion': suggestion,
        'timestamp': datetime.now().isoformat()
    })

    # Сохраняем предложение в файл idea.txt
    with open('idea.txt', 'a', encoding='utf-8') as f:
        f.write(f"{datetime.now().isoformat()} | {username} | {suggestion}\n")

    await update.message.reply_text(
        "✅ Спасибо за предложение!\n"
        "Админ рассмотрит его и улучшит мои способности. "
        "Вы помогаете мне становиться умнее! 🚀"
    )


# Функция для рассылки сообщений
async def broadcast_message(context: ContextTypes.DEFAULT_TYPE, message: str):
    """Отправляет сообщение всем пользователям из users.txt"""
    users = get_all_users()
    successful = 0
    failed = 0

    for user_id in users:
        try:
            await context.bot.send_message(
                chat_id=int(user_id),
                text=message
            )
            successful += 1
            logging.info(f"Message sent to {user_id}")
        except Exception as e:
            logging.error(f"Failed to send message to {user_id}: {e}")
            failed += 1

    return successful, failed


# Админ панель
async def admin_panel(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.message.from_user.id

    if not context.args:
        await update.message.reply_text(
            "🔐 Админ панель\n\n"
            "Введите пароль: /admin пароль\n\n"
            "Доступные команды:\n"
            "/admin stats - статистика\n"
            "/admin history - история сообщений\n"
            "/admin suggestions - предложения пользователей\n"
            "/admin add знание - добавить знание\n"
            "/admin delete знание - удалить знание\n"
            "/admin knowledge - просмотр знаний\n"
            "/admin train - применить обучение\n"
            "/admin closed - уведомить о выключении\n"
            "/admin send - рассылка всем пользователям"
        )
        return

    if context.args[0] == "Введите сюда пороль для админа":
        admin_users.add(user_id)
        await update.message.reply_text(
            "✅ Доступ предоставлен!\n\n"
            "Админ команды:\n"
            "/admin stats - статистика\n"
            "/admin history [N] - последние N сообщений\n"
            "/admin suggestions - предложения пользователей\n"
            "/admin add вопрос::ответ - добавить знание\n"
            "/admin delete вопрос - удалить знание\n"
            "/admin knowledge - вся база знаний\n"
            "/admin train категория текст - обучение модели\n"
            "/admin closed - уведомить всех о выключении\n"
            "/admin send текст - рассылка всем пользователям\n"
            "Админка была предоставлена по поролю и сохранится до перезапуска бота!"
        )
        return

    if user_id not in admin_users:
        await update.message.reply_text("❌ Доступ запрещен!")
        return

    # Статистика
    if context.args[0] == "stats":
        users_count = len(get_all_users())
        knowledge_size = len(nn.knowledge_base)
        suggestions_count = len(nn.training_suggestions)
        total_messages = len(nn.conversation_history)

        await update.message.reply_text(
            f"📊 Статистика:\n"
            f"Зарегистрированных пользователей: {users_count}\n"
            f"Всего сообщений: {total_messages}\n"
            f"Знаний в базе: {knowledge_size}\n"
            f"Предложений: {suggestions_count}\n"
            f"Админов онлайн: {len(admin_users)}"
        )

    # История сообщений
    elif context.args[0] == "history":
        limit = 10
        if len(context.args) > 1:
            try:
                limit = min(int(context.args[1]), 50)
            except:
                pass

        history_lines = get_message_history(limit)
        if history_lines:
            history_text = f"📝 Последние {limit} сообщений:\n\n"
            history_text += "".join(history_lines)
        else:
            history_text = "📭 История сообщений пуста"

        await update.message.reply_text(history_text[:4000])

    # Предложения пользователей
    elif context.args[0] == "suggestions":
        suggestions_text = "💡 Предложения пользователей:\n\n"
        for i, suggestion in enumerate(nn.training_suggestions[-20:]):
            suggestions_text += f"{i + 1}. 👤 {suggestion['username']}:\n"
            suggestions_text += f"   💬 {suggestion['suggestion']}\n"
            suggestions_text += f"   ⏰ {suggestion['timestamp'][:19]}\n\n"

        if not nn.training_suggestions:
            suggestions_text = "📝 Предложений пока нет"

        await update.message.reply_text(suggestions_text[:4000])

    # Добавление знаний
    elif context.args[0] == "add" and len(context.args) > 1:
        knowledge_text = ' '.join(context.args[1:])
        if "::" in knowledge_text:
            question, answer = knowledge_text.split("::", 1)
            nn.add_knowledge(question.strip(), answer.strip())
            await update.message.reply_text(f"✅ Знание добавлено!\nВопрос: {question}\nОтвет: {answer}")
        else:
            await update.message.reply_text("❌ Формат: /admin add вопрос::ответ")

    # Удаление знаний
    elif context.args[0] == "delete" and len(context.args) > 1:
        question = ' '.join(context.args[1:]).lower()
        if question in nn.knowledge_base:
            del nn.knowledge_base[question]
            nn.save_knowledge()
            await update.message.reply_text(f"✅ Знание '{question}' удалено!")
        else:
            await update.message.reply_text("❌ Такого знания нет в базе!")

    # Просмотр знаний
    elif context.args[0] == "knowledge":
        knowledge_text = "📚 База знаний:\n\n"
        if nn.knowledge_base:
            for i, (question, answer) in enumerate(list(nn.knowledge_base.items())[:20]):
                knowledge_text += f"{i + 1}. {question}: {answer[:50]}...\n"
            knowledge_text += f"\nВсего знаний: {len(nn.knowledge_base)}"
        else:
            knowledge_text = "📭 База знаний пуста. Используйте /admin add вопрос::ответ чтобы добавить знания."

        await update.message.reply_text(knowledge_text)

    # Обучение от админа
    elif context.args[0] == "train" and len(context.args) > 2:
        try:
            category_num = int(context.args[1])
            training_text = ' '.join(context.args[2:])
            correct_category = PROMPTS[category_num - 1]
            nn.train_on_example(training_text, correct_category)
            await update.message.reply_text(f"✅ Обучено: '{training_text}' -> {correct_category}")
        except:
            await update.message.reply_text("❌ Ошибка формата: /admin train номер текст")

    # Уведомление о выключении
    elif context.args[0] == "closed":
        message = "🔴 Внимание! Бот будет отключен на ночь (сервера отключаются для экономии энергии). Работа возобновится утром. Спасибо за понимание!"

        await update.message.reply_text("🔄 Начинаю рассылку уведомления о выключении...")
        successful, failed = await broadcast_message(context, message)

        await update.message.reply_text(
            f"✅ Уведомление о выключении отправлено!\n"
            f"✅ Успешно: {successful} пользователей\n"
            f"❌ Не удалось: {failed} пользователей"
        )

    # Рассылка сообщений
    elif context.args[0] == "send" and len(context.args) > 1:
        message_text = ' '.join(context.args[1:])

        await update.message.reply_text("🔄 Начинаю рассылку сообщения...")
        successful, failed = await broadcast_message(context, message_text)

        await update.message.reply_text(
            f"✅ Рассылка завершена!\n"
            f"✅ Успешно: {successful} пользователей\n"
            f"❌ Не удалось: {failed} пользователей\n"
            f"📨 Сообщение: {message_text}"
        )

    # Экспорт
    elif context.args[0] == "export":
        nn.save_knowledge()
        await update.message.reply_text("✅ Данные экспортированы в файл!")


# Основная функция
def main():
    TOKEN = "Введите сюда ваш токен!"

    application = Application.builder().token(TOKEN).build()

    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("suggest", suggest_improvement))
    application.add_handler(CommandHandler("admin", admin_panel))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    print("🤖 Умный бот запущен...")
    print("📚 База знаний загружена")
    print("👥 Система пользователей активна")
    print("📝 Логи сохраняются в sends.txt")
    print("💡 Используйте /admin для управления")
    application.run_polling()


if __name__ == "__main__":
    main()
