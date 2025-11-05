import logging
import numpy as np
import json
import os
from datetime import datetime
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes

# Настройка логирования
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

# Файл для хранения данных
DATA_FILE = "bot_data.json"

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
        self.training_suggestions = []  # Предложения по улучшению
        self.pre_train()

    def load_knowledge(self):
        """Загружаем базу знаний - теперь пустая"""
        if os.path.exists(DATA_FILE):
            with open(DATA_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        else:
            # Возвращаем полностью пустую базу знаний
            return {}

    def save_knowledge(self, knowledge=None):
        """Сохраняем базу знаний"""
        if knowledge is None:
            knowledge = self.knowledge_base
        with open(DATA_FILE, 'w', encoding='utf-8') as f:
            json.dump(knowledge, f, ensure_ascii=False, indent=2)

    def add_knowledge(self, question, answer):
        """Добавляем новое знание"""
        self.knowledge_base[question.lower()] = answer
        self.save_knowledge()

    def pre_train(self):
        """Предварительное обучение на базовых примерах"""
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
        """Преобразуем текст в числовой вектор"""
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
        """Предсказываем категорию текста"""
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
        """Простое обучение на одном примере"""
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
        """Ищем ответ в базе знаний"""
        question_lower = question.lower()

        # Проверяем точное совпадение
        if question_lower in self.knowledge_base:
            return self.knowledge_base[question_lower]

        # Проверяем частичные совпадения
        for key, answer in self.knowledge_base.items():
            if key in question_lower or question_lower in key:
                return answer

        return None

    def generate_response(self, user_message, user_id, username):
        """Генерируем интеллектуальный ответ"""
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

        # Ищем ответ в базе знаний
        answer = self.find_answer(user_message)
        if answer:
            return answer

        # Если это вопрос, на который нет ответа
        if any(word in user_message.lower() for word in
               ['что такое', 'кто такой', 'как', 'почему', 'зачем', 'расскажи о', 'что значит']):
            return "Интересный вопрос! Пока у меня нет информации об этом. Админ может добавить ответ через админ-панель. Или используйте /suggest чтобы предложить тему для обучения!"

        # Классифицируем и отвечаем
        predictions = self.predict(user_message)
        main_category = predictions[0]['category']

        responses = {
            "техническая поддержка": "Понимаю, у вас технический вопрос. Опишите проблему подробнее, и я постараюсь помочь!",
            "продажи и маркетинг": "Интересуетесь нашими продуктами? Расскажите, что именно вас интересует!",
            "финансовые вопросы": "По финансовым вопросам готов предоставить информацию. Что конкретно вас интересует?",
            "жалобы и предложения": "Спасибо за обратную связь! Расскажите подробнее о вашем предложении или жалобе.",
            "информационные запросы": "С удовольствием отвечу на ваш вопрос! Что именно хотите узнать?",
            "сотрудничество": "Интересует сотрудничество? Расскажите, в какой области хотели бы сотрудничать!",
            "отзывы о продукте": "Буду рад услышать ваш отзыв! Поделитесь впечатлениями о продукте.",
            "трудоустройство": "По вопросам трудоустройства готов предоставить информацию!",
            "обучение и инструкции": "Нужна помощь с обучением? Расскажите, с чем возникли трудности!",
            "безопасность и конфиденциальность": "По вопросам безопасности готов предоставить информацию!",
            "обновления и новости": "Хотите узнать о последних обновлениях? Что именно интересует?",
            "другое": "Интересный вопрос! Пока у меня нет информации об этом. Админ может добавить ответ через админ-панель."
        }

        return responses.get(main_category,
                             "Интересный вопрос! Пока у меня нет информации об этом. Админ может добавить ответ через админ-панель.")


# Создаем нейросеть
nn = AdvancedNeuralNetwork()
admin_users = set()


# Команда /start
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    welcome_text = """🤖 Привет! Я нейросеть с искусственным интеллектом! Хочю предупредить!
Внимание при использовании нейросети вы соглашаетесь что мы будем собирать ваши данные из переписок для обучение модели!

Я умею:
• Отвечать на вопросы из моей базы знаний
• Классифицировать сообщения по категориям  
• Общаться на различные темы
• Понимать контекст и учиться новому

📋 Доступные команды:

/start - показать это сообщение
/suggest - предложить улучшение для нейросети
/prompt - выполнить специальный запрос

💡 Просто напиши мне любой вопрос или сообщение, и я постараюсь помочь!
Удачи в использовании!"""

    await update.message.reply_text(welcome_text)


# Обработка текстовых сообщений
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_text = update.message.text
    user_id = update.message.from_user.id
    username = update.message.from_user.username or f"user_{user_id}"

    # Генерируем интеллектуальный ответ
    response = nn.generate_response(user_text, user_id, username)

    await update.message.reply_text(response)


# Команда для предложений по улучшению (вместо /train)
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

    # Сохраняем предложение
    nn.training_suggestions.append({
        'user_id': user_id,
        'username': username,
        'suggestion': suggestion,
        'timestamp': datetime.now().isoformat()
    })

    await update.message.reply_text(
        "✅ Спасибо за предложение!\n"
        "Админ рассмотрит его и улучшит мои способности. "
        "Вы помогаете мне становиться умнее! 🚀"
    )


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
            "/admin train - применить обучение"
        )
        return

    if context.args[0] == "Password":
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
            "Админка была предоставлена по поролю и сохранится до перезапуска бота!"
        )
        return

    if user_id not in admin_users:
        await update.message.reply_text("❌ Доступ запрещен!")
        return

    # Статистика
    if context.args[0] == "stats":
        total_messages = len(nn.conversation_history)
        unique_users = len(set(msg['user_id'] for msg in nn.conversation_history))
        knowledge_size = len(nn.knowledge_base)
        suggestions_count = len(nn.training_suggestions)

        await update.message.reply_text(
            f"📊 Статистика:\n"
            f"Сообщений: {total_messages}\n"
            f"Уникальных пользователей: {unique_users}\n"
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

        history_text = f"📝 Последние {limit} сообщений:\n\n"
        for msg in nn.conversation_history[-limit:]:
            history_text += f"👤 {msg['username']}: {msg['message']}\n"
            history_text += f"⏰ {msg['timestamp'][:19]}\n\n"

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

    # Экспорт
    elif context.args[0] == "export":
        nn.save_knowledge()
        await update.message.reply_text("✅ Данные экспортированы в файл!")


# Команда /prompt
async def custom_prompt(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text(
            "🎯 Кастомный промт\n\n"
            "Использование: /prompt ваш запрос\n\n"
            "Пример: /prompt напиши план развития бизнеса"
        )
        return

    user_prompt = ' '.join(context.args)

    # Теперь ответы берутся из базы знаний
    answer = nn.find_answer(user_prompt)
    if answer:
        await update.message.reply_text(answer)
    else:
        await update.message.reply_text(
            "Пока не могу ответить на этот запрос. Админ может добавить информацию через админ-панель.")


# Основная функция
def main():
    TOKEN = "token"

    application = Application.builder().token(TOKEN).build()

    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("suggest", suggest_improvement))
    application.add_handler(CommandHandler("admin", admin_panel))
    application.add_handler(CommandHandler("prompt", custom_prompt))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    print("🤖 Умный бот запущен...")
    print("📚 База знаний: ПУСТА (будет заполняться через админку)")
    print("💡 Используйте /admin add для обучения бота")
    application.run_polling()


if __name__ == "__main__":
    main()
