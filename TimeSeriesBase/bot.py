# # bot.py
# # token("").build()
BOT_TOKEN = 'BOT-KEY'

import asyncio
import time
import threading
import logging
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, CallbackQueryHandler, filters, ContextTypes
from asgiref.sync import sync_to_async
from django.contrib.auth import authenticate
from UserManagement.models import CustomUser
from TimeSeriesBase.models  import Topic, Category, Indicator, DataValue, DataPoint
import requests


logging.basicConfig(level=logging.INFO)

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await context.bot.send_message(chat_id=update.effective_chat.id, text="Hi! Use the /login command to authenticate.")

# Initiate the login process
async def login_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await context.bot.send_message(chat_id=update.effective_chat.id, text="Please enter your email:")
    context.user_data['state'] = 'waiting_for_email'

# Handle email input
async def handle_email(update: Update, context: ContextTypes.DEFAULT_TYPE):
    email = update.message.text
    user_exists = await sync_to_async(CustomUser.objects.filter(email=email).exists)()

    if user_exists:
        context.user_data['email'] = email
        await context.bot.send_message(chat_id=update.effective_chat.id, text="Email found. Please enter your password:")
        context.user_data['state'] = 'waiting_for_password'
    else:
        await context.bot.send_message(chat_id=update.effective_chat.id, text="Email not found. Please try again.")
        context.user_data['state'] = 'waiting_for_email'

# Handle password input and authenticate the user
async def handle_password(update: Update, context: ContextTypes.DEFAULT_TYPE):
    password = update.message.text
    email = context.user_data.get('email')

    # Authenticate the user
    user = await sync_to_async(authenticate)(username=email, password=password)

    if user is not None:
        await context.bot.send_message(chat_id=update.effective_chat.id, text=f"Hello, {user.username}!")
        context.user_data[update.message.from_user.id] = {'user_id': user.id, 'is_authenticated': True}
        await update.message.delete()
        context.user_data['state'] = None  # Reset state after successful login
    else:
        await context.bot.send_message(chat_id=update.effective_chat.id, text="Invalid password. Please try again.")
        context.user_data['state'] = 'waiting_for_password'

# Unified message handler to check for the current state and act accordingly
async def message_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    state = context.user_data.get('state')

    if state == 'waiting_for_email':
        await handle_email(update, context)
    elif state == 'waiting_for_password':
        await handle_password(update, context)
    else:
        await update.message.reply_text("Please use /login to authenticate.")

# Handle /users command to list users if authenticated
async def users(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_data = context.user_data.get(update.message.from_user.id)

    if user_data and user_data.get('is_authenticated'):
        users = await sync_to_async(list)(CustomUser.objects.all())
        user_list = '\n'.join([f'{user.id}: {user.username}' for user in users])
        await update.message.reply_text(f'User {update.effective_user.first_name}, here are the users:\n{user_list}')
    else:
        await update.message.reply_text('You need to log in first. Use /login.')

# Handle logout
async def logout(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.message.from_user.id

    if user_id in context.user_data:
        del context.user_data[user_id]
        await update.message.reply_text('You have been logged out successfully.')
    else:
        await update.message.reply_text('You are not logged in.')


import logging
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Updater, CommandHandler, CallbackQueryHandler, ContextTypes
from asgiref.sync import sync_to_async
import requests

# Set up logging
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

# Your Telegram bot token
BASE_URL = 'http://127.0.0.1:8000/'

async def filter_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_data = context.user_data.get(update.message.from_user.id)
    if user_data and user_data.get('is_authenticated'):
        topics = await sync_to_async(list)(Topic.objects.filter(is_deleted=False))
        if topics:
            keyboard = [[InlineKeyboardButton(topic.title_ENG, callback_data=f"topic_{topic.id}")] for topic in topics]
            reply_markup = InlineKeyboardMarkup(keyboard)
            await update.message.reply_text('Please choose a topic:', reply_markup=reply_markup)
        else:
            await update.message.reply_text('No topics available.')
    else:
        await update.message.reply_text('You need to log in first. Use /login')

async def handle_topic_selection(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    
    # Extract topic ID from callback data
    topic_id = int(query.data.split('_')[1])
    context.user_data['selected_topic'] = topic_id

    logger.info(f"Selected topic ID: {topic_id}")
    
    try:
        # Retrieve the selected topic (both English and Amharic names)
        selected_topic = await sync_to_async(Topic.objects.get)(id=topic_id)
        
        # Retrieve categories related to the selected topic
        categories = await sync_to_async(list)(
            Category.objects.filter(topic_id=topic_id, is_deleted=False)
        )

        # Display the topic name (in Amharic and/or English) instead of the ID
        if categories:
            keyboard = [
                [InlineKeyboardButton(category.name_ENG, callback_data=f"category_{category.id}")]
                for category in categories
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)
            await query.edit_message_text(
                f'ተመርጧል ርዕስ: {selected_topic.title_AMH} ({selected_topic.title_ENG})\nእባክዎን ምድብ ይምረጡ:',
                reply_markup=reply_markup
            )
        else:
            await query.edit_message_text(f'ርዕስ: {selected_topic.title_AMH} ({selected_topic.title_ENG})\nምድቦች አልተገኙም።')

    except Exception as e:
        await query.edit_message_text(f"Error: {str(e)}")

async def handle_category_selection(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    
    # Extract category ID from callback data
    category_id = int(query.data.split('_')[1])
    context.user_data['selected_category'] = category_id

    logger.info(f"Selected category ID: {category_id}")
    
    try:
        # Retrieve the selected topic (stored in user_data)
        selected_topic = await sync_to_async(Topic.objects.get)(id=context.user_data.get('selected_topic'))
        
        # Retrieve the selected category
        selected_category = await sync_to_async(Category.objects.get)(id=category_id)

        # Store names in user_data
        context.user_data['selected_topic_name'] = selected_topic.title_AMH
        context.user_data['selected_category_name'] = selected_category.name_AMH

        # Define keyboard options for data point types with Amharic labels
        keyboard = [
            [InlineKeyboardButton('ዓመታዊ', callback_data='data_point_yearly')],
            [InlineKeyboardButton('በሩብ በሩብ', callback_data='data_point_quarterly')],
            [InlineKeyboardButton('ወርሃዊ', callback_data='data_point_monthly')]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)

        # Show topic and category names instead of their IDs
        await query.edit_message_text(
            f'ተመርጧል ርዕስ: {selected_topic.title_AMH} ({selected_topic.title_ENG})\n'
            f'ተመርጧል ምድብ: {selected_category.name_AMH} ({selected_category.name_ENG})\n'
            'እባክዎን የውሂብ ነጥብ አይነት ይምረጡ:',
            reply_markup=reply_markup
        )

    except Exception as e:
        await query.edit_message_text(f"ስህተት: {str(e)}")


DATA_POINT_TYPE_TRANSLATIONS = {
    'yearly': 'ዓመታዊ',
    'quarterly': 'በሩብ በሩብ',
    'monthly': 'ወርሃዊ'
}
async def handle_data_point_type_selection(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    
    # Extract and translate the data point type
    data_point_type_key = query.data.split('_')[2]
    data_point_type = DATA_POINT_TYPE_TRANSLATIONS.get(data_point_type_key, data_point_type_key)
    context.user_data['data_point_type'] = data_point_type_key

    category_id = context.user_data['selected_category']
    topic_id = context.user_data.get('selected_topic')
    logger.info(f"Selected data point type: {data_point_type_key} for category ID: {category_id}")

    try:
        # Fetch indicators for the selected category
        response = await sync_to_async(requests.get)(f'{BASE_URL}user-list-view-indicator/{category_id}')
        response.raise_for_status()
        indicators = response.json()

        # Retrieve names from user_data instead of IDs
        selected_topic_name = context.user_data.get('selected_topic_name', 'Unknown Topic')
        selected_category_name = context.user_data.get('selected_category_name', 'Unknown Category')

        if indicators:
            keyboard = [
                [InlineKeyboardButton(indicator['title_ENG'], callback_data=f"indicator_{indicator['id']}")]
                for indicator in indicators
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)
            await query.edit_message_text(
                f'ተመርጧል ርዕስ: {selected_topic_name}\n'
                f'ተመርጧል ምድብ: {selected_category_name}\n'
                f'ተመርጧል ውሂብ ነጥብ አይነት: {data_point_type}\n'
                'እባክዎን የእርምጃ ነጥብ ይምረጡ:',
                reply_markup=reply_markup
            )
        else:
            await query.edit_message_text(
                f'ተመርጧል ውሂብ ነጥብ አይነት: {data_point_type}\n'
                'ለዚህ ምድብ እንደምንም እንቅስቃሴ አልተገኙም።'
            )

    except requests.exceptions.HTTPError as http_err:
        await query.edit_message_text(f"HTTP ስህተት አጋጥሟል: {http_err}")
    except requests.exceptions.RequestException as req_err:
        await query.edit_message_text(f"ስህተት በመጠየቅ: {req_err}")
    except ValueError as json_err:
        await query.edit_message_text(f"በJSON ማንበብ ላይ ስህተት አጋጥሟል: {json_err}")

async def handle_indicator_selection(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    
    # Extract the indicator ID from the callback data
    indicator_id = int(query.data.split('_')[1])
    context.user_data['selected_indicator'] = indicator_id

    logger.info(f"Selected indicator ID: {indicator_id}")

    try:
        # Retrieve the selected indicator from the database
        selected_indicator = await sync_to_async(Indicator.objects.get)(id=indicator_id)
        
        # Retrieve the values for the selected indicator
        values = await sync_to_async(list)(
            DataValue.objects.filter(for_indicator_id=indicator_id).select_related('for_datapoint')
        )
        years = sorted(set(val.for_datapoint.year_EC for val in values))

        # Retrieve names from user_data
        selected_topic_name = context.user_data.get('selected_topic_name', 'Unknown Topic')
        selected_category_name = context.user_data.get('selected_category_name', 'Unknown Category')
        selected_data_point_type = context.user_data.get('data_point_type', 'Unknown Data Point Type')

        # Create the keyboard with years
        keyboard = [
            [InlineKeyboardButton(year, callback_data=f"year_{year}")] for year in years
        ]
        keyboard.append([InlineKeyboardButton("Submit", callback_data="submit_selection")])

        reply_markup = InlineKeyboardMarkup(keyboard)
        await query.edit_message_text(
            f'ተመርጧል ርዕስ: {selected_topic_name}\n'
            f'ተመርጧል ምድብ: {selected_category_name}\n'
            f'ተመርጧል ውሂብ ነጥብ አይነት: {selected_data_point_type}\n'
            f'ተመርጧል እርምጃ ነጥብ: {selected_indicator.title_AMH} ({selected_indicator.title_ENG})\n'
            'እባክዎን የአመታዊ አመት ይምረጡ:',
            reply_markup=reply_markup
        )

        # Initialize the selected years in user_data
        context.user_data['selected_years'] = set()

    except Exception as e:
        await query.edit_message_text(f"ስህተት: {str(e)}")

async def handle_year_selection(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    
    year = query.data.split('_')[1]
    selected_years = context.user_data.get('selected_years', set())

    if year in selected_years:
        selected_years.remove(year)
    else:
        selected_years.add(year)

    context.user_data['selected_years'] = selected_years

    selected_years_text = ", ".join(selected_years)
    keyboard = [
        [InlineKeyboardButton(y, callback_data=f"year_{y}")] for y in sorted(selected_years)
    ]
    keyboard.append([InlineKeyboardButton("Submit", callback_data="submit_selection")])

    reply_markup = InlineKeyboardMarkup(keyboard)
    await query.edit_message_text(
        f'ተመርጧል ርዕስ: {context.user_data.get("selected_topic_name", "አይታወቅም")}\n'
        f'ተመርጧል ምድብ: {context.user_data.get("selected_category_name", "አይታወቅም")}\n'
        f'ተመርጧል ውሂብ ነጥብ አይነት: {context.user_data.get("data_point_type", "አይታወቅም")}\n'
        f'ተመርጧል እርምጃ ነጥብ: {context.user_data.get("selected_indicator_name", "አይታወቅም")}\n'
        f'ተመርጧል አመታዊ ዓመታት: {selected_years_text}\n'
        'ተጨማሪ ዓመታት ይምረጡ ወይም ይስጡ እንደተመረጠው መረጃ ይለውጡ።',
        reply_markup=reply_markup
    )

async def handle_submission(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()

    selected_years = context.user_data.get('selected_years', set())
    indicator_id = context.user_data.get('selected_indicator')

    if not selected_years or not indicator_id:
        await query.edit_message_text(
            text="No years selected or no indicator chosen. Please choose at least one year and an indicator before submitting."
        )
        return

    try:
        indicator = await sync_to_async(Indicator.objects.get)(pk=indicator_id)
        selected_years_text = ", ".join(selected_years)
        indicator_text = indicator.title_ENG  # Get the English title for the indicator

        # Fetch the data values
        data_values = await sync_to_async(lambda: list(DataValue.objects.filter(
            for_indicator_id=indicator_id,
            for_datapoint__year_EC__in=selected_years,
            is_deleted=False
        ).select_related('for_datapoint')  # Ensure related fields are fetched
        ))()

        if data_values:
            # Format the response text with values
            response_text = "\n".join(
                [f"Year: {val.for_datapoint.year_EC}, Value: {val.value}" for val in data_values]
            )
            await query.edit_message_text(
                text=f"ተመርጧል ርዕስ: {context.user_data.get('selected_topic_name')}\n"
                     f"ተመርጧል ምድብ: {context.user_data.get('selected_category_name')}\n"
                     f"ተመርጧል ውሂብ ነጥብ አይነት: {context.user_data.get('data_point_type')}\n"
                     f"ተመርጧል እርምጃ ነጥብ: {indicator_text}\n"
                     f"ተመርጧል አመታዊ ዓመታት: {selected_years_text}\n\n"
                     "የተመረጡትን ውሂብ ይመልከቱ።\n"
                     f"{response_text}",
            )
        else:
            await query.edit_message_text(
                text="No data values found for the selected years and indicator."
            )

    except Exception as e:
        await query.edit_message_text(f"Error: {str(e)}")

    # Clear the user_data for selected years and indicator
    context.user_data.pop('selected_years', None)
    context.user_data.pop('selected_indicator', None)

def start_bot():
    logging.info("Starting bot")
    app = ApplicationBuilder().token(BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("login", login_command))
    app.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), message_handler))
    app.add_handler(CommandHandler("users", users))
    app.add_handler(CommandHandler("logout", logout))
    app.add_handler(CommandHandler('filter', filter_command))
    app.add_handler(CallbackQueryHandler(handle_topic_selection, pattern='^topic_'))
    app.add_handler(CallbackQueryHandler(handle_category_selection, pattern='^category_'))
    app.add_handler(CallbackQueryHandler(handle_data_point_type_selection, pattern='^data_point_'))
    app.add_handler(CallbackQueryHandler(handle_indicator_selection, pattern='^indicator_'))
    app.add_handler(CallbackQueryHandler(handle_year_selection, pattern='^year_'))
    app.add_handler(CallbackQueryHandler(handle_submission, pattern='^submit_selection'))

    # Create a new event loop and set it
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    # Run the bot
    app.run_polling()

bot_thread = None

def run_bot_in_background():
    global bot_thread
    if bot_thread is None or not bot_thread.is_alive():
        bot_thread = threading.Thread(target=start_bot, daemon=True)
        bot_thread.start()

# if __name__ == '__main__':
#     start_bot()
