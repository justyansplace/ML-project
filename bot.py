import logging
import os
from typing import Dict

import pandas as pd
from telegram import (
    Update,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
)
from telegram.ext import (
    ApplicationBuilder,
    CallbackContext,
    CallbackQueryHandler,
    CommandHandler,
)


logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


RESULTS_CSV_PATH = os.getenv(
    "RESULTS_CSV_PATH",
    os.path.join(os.path.dirname(__file__), "results_df.csv"),
)


def load_results() -> pd.DataFrame:
    """
    Load results dataframe from CSV.

    Expected columns: raceId, y_true, y_pred, race_name, Link.
    """
    df = pd.read_csv(RESULTS_CSV_PATH)
    required_cols = {"raceId", "y_true", "y_pred", "race_name", "Link"}
    if not required_cols.issubset(df.columns):
        missing = required_cols - set(df.columns)
        raise ValueError(
            f"results_df is missing required columns: {', '.join(missing)}"
        )
    return df


def build_race_names(df: pd.DataFrame) -> Dict[int, str]:
    """
    Build a mapping raceId -> race_name from the dataframe.
    """
    # Get unique raceId and race_name pairs (take first occurrence for each raceId)
    unique_races = df[["raceId", "race_name"]].drop_duplicates(subset=["raceId"])
    return {
        int(row["raceId"]): str(row["race_name"])
        for _, row in unique_races.iterrows()
    }


try:
    RESULTS_DF = load_results()
    RACE_NAMES = build_race_names(RESULTS_DF)
except Exception as exc:  # pragma: no cover - startup error logging
    logger.error("Failed to load results_df: %s", exc)
    RESULTS_DF = pd.DataFrame(columns=["raceId", "y_true", "y_pred", "race_name", "Link"])
    RACE_NAMES = {}


def build_races_keyboard() -> InlineKeyboardMarkup:
    """
    Build an inline keyboard with one button per race.

    The layout is similar to the screenshot: one wide button per race.
    Telegram itself handles the visual style – we just make each row
    contain exactly one button.
    """
    buttons = []
    for race_id, name in RACE_NAMES.items():
        buttons.append(
            [InlineKeyboardButton(text=name, callback_data=f"race_{race_id}")]
        )
    return InlineKeyboardMarkup(buttons)


def get_welcome_text() -> str:
    """Return bilingual welcome text used in /start and in the menu."""
    return (
        "<b>Привет! Я бот, который показывает вероятность выезда Safety Car "
        "в гонках Формулы‑1.</b>\n\n"
        "1. Выбери интересующую гонку из списка ниже.\n"
        "2. Я покажу предсказание модели (выезд Safety Car: да/нет)\n"
        "   и реальный результат.\n\n"
    )


async def start(update: Update, context: CallbackContext) -> None:
    """Handler for /start command."""
    if not RACE_NAMES:
        await update.effective_chat.send_message(
            "Не удалось загрузить данные о гонках. "
            "Проверь файл results_df.csv."
        )
        return

    await update.effective_chat.send_message(
        text=get_welcome_text(),
        reply_markup=build_races_keyboard(),
        parse_mode='HTML',
    )


async def handle_race_choice(update: Update, context: CallbackContext) -> None:
    """Handle button click with selected race."""
    query = update.callback_query
    await query.answer()

    data = query.data or ""
    if data == "back_to_menu":
        # Show main menu again
        if not RACE_NAMES:
            await query.edit_message_text(
                "Не удалось загрузить данные о гонках. "
                "Проверь файл results_df.csv."
            )
            return

        await query.edit_message_text(
            text=get_welcome_text(),
            reply_markup=build_races_keyboard(),
            parse_mode='HTML',
        )
        return

    if not data.startswith("race_"):
        return

    try:
        race_id = int(data.split("_", maxsplit=1)[1])
    except ValueError:
        await query.edit_message_text(
            "Некорректный идентификатор гонки."
        )
        return

    row = RESULTS_DF[RESULTS_DF["raceId"] == race_id]
    if row.empty:
        await query.edit_message_text(
            f"Нет данных для гонки с ID {race_id}."
        )
        return

    row = row.iloc[0]
    y_true = int(row["y_true"])
    y_pred = int(row["y_pred"])
    race_name = str(row["race_name"])
    link = str(row["Link"]) if pd.notna(row["Link"]) else None

    true_text_ru = "Да" if y_true == 1 else "Нет"
    pred_text_ru = "Да" if y_pred == 1 else "Нет"

    # Проверяем, правильно ли предсказала модель
    is_correct = (y_pred == y_true)
    
    message = f"<b>{race_name}</b>\n\n"
    
    message += (
        f"Предсказание модели:\n"
        f"- Safety Car: {pred_text_ru}\n\n"
        f"Реальный результат:\n"
        f"- Safety Car: {true_text_ru}\n\n"
    )
    if is_correct:
        message += "✅ Модель оказалась права"
    else:
        message += "❌ Модель ошиблась"
    
    if link:
        message += f"\n\n Хайлайты гонки:\n{link}"

    back_keyboard = InlineKeyboardMarkup(
        [[InlineKeyboardButton(text=" Вернуться в меню", callback_data="back_to_menu")]]
    )

    await query.edit_message_text(
        text=message,
        reply_markup=back_keyboard,
        parse_mode='HTML',
    )


def main() -> None:
    """Run the bot."""
    token = "8208361078:AAGyAwhaqF9y1jWZAxt7AdiyBqQxYqAuVzg"
    if not token:
        raise RuntimeError(
            "TELEGRAM_BOT_TOKEN is not set. "
            "Set it to your bot token from BotFather."
        )

    application = ApplicationBuilder().token(token).build()

    application.add_handler(CommandHandler("start", start))
    application.add_handler(CallbackQueryHandler(handle_race_choice))

    logger.info("Bot is starting...")
    application.run_polling()


if __name__ == "__main__":
    main()


