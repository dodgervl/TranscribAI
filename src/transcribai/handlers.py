import os
import re
import asyncio
import concurrent.futures
from dotenv import load_dotenv
from subprocess import CalledProcessError, run
import numpy as np

from aiogram import Router, F
from aiogram.types import (
    Message,
    FSInputFile,
    ReplyKeyboardMarkup,
    KeyboardButton,
    ReplyKeyboardRemove,
)
from aiogram.filters import Command
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import StatesGroup, State

from .db import insert_file, get_next_index, make_video_id, get_user_files

# added summarization logic to the bot
from .summarizer import full_process

import whisper
import torch
from imageio_ffmpeg import get_ffmpeg_exe

# path to video processing executable
if "ffmpeg" not in str(os.environ.get("PATH")):
    FFMPEG_BINARY = get_ffmpeg_exe()
    os.environ["PATH"] = os.environ["PATH"] + ";" + FFMPEG_BINARY

load_dotenv("secrets.env")

router = Router()

DATABASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../data/database")
)
os.makedirs(DATABASE_DIR, exist_ok=True)

GOOGLE_DRIVE_LINK_RE = re.compile(
    r"(https?://)?(drive\.google\.com|docs\.google\.com)/[^\s]+"
)
YANDEX_DISK_LINK_RE = re.compile(
    r"(https?://)?(yadi\.sk|disk\.(?:360\.)?yandex\.[^/]+)/[^\s]+"
)
MAX_BOT_FILE_SIZE = 20 * 1024 * 1024  # 20 MB


LANG_OPTIONS = [
    ("Auto", None),
    ("English", "en"),
    ("Русский", "ru"),
    ("Español", "es"),
    ("Français", "fr"),
]
LANG_LABELS = [label for label, _ in LANG_OPTIONS]
LANG_LABEL_TO_CODE = {label: code for label, code in LANG_OPTIONS}
LANG_CODE_TO_LABEL = {
    code if code is not None else "auto": label for label, code in LANG_OPTIONS
}


def lang_label(lang_code):
    if lang_code is None:
        return "Auto-detect"
    return LANG_CODE_TO_LABEL.get(lang_code, lang_code)


def language_keyboard():
    return ReplyKeyboardMarkup(
        keyboard=[[KeyboardButton(text=label) for label, _ in LANG_OPTIONS]],
        resize_keyboard=True,
        one_time_keyboard=True,
    )


class UploadStates(StatesGroup):
    waiting_for_language = State()


pending_files = {}


def load_audio(file: str, sr: int = 16000):
    cmd = [
        FFMPEG_BINARY,
        "-nostdin",
        "-threads",
        "0",
        "-i",
        file,
        "-f",
        "s16le",
        "-ac",
        "1",
        "-acodec",
        "pcm_s16le",
        "-ar",
        str(sr),
        "-",
    ]
    try:
        out = run(cmd, capture_output=True, check=True).stdout
    except CalledProcessError as e:
        raise RuntimeError(f"Failed to load audio: {e.stderr.decode()}") from e

    return np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0


def srt_time(seconds):
    h, m = divmod(int(seconds // 60), 60)
    s = int(seconds % 60)
    ms = int((seconds - int(seconds)) * 1000)
    return f"{h:02}:{m:02}:{s:02},{ms:03}"


def write_srt(segments, srt_path):
    with open(srt_path, "w") as f:
        for i, seg in enumerate(segments, 1):
            f.write(f"{i}\n")
            f.write(f"{srt_time(seg['start'])} - {srt_time(seg['end'])}\n")
            f.write(seg["text"].strip() + "\n\n")


def write_txt_with_timecodes(segments, txt_path):
    with open(txt_path, "w", encoding="utf8") as f:
        for seg in segments:
            start = srt_time(seg["start"])
            end = srt_time(seg["end"])
            try:
                text = seg["text"].strip()
            except:
                text = ""
            f.write(f"[{start} --> {end}]  {text}\n")


def save_transcripts(result, transcriptions_dir):
    os.makedirs(transcriptions_dir, exist_ok=True)
    srt_path = os.path.join(transcriptions_dir, "transcript.srt")
    txt_path = os.path.join(transcriptions_dir, "transcript.txt")
    write_srt(result["segments"], srt_path)
    write_txt_with_timecodes(result["segments"], txt_path)
    return srt_path, txt_path


# NEW VERSION
async def async_transcribe(file_path, transcriptions_dir, language=None):
    loop = asyncio.get_event_loop()

    def task():
        kwargs = {}
        mm = {6: "turbo", 5: "medium", 2: "small", 1: "base"}  # VRAM for model sizes

        if language:
            kwargs["language"] = language
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        if "cuda" in device:  # auto choose model size
            tm = torch.cuda.get_device_properties().total_memory / 1e9 - 1
            for m in mm.keys():
                if tm > m:
                    model_size = mm[m]
                    break
        else:
            model_size = "small"  # biggest suitable for CPU imho
        model = whisper.load_model(model_size)
        result = model.transcribe(load_audio(file_path), **kwargs)
        return save_transcripts(result, transcriptions_dir)

    return await loop.run_in_executor(None, task)


@router.message(Command("help"))
async def help_handler(message: Message):
    await message.answer(
        "<b>TranscribAI Help</b>\n\n"
        "<b>How to use this bot:</b>\n"
        "• <b>Send a video/audio file (≤20MB)</b> directly here, or a public Google Drive/Yandex Disk link to a file.\n"
        "• After upload, <b>choose the transcription language</b> (Auto = automatic detection).\n"
        "• The bot will process the file and send you:\n"
        "   – <b>SRT</b> subtitle file (with timecodes)\n"
        "   – <b>TXT</b> transcript (with timecodes, human-readable)\n"
        "   – <b>Summary</b> of the text with timecodes\n"
        "• Use /list to see your uploaded files/links and their IDs.\n\n"
        "<b>Commands:</b>\n"
        "/help — Show this help and command descriptions.\n"
        "/list — List all your uploaded files and links (IDs are clickable links for link uploads).\n"
        "\nSend a file or link to start!",
        parse_mode="HTML",
    )


@router.message(F.text)
async def echo_handler(message: Message):
    await message.answer(message.text)


@router.message(Command("list"))
async def list_handler(message: Message):
    user_id = message.from_user.id
    files = get_user_files(user_id)
    if not files:
        await message.answer("You haven't uploaded any files or links yet.")
        return
    lines = []
    for idx, video_id, video_link, video_file_path, language in files:
        lang_str = language if language else "auto"
        if video_link:
            lines.append(f'<a href="{video_link}">{video_id}</a> ({lang_str})')
        else:
            lines.append(
                f"<b>{video_id}</b>: {os.path.basename(video_file_path)} ({lang_str})"
            )
    await message.answer("Your files/links:\n" + "\n".join(lines), parse_mode="HTML")


@router.message(Command("start"))
async def start_handler(message: Message):
    await help_handler(message)


@router.message(F.video | F.audio)
async def handle_media(message: Message, state: FSMContext):
    user_id = message.from_user.id
    idx = get_next_index(user_id)
    video_id = make_video_id(user_id, idx)
    file = message.video or message.audio
    file_name = file.file_name or "file"
    file_size = file.file_size

    if file_size and file_size > MAX_BOT_FILE_SIZE:
        await message.answer(
            "File too large (max 20MB). For large files, use Yandex Disk or Google Drive."
        )
        return

    user_video_dir = os.path.join(DATABASE_DIR, str(user_id), video_id)
    os.makedirs(user_video_dir, exist_ok=True)
    local_filename = os.path.join(user_video_dir, file_name)

    file_info = await message.bot.get_file(file.file_id)
    file_path = file_info.file_path
    await message.bot.download_file(file_path, local_filename)

    await message.answer(f"File saved with ID <b>{video_id}</b>.", parse_mode="HTML")

    pending_files[user_id] = (
        local_filename,
        os.path.join(user_video_dir, "transcriptions"),
        (user_id, idx, video_id, None, local_filename),  # We'll append lang_code later
    )
    await message.answer(
        "Please choose the language of the lecture (or press 'Auto' for automatic detection):",
        reply_markup=language_keyboard(),
    )
    await state.set_state(UploadStates.waiting_for_language)


def gdrive_download(link, output_dir):
    import gdown, requests, re

    resp = requests.get(link)
    # iterate on metadata to find filename and format
    for i in resp.content.decode().split("meta property="):
        if "og:title" in i:
            filename = re.findall('".*?"', i)[1].replace('"', "")
            break
    return gdown.download(
        url=link, output=os.path.join(output_dir, filename), quiet=False, fuzzy=True
    )


def yandex_disk_download(link, output_dir):
    import yadisk

    try:
        y = yadisk.YaDisk()
        public_key = link
        meta = y.get_public_meta(public_key)
        filename = meta["name"]
        local_path = os.path.join(output_dir, filename)
        y.download_public(public_key, local_path)
        return local_path
    except Exception as e:
        raise RuntimeError(f"Yandex Disk download failed: {e}")


@router.message(
    lambda m: m.text
    and (GOOGLE_DRIVE_LINK_RE.search(m.text) or YANDEX_DISK_LINK_RE.search(m.text))
)
async def handle_disk_link(message: Message, state: FSMContext):
    user_id = message.from_user.id
    idx = get_next_index(user_id)
    video_id = make_video_id(user_id, idx)
    text = message.text.strip()
    gd_match = GOOGLE_DRIVE_LINK_RE.search(text)
    yd_match = YANDEX_DISK_LINK_RE.search(text)

    user_video_dir = os.path.join(DATABASE_DIR, str(user_id), video_id)
    os.makedirs(user_video_dir, exist_ok=True)

    loop = asyncio.get_event_loop()
    with concurrent.futures.ThreadPoolExecutor() as pool:
        if gd_match:
            link = gd_match[0]
            await message.answer("Downloading google drive file. Please wait...")
            try:
                out_path = await loop.run_in_executor(
                    pool, gdrive_download, link, user_video_dir
                )
                await message.answer(
                    f'File saved with ID <a href="{link}">{video_id}</a>.',
                    parse_mode="HTML",
                )
                pending_files[user_id] = (
                    out_path,
                    os.path.join(user_video_dir, "transcriptions"),
                    (user_id, idx, video_id, link, out_path),
                )
                await message.answer(
                    "Please choose the language of the lecture (or press 'Auto' for automatic detection):",
                    reply_markup=language_keyboard(),
                )
                await state.set_state(UploadStates.waiting_for_language)
            except Exception as ex:
                await message.answer(f"Failed to download Google Drive file: {ex}")
        elif yd_match:
            link = yd_match[0]
            await message.answer("Downloading yandex disk file. Please wait...")
            try:
                out_path = await loop.run_in_executor(
                    pool, yandex_disk_download, link, user_video_dir
                )
                await message.answer(
                    f'File saved with ID <a href="{link}">{video_id}</a>.',
                    parse_mode="HTML",
                )
                pending_files[user_id] = (
                    out_path,
                    os.path.join(user_video_dir, "transcriptions"),
                    (user_id, idx, video_id, link, out_path),
                )
                await message.answer(
                    "Please choose the language of the lecture (or press 'Auto' for automatic detection):",
                    reply_markup=language_keyboard(),
                )
                await state.set_state(UploadStates.waiting_for_language)
            except Exception as ex:
                await message.answer(f"Failed to download Yandex Disk file: {ex}")
        else:
            await message.answer("Unrecognized link.")


@router.message(lambda m: m.text in LANG_LABELS, UploadStates.waiting_for_language)
async def language_selected(message: Message, state: FSMContext):
    user_id = message.from_user.id
    lang_label_ = message.text
    lang_code = LANG_LABEL_TO_CODE.get(lang_label_)
    await message.answer(
        f"Now transcribing in <b>{lang_label(lang_code)}</b>...",
        reply_markup=ReplyKeyboardRemove(),
        parse_mode="HTML",
    )
    info = pending_files.pop(user_id, None)
    if not info:
        await message.answer("No file pending for transcription.")
        await state.clear()
        return
    local_filename, transcriptions_dir, insert_args = info
    try:
        insert_file(*insert_args, lang_code)
        srt_path, txt_path = await async_transcribe(
            local_filename, transcriptions_dir, language=lang_code
        )
        await message.answer_document(
            FSInputFile(srt_path), caption="SRT (subtitles with timecodes)"
        )
        await message.answer_document(
            FSInputFile(txt_path), caption="Transcript with timecodes"
        )
        await message.answer(
            "Transcription done! Files also saved in your folder. Use /list to see your files."
        )
        transcript_path = os.path.normpath(
            os.path.join(transcriptions_dir, "transcript.txt")
        )
        os.makedirs(os.path.dirname(transcript_path), exist_ok=True)
        text = open(transcript_path, "r", encoding="utf8").read()
        summary = full_process(text)
        await message.answer("Summarization complete, here it is:\n" + summary)

    except Exception as ex:
        await message.answer(f"Transcription failed: {ex}")
    await state.clear()


@router.message()
async def catch_all(message: Message):
    pass
