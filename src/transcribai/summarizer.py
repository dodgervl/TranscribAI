from yandex_cloud_ml_sdk import YCloudML
from dotenv import load_dotenv
import re
import os

load_dotenv('secrets.env')
api_token = os.getenv('API_TOKEN')
folder_id = os.getenv('folder_id')

# убирает ненужные мс в транскрипте
def cut_ms(txt):
    pattern = r'(\d{2}(?::\d{2}){1,2})\.\d{3}'
    replacement = r'\1'
    return re.sub(pattern, replacement, txt)

# прогоняет промпт 
def run_prompt(api_token, folder_id, system_prompt, input_text):
    
    messages = [
        {
            'role':'system',
            'text': system_prompt
        },
        {
            'role':'user',
            'text': input_text
        }
    ]
    
    sdk = YCloudML(
        folder_id=folder_id,
        auth=api_token
    )
    model = sdk.models.completions("yandexgpt").configure(temperature=0.3)
    result = model.run(messages)
    return result.alternatives[0].text

# условный токенайзер))))
def count_token(text):
    if type(text)==str:
        return len(text)/4
    else:
        return False

# разбивка на части для лимита по токенам    
def process_transcript(text, limit_user_prompt):
    lines = text.split('\n')
    splits=[]
    curr_chunk ='Входные данные:\n'
    for line in lines:
        if count_token(curr_chunk+line) < limit_user_prompt:
            curr_chunk+= line
        else:
            splits.append(curr_chunk)
            curr_chunk = 'Входные данные:\n'+line
    if curr_chunk!= 'Входные данные:\n':
        splits.append(curr_chunk)
    return splits

# Суммаризация частей с учетом контекста    
def iterate_run(text):
    system_prompt = f"""Ты программа, которая должна конспектировать транскрипцию, которая подается на входе. 
    Выдели самые важные моменты из этой записи и расскажи об их сути в 2-3 предложениях, для каждого момента укажи таймкод его начала.
Правила:
1. Фокусируйся только на самой важной информации. Не уходи в детали.
2. Для каждого ключевого момента укажи таймкод, взятый из начала соответствующего фрагмента в транскрипте.
3. Убедись, что таймкод точно соответствует началу обсуждаемого ключевого момента.
4. Отформатируй вывод в виде списка, где каждый элемент списка – это "Ключевой момент - ТАЙМКОД". Не добавляй ничего лишнего до или после списка.
5. Не дублируй темы, которые были недавно законспектированны.
Пример вывода:
- Основная проблема психологии. - 10:15
- Предложение первого подхода решения. - 15:30
Ты уже законспектировал:\n"""
    
    context_len = int((500 - count_token(system_prompt))*4)
    
    limit = 1500
    
    previous_summ = ''
    
    chunks = process_transcript(text, limit)
    
    for chunk in chunks:
        system_prompt_context = system_prompt + previous_summ[-context_len:]
        previous_summ+= run_prompt(api_token, folder_id, system_prompt_context, chunk)+'\n'
    return previous_summ

# Название из конспекта
def get_name(summary):
    system_prompt = 'Ты - программа, которвая должна дать название предоставленному конспекту. УЧТИ специфику и тему именно данного конспекта и включи их в название. В ответе дай только название в формате `Название конспекта`'
    user_prompt = 'Входные данные:\n' + summary
    sum_name = run_prompt(api_token, folder_id, system_prompt, user_prompt)
    return sum_name

# Починка таймкодов вида 01:07 -> 00:01:07
def extract_timecodes(text):
    pattern = r'(?<!\d)(\d{1,2}:\d{2}(?::\d{2})?)(?!\d)'
    timecodes = re.findall(pattern, text)
    timecodes =list(map(lambda x: x if x.count(':')>1 else '00:'+x,timecodes))
    text_no_tc = re.sub(pattern, '', text)
    text_no_tc =text_no_tc.strip('\n').split('\n')
    for i in range(len(timecodes)):
        text_no_tc[i]= text_no_tc[i]+timecodes[i]
    return '\n'.join(text_no_tc)

# Ну и типа общая логика
def full_process(text):
    text = cut_ms(text)
    summary = extract_timecodes(iterate_run(text))
    name = get_name(text)
    return name + '\n' +summary