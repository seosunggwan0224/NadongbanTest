from flask import Flask, render_template, request, jsonify,send_file
import httpcore
setattr(httpcore, 'SyncHTTPTransport', httpcore.AsyncHTTPProxy)
import os
from openai import OpenAI
import openai
import json
import numpy as np
from numpy.linalg import norm
import re
from time import time,sleep
from uuid import uuid4
from pinecone import Pinecone
import base64
from gtts import gTTS
from googletrans import Translator
import pymysql
import pyttsx3
import speech_recognition as sr
import datetime
from dotenv import load_dotenv

# .env 파일 활성화
load_dotenv()

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')


def open_file(filepath):   
    with open(filepath, 'r', encoding='utf-8') as infile:
        return infile.read()

# 파일 저장하는 함수
def save_file(filepath, content):  
    with open(filepath, 'w', encoding='utf-8') as outfile:
        outfile.write(content)

#제이슨 파일 불러오는 함수
def load_json(filepath):
    with open(filepath, 'r', encoding='utf-8') as infile:
        return json.load(infile)

# 제이슨파일 저장하는함수
def save_json(filepath, payload):                                                   #payload: JSON 형식으로 저장하고자 하는 데이터를 담고 있는 변수
    with open(filepath, 'w', encoding='utf-8') as outfile:
        json.dump(payload, outfile, ensure_ascii=False, sort_keys=True, indent=2)  #json.dump() 함수는 Python에서 JSON 데이터를 파일로 저장하는 데 사용

#시간,날짜체계 만들기
def timestamp_to_datetime(unix_time):
    return datetime.datetime.fromtimestamp(unix_time).strftime("%A, %B %d, %Y at %I:%M%p %Z")

#문장 벡터로 임베딩
def gpt3_embedding(content, model='text-embedding-ada-002'):
    # content = content.encode(encoding='ASCII',errors='ignore').decode()  # fix any UNICODE errors
    # response = client.embeddings.create(input=content, model=model)
    # vector = response['data'][0]['embedding']  # this is a normal list
    content = content.replace("\n", " ")
    vector = client.embeddings.create(input = [content], model=model).data[0].embedding
    return vector

def gpt3_completion(prompt, model='ft:gpt-3.5-turbo-0125:personal::9LmnTfYw'):
    max_retry = 5
    retry = 0
    # prompt = prompt.encode(encoding='ASCII',errors='ignore').decode()
    while True:
        try:
            response = client.chat.completions.create(
                model = model,
                messages=[
                    {"role": "system", "content": "날짜 체계를 고려해서 두번째 문장에대한 대답해"},
                    {"role": "user", "content": prompt}
                ]
            )
            text = response.choices[0].message.content.strip()
            text = re.sub('[\r\n]+', '\n', text)
            text = re.sub('[\t ]+', ' ', text)
            filename = '%s_gpt3.txt' % time()
            if not os.path.exists('gpt3_logs'):
                os.makedirs('gpt3_logs')
            save_file('gpt3_logs/%s' % filename, prompt + '\n\n==========\n\n' + text)
            return text
        except Exception as oops:
            retry += 1
            if retry >= max_retry:
                return "GPT3 error: %s" % oops
            print('Error communicating with OpenAI:', oops)
            sleep(1)

# 유사도로 가까운거 뽑은것들 텍스트로 쭉
def load_conversation(results):    #results = vdb.query(vector=vector, top_k=convo_length)
    result = list()   #-> result라는 리스트 만듬
    for m in results['matches']:   #results에는 matches라는게있음 그만큼 반복 -> 아마 top_k 일듯?
        info = load_json('nexus/%s.json' % m['id'])   
        result.append(info)                           #result 리스트에 info 삽입
    ordered = sorted(result, key=lambda d: d['time'], reverse=False)  # sort them all chronologically
    messages = [i['message'] for i in ordered]
    return '\n'.join(messages).strip()

def save_list_file_for_finetuning(file_name, content):     #-> 파인튜닝할용도로 prompt와 output저장
    try:
        # 파일을 쓰기 모드로 엽니다.
        with open(file_name, 'a') as file:
            # content를 파일에 씁니다.
            file.write(content)
        print(f'파일 "{file_name}"에 성공적으로 저장되었습니다.')
    except Exception as e:
        print(f'파일 "{file_name}" 저장 중 오류가 발생했습니다: {e}')


# 유사도로 가까운거 뽑은것들 텍스트로 쭉
def load_conversation2(results):    #results = vdb.query(vector=vector, top_k=convo_length)
    result = list()   #-> result라는 리스트 만듬
    #parsed_data = json.loads(results)           #첫번째 문제  the JSON object must be str, bytes or bytearray, not QueryResponse
    highest_score = float("-inf")  # 가장 작은 값으로 초기화
    highest_score_id = None
    for m in results["matches"]:
        if m["score"] > highest_score:
            highest_score = m["score"]
            highest_score_id = m["id"]   #lowest_score_id에 제일 유사한 문장의 벡터의 id가 들어가있음  여기까진 잘뽑아짐
    print(highest_score_id)
    info = load_json('nexus/%s.json' % highest_score_id)  #두번째 에러: [Errno 2] No such file or directory: 'nexus/3eba54a1-95eb-4996-b0e9-f2e4f04d973c.json'
    result.append(info)
    ordered = sorted(result, key=lambda d: d['time'], reverse=False)  # sort them all chronologically
    messages = [i['message'] for i in ordered]
    return '\n'.join(messages).strip() 

# openai_api_key = open_file('key_openai.txt')


def chat_main(a):
    if __name__ == '__main__':
        convo_length = 5 #유사도 가장 높은 거 3개 뽑는 용도
        # openai_api_key = open_file('key_openai.txt')
        pc = Pinecone(api_key=PINECONE_API_KEY)
        vdb = pc.Index("nadongban")
        #while True:
        now = datetime.datetime.now()                      #현재시간 가져오기  #now에는 현재시간 들어감
        formatted_date_time = now.strftime("%Y%m%d-%H:%M")  #날짜체계가들어감 ex) -> 20240307-21:50
        hangletime = str(formatted_date_time)             #ormatted_date_time과 똑같은게 들어감
        print(type(formatted_date_time))
        #### get user input, save it, vectorize it, save to pinecone
        payload = list()                                            #JSON 형식으로 저장하고자 하는 데이터를 담고 있는 변수를 리스트로 형변환
        #a = input('\n\nUSER: ')                  # 사용자가 프로그램에게 말하고싶은것,질문하고싶은건 ex) -> 내가 4월27일날 뭐하기로 했더라
        timestamp = time()          
        timestring = timestamp_to_datetime(timestamp)
        #message = '%s: %s - %s' % ('USER', timestring, a)
        message = hangletime + " : " + a        # ex)-> 20240307-21:50 : 내가 4월 27일날 뭐하기로 했지?
        print(message)                          #사용자가 말한것 출력
        vector = gpt3_embedding(message)        #사용자가 말한것 벡터로 임베딩
        unique_id = str(uuid4())      #uid를 생성 
        metadata = {'speaker': 'USER', 'time': timestamp, 'message': message, 'timestring': timestring, 'uuid': unique_id} #메타데이터 만들고
        save_json('nexus/%s.json' % unique_id, metadata) #제이슨파일만들어서 저장
        #payload.append((unique_id, vector))
        ################################여기까지가 사용자의 답변 #################

        #### search for relevant messages, and generate a response
        results = vdb.query(vector=vector, top_k=convo_length) # 유사문장 찾기(vector값, top_k= 유사문장 갯수)
        print(type(results))   #<class 'pinecone.core.client.model.query_response.QueryResponse'>
        print(results)
        '''
            {'matches': [{'id': '3eba54a1-95eb-4996-b0e9-f2e4f04d973c',
                'score': 0.786891937,
                'values': []}],
                'namespace': '',
                'usage': {'readUnits': 5}}
        '''
        #'3eba54a1-95eb-4996-b0e9-f2e4f04d973c',  내 4월5일날 누구하고 가평가기로 했지?
        #3eba54a1-95eb-4996-b0e9-f2e4f04d973c     내가 4월27일날 뭐하기로 했지?
        #3eba54a1-95eb-4996-b0e9-f2e4f04d973c  안녕
        
        #내가 예상하기론 그때 파인콘 벡터 한번 지워서 파인콘에 3eba54a1-95eb-4996-b0e9-f2e4f04d973c 이거 하나밖에 안들어있어서 그런듯
        conversation = load_conversation2(results)  # results should be a DICT with 'matches' which is a LIST of DICTS, with 'id' # 결과는 'id'가 포함된 'matches'가 포함된 DICT여야 합니다
        prompt = open_file('prompt_response.txt').replace('<<CONVERSATION>>', conversation).replace('<<MESSAGE>>', a)  #사용자가 현재말한 문장과 conversation을 합치기 ex) 20240226-13:56 : 오 이 자주색 코트 이쁘다 내일 인터넷으로 찾아봐야겠다.  20240227-16:17 : 어제 내가 이쁘다고한 코트 무슨색이었지?
        #### generate response, vectorize, save, etc
        output = gpt3_completion(prompt)    #gptapi에 prompt 넣고 파인튜닝된 gpt의 답변 구하기   ex) 어제 자주색 코트가 이쁘다며 말씀하셨어요.
        timestamp = time()
        timestring = timestamp_to_datetime(timestamp)
        #message = '%s: %s - %s' % ('RAVEN', timestring, output)
        message = output      #ex)message에는 어제 자주색 코트가 이쁘다며 말씀하셨어요. 가 들어가있음
            

        #vector = gpt3_embedding(message)      # ex)message에는 어제 자주색 코트가 이쁘다며 말씀하셨어요. 가 임베딩되어있음
        #답변은 굳이 임베딩할필요가 없을수도?  답변은 그냥 사용자에게 보여주기만해도 될듯 -> vdb에 넣을게 아니라서 임베딩 안해도될듯

        #############################################안해도되는것###############################
        '''unique_id = str(uuid4())    #uid 생성
        metadata = {'speaker': '나동반', 'time': timestamp, 'message': message, 'timestring': timestring, 'uuid': unique_id} #메타데이터 만들기
        save_json('nexus/%s.json' % unique_id, metadata)   #답변을 json파일 만들어서 저장  metadata가 payload
        payload.append((unique_id, vector))
        vdb.upsert(payload)  #payload는 리스트임'''
        #############################################안해도되는것###############################
            
        #vdb.upsert(vector)
        payload.append((unique_id, vector))
        vdb.upsert(payload)
        #vdb.upsert(vectors=[{"id":unique_id,"values":vector}])

        #print('\n\n나동반: %s' % output) # output 출력 ex)어제 자주색 코트가 이쁘다며 말씀하셨어요.  ->성공!!

        file_name = "listup_file_for_finetuning"   #->그냥 텍스트 파일
        content = prompt+'\n'+output+'\n\n\n'  #-> ex) 20240226-13:56 : 오 이 자주색 코트 이쁘다 내일 인터넷으로 찾아봐야겠다.  20240227-16:17 : 어제 내가 이쁘다고한 코트 무슨색이었지? \n 어제 자주색 코트가 이쁘다며 말씀하셨어요. \n\n\n
        save_list_file_for_finetuning(file_name, content)
    return output

def detect_language(input):
    translator = Translator()
    detected_lang = translator.detect(input).lang
    return detected_lang

def generate_speech(input_text, output_file_path, model="tts-1", voice="alloy"):
    response = client.audio.speech.create(
        model=model,
        voice=voice,
        input=input_text
    )
    response.stream_to_file(output_file_path)

app = Flask(__name__, static_folder='static')
app.config['JSON_AS_ASCII'] = False

conn = pymysql.connect(host='ls-94e556626c88eb365a7ec359470d1d33f56b67c1.ch0q0mic69gt.ap-northeast-2.rds.amazonaws.com', user='dbmasteruser', password='QNx87|udqkmuNTRN>%i1aaI*^9t<D1.s', db="nadongban", charset='utf8')

cursor = conn.cursor()

# 음성 인식 관련 설정
recognizer = sr.Recognizer()
today = "2024-5-7" #오늘 날짜
insert_sql = '''INSERT INTO Health (Alcohol, Outside, Exercise, Today, Content) VALUES (%s, %s, %s, %s, %s)'''
# openai키
client = OpenAI(api_key=OPENAI_API_KEY)

from pathlib import Path
from openai import OpenAI

@app.route("/")
def index():
    return render_template('main.html')

@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    return chat_main(input)

@app.route('/generate_speech', methods=['POST'])
def generate_and_send_speech():
    data = request.json
    input_text = data.get('text')
    if not input_text:
        return "Text field is required", 400

    speech_file_path = Path(__file__).parent / "speech.mp3"
    generate_speech(input_text, speech_file_path)
    return send_file(speech_file_path, as_attachment=True)

# 메인 페이지 라우트
@app.route('/sub')
def calender():
    # 데이터베이스에서 정보를 가져옴
    cursor.execute("SELECT * FROM Health")
    data = cursor.fetchall()
    return render_template('index.html', data=data)

# 음성 데이터 처리 라우트
@app.route('/process_voice_data', methods=['POST'])
def process_voice_data():
    cursor.execute("SELECT 1 FROM Health WHERE Today = %s LIMIT 1", (today,))
    if cursor.fetchone():
        # 데이터가 이미 존재하면 처리하지 않고 메시지 반환
        return jsonify({'error': 'Data for today already exists.'}), 400
    try:
        with sr.Microphone() as source:
            print("오늘 하루 어땠는지 말씀해주세요.")
            audio = recognizer.listen(source)
            # Google Web Speech API를 이용해 음성을 텍스트로 변환
            text = recognizer.recognize_google(audio, language="ko-KR")
            print("음성 인식 결과:", text)

        # Drink Alcohol
        completion = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "음주를 했으면 1을 반환하고 음주를 하지 않았으면 2를 반환해줘"},
                {"role": "user", "content": text}
            ]
        )

        if "1" in completion.choices[0].message.content:
            Alcohol = 1
        else:
            Alcohol = 0

        print("음주 여부 : ", Alcohol)

        # Outside
        completion = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "외출을 했으면 1을 반환하고 외출을 하지 않았으면 2를 반환해줘"},
                {"role": "user", "content": text}
            ]
        )
        if "1" in completion.choices[0].message.content:
            OutSide = 1
        else:
            OutSide = 0

        print("외출 여부 : ", OutSide)

        # Exercise
        completion = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "운동을 했으면 1을 반환하고 운동을 하지 않았으면 2를 반환해줘"},
                {"role": "user", "content": text}
            ]
        )
        if "1" in completion.choices[0].message.content:
            Exercise = 1
        else:
            Exercise = 0

        print("운동 여부 : ", Exercise)

        cursor.execute(insert_sql, (Alcohol, OutSide, Exercise, today, text))
        conn.commit()

        # 처리 완료 후 데이터베이스에서 정보를 가져와서 반환
        cursor.execute("SELECT * FROM Health")
        data = cursor.fetchall()
        return jsonify(data)

    except sr.UnknownValueError:
        print("음성을 인식할 수 없습니다.")
        return jsonify({'error': '음성을 인식할 수 없습니다.'})
    except sr.RequestError as e:
        print(f"음성 인식 서비스에 접근할 수 없습니다: {e}")
        return jsonify({'error': '음성 인식 서비스에 접근할 수 없습니다.'})

# 선택한 날짜 처리 라우트
@app.route('/process_selected_date', methods=['GET'])
def process_selected_date():
    selected_date = request.args.get('date')
    cursor.execute("SELECT alcohol, outside, exercise, content FROM Health WHERE today = %s ORDER BY id DESC LIMIT 1;", (selected_date,))
    data = cursor.fetchone()
    if data:
        # 데이터가 존재하는 경우에만 JSON 형식으로 반환합니다.
        return jsonify({'date': selected_date, 'alcohol': data[0], 'outside': data[1], 'exercise': data[2], 'content':data[3]})
    else:
        # 데이터가 없는 경우 오류 메시지를 반환합니다.
        return jsonify({'error': 'No data found for the selected date.'}), 404

@app.route('/get-data-for-date', methods=['POST'])
def get_data_for_date():
    # 요청의 JSON 본문에서 날짜를 추출합니다.
    date = request.json.get('date')
    if date:
        # 여기서 해당 날짜에 대한 데이터베이스 쿼리를 수행합니다.
        # 일단은 가짜 응답을 반환합니다.
        data = {'message': date + '에 대한 데이터'}
        return jsonify(data)
    else:
        # 날짜가 제공되지 않으면 오류 응답을 반환합니다.
        return jsonify({'error': '날짜가 제공되지 않았습니다.'}), 400

@app.route('/check_date_data', methods=['POST'])
def check_date_data():
    dates = request.json.get('dates')
    if not dates:
        return jsonify({'error': 'No dates provided'}), 400

    placeholder = ', '.join(['%s'] * len(dates))
    query = f"SELECT today, alcohol, exercise FROM Health WHERE today IN ({placeholder})"
    cursor.execute(query, dates)
    results = cursor.fetchall()

    # 결과 데이터를 처리하여 응답 데이터 구성
    dates_data = {}
    for result in results:
        date, alcohol, exercise = result
        formatted_date = date.strftime('%Y-%m-%d')
        dates_data[formatted_date] = {
            'hasData': True,
            'alcohol': bool(alcohol),
            'exercise': bool(exercise)
        }

    # 요청받은 모든 날짜에 대해 데이터가 없는 경우도 처리
    for date in dates:
        if date not in dates_data:
            dates_data[date] = {'hasData': False}

    return jsonify(dates_data)

@app.route('/get-monthly-stats', methods=['POST'])
def get_monthly_stats():
    month = request.json['month']
    query = """
        SELECT 
            SUM(alcohol) as alcohol_count, 
            SUM(outside) as outside_count, 
            SUM(exercise) as exercise_count 
        FROM Health 
        WHERE DATE_FORMAT(today, '%Y-%m') = %s;
    """
    cursor.execute(query, (month,))
    result = cursor.fetchone()
    return jsonify({
        'alcohol': int(result[0] or 0),
        'outside': int(result[1] or 0),
        'exercise': int(result[2] or 0)
    })

@app.route('/get-monthly-data', methods=['POST'])
def get_monthly_data():
    month = request.json['month']
    try:
        # DATE_FORMAT의 포맷 문자열에서 %%Y와 %%m 사용
        query = """
            SELECT 
                SUM(alcohol) as alcohol_count, 
                SUM(outside) as outside_count, 
                SUM(exercise) as exercise_count 
            FROM Health 
            WHERE DATE_FORMAT(today, '%%Y-%%m') = %s;
        """
        cursor.execute(query, (month,))
        results = cursor.fetchone()
        return jsonify({
            'alcohol': int(results[0] or 0),
            'outside': int(results[1] or 0),
            'exercise': int(results[2] or 0)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(port=5000,debug=True)