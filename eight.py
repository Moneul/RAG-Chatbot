import requests
import re
import os
from dotenv import load_dotenv

# 환경변수 등록
dotenv_path = '.env'
load_dotenv(dotenv_path)

# Azure OpenAI 설정
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
DEPLOYMENT_NAME = os.getenv("DEPLOYMENT_NAME")

# Azure Search 설정
AZURE_SEARCH_ENDPOINT = os.getenv("AZURE_SEARCH_ENDPOINT")
AZURE_SEARCH_KEY = os.getenv("AZURE_SEARCH_KEY")
AZURE_SEARCH_INDEX_NAME = os.getenv("AZURE_SEARCH_INDEX_NAME")

# Speech 설정
SPEECH_KEY = os.getenv("SPEECH_KEY")
SPEECH_REGION = os.getenv("SPEECH_REGION")

# Custom Vision 설정
CV_PREDICTION_ENDPOINT = os.getenv("CV_PREDICTION_ENDPOINT")
CV_PREDICTION_KEY = os.getenv("CV_PREDICTION_KEY")
CV_PROJECT_ID = os.getenv("CV_PROJECT_ID")
CV_PUBLISHED_NAME = os.getenv("CV_PUBLISHED_NAME")

#시스템 메세지 & 역할 정보
sys_prompt = "You are an artificial intelligence that takes questions from employees of company 'eight', understands them, and answers them. Your answers should always be friendly, like a tour guide, but without unnecessary platitudes. Also, when writing your answers, you should write from the company's perspective. Your answers should only include the information you received as a question, and if asked for more information, you should provide other additional information.  Example) User: 신입사원은 몇호봉이야? Chatbot: 신입사원은 1호봉입니다.  If you don't have any relevant information about the question, give the following answer without any other response ''죄송합니다. 해당 질문에 대한 정보가 없습니다. 관리자 혹은 인사팀에 문의해주세요.'  If you ask a personal question, give the following answer without any other response '죄송합니다. 해당 질문에는 대답할 수 없습니다.'"
role_information = "'user' is an employee of the company who asks questions about the company. 'assistant' is an AI Chatbot in the company that should understand the user's question and provide information to answer it."

# 이전 대화기록 정보가 저장될 메세지 리스트 (role, content)
messages = []
# chatbot표기용 history
history = []

# API요청을 보내기위한 헤더 설정
headers = {  
    'Content-Type': 'application/json',  
    'api-key': AZURE_OPENAI_API_KEY  
}

# messages 이전 대화 추가용 함수
def append_message(role, content):
    global messages
    messages.append(   
    {
    "role": role,
    "content": content
    },
    )

# 시스템 메세지 설정
append_message("system", sys_prompt)

def chatbot_response(user_input, history):
    global messages

    # 매개변수
    temperature = 0.55
    top_p = 0.85
    max_tokens = 1000

    #user
    append_message("user", user_input)

    payload = {
        "messages": messages,
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
        "data_sources": [
            {
                "type": "azure_search",
                "parameters": {
                    "endpoint": AZURE_SEARCH_ENDPOINT,
                    "index_name": AZURE_SEARCH_INDEX_NAME,
                    "semantic_configuration": "default",
                    "query_type": "vector_semantic_hybrid",
                    "fields_mapping": {},
                    "in_scope": False,
                    "role_information": role_information,
                    "filter": None,
                    "strictness": 3,
                    "top_n_documents": 3,
                    "authentication": {
                        "type": "api_key",
                        "key": AZURE_SEARCH_KEY
                    },
                    "embedding_dependency": {
                        "type": "deployment_name",
                        "deployment_name": "text-embedding-ada-002"
                    }
                }
            }
        ]
    }
    
    response = requests.post(  
        f"{AZURE_OPENAI_ENDPOINT}/openai/deployments/{DEPLOYMENT_NAME}/chat/completions?api-version=2024-05-01-preview",  
        headers=headers,  
        json=payload  
    )

    # 받아온 결과 json으로 파싱
    result = response.json()

    # 정상적으로 리턴 받은경우 
    if result and result['choices']:
        bot_response = result['choices'][0]['message']['content'].strip()
        bot_response = re.sub(r'\[doc\d+\]', '', bot_response)
        append_message('assistant', bot_response)

    # 한도 초과 오류
    elif result and result['error']:
        bot_response = result['error']['message']

    else:
        # if messages[-1]['role'] == "user":
        #     messages.pop()
        history.append((user_input, '일시적으로 오류가 발생하였습니다. 잠시 후 다시 시도해 주세요.'))
        return "", history

    history.append((user_input, bot_response))
    citation_list = []

    if result['choices'][0]['message']['context']:
        citations = result['choices'][0]['message']['context']['citations']
        i = 0
        for citation in citations:
            i += 1
            html = f"<details><summary>참조{i}</summary><ul>{citation['content']}</ul></details>"
            citation_list.append(html)
            # citation_list.append(gr.Textbox(citation['content']))

    # print(messages)
    # print(result)

    return '', history, ''.join(citation_list)
    # return "", history
    

########### 스피치

import azure.cognitiveservices.speech as speechsdk

# speech 설정 리소스 키 / 지역 
speech_config = speechsdk.SpeechConfig(subscription=SPEECH_KEY, region=SPEECH_REGION)
# 텍스트로 변환할 언어 선택 https://learn.microsoft.com/ko-kr/azure/ai-services/speech-service/language-support?tabs=stt
speech_config.speech_recognition_language="ko-KR"

# 시스템 기본 마이크로 연결
audio_config = speechsdk.audio.AudioConfig(use_default_microphone=True)
speech_recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)

# 초기 침묵 및 종료 침묵 시간 설정
speech_config.set_property(speechsdk.PropertyId.SpeechServiceConnection_InitialSilenceTimeoutMs, "3000")  # 초기 침묵 시간 (밀리초)
speech_config.set_property(speechsdk.PropertyId.SpeechServiceConnection_EndSilenceTimeoutMs, "3000")  # 종료 침묵 시간 (밀리초)

def voice2text():
    # speech 객체 연결
    speech_recognition_result = speech_recognizer.recognize_once_async().get()

    # 음성이 인식된 경우
    if speech_recognition_result.reason == speechsdk.ResultReason.RecognizedSpeech:
        return speech_recognition_result.text
    
    # 음성을 인식할 수 없었던 경우
    elif speech_recognition_result.reason == speechsdk.ResultReason.NoMatch:
        return "음성을 인식할 수 없습니다."
    
    # 음성 인식이 취소된 경우
    elif speech_recognition_result.reason == speechsdk.ResultReason.Canceled:
        cancellation_details = speech_recognition_result.cancellation_details
        return "음성 인식이 취소되었습니다."
    

############## 커스텀 비전

from azure.cognitiveservices.vision.customvision.prediction import CustomVisionPredictionClient
from msrest.authentication import ApiKeyCredentials

# API 인증 설정
credentials = ApiKeyCredentials(in_headers={"Prediction-key": CV_PREDICTION_KEY})
predictor = CustomVisionPredictionClient(CV_PREDICTION_ENDPOINT, credentials)

tool_dict = {
    "beamprojector" : "빔프로젝터",
    "printer" : "프린터",
    "router" : "라우터",
    "sterilizer" : "소독기",
    "paperpunch" : "천공기"
    }

from PIL import Image
import io
import numpy as np

def predict_image(image_data, history):
    global tool_dict
    format = image_data.format if image_data.format else "PNG"
    image_bytes = io.BytesIO()
    image_data.save(image_bytes, format=format)
    image_bytes = image_bytes.getvalue()
    
    try:
        results = predictor.detect_image(CV_PROJECT_ID, CV_PUBLISHED_NAME, image_bytes)
        if results.predictions:
            best_pred = max(results.predictions, key=lambda x: x.probability)
            if best_pred.probability >= 0.5:
                tool = tool_dict.get(best_pred.tag_name, "Unknown Tool")
                text = f"{tool}의 사용법을 알려줘"
                _, history, citations = chatbot_response(text, history)
                return history, citations
        
        # 객체를 인식하지 못한 경우
        return history, ""
    
    except Exception as e:
        print(f"Error during prediction: {e}")
        return history, f"Error: {e}"
