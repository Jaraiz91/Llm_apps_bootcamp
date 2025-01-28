from datetime import datetime, timedelta
import pickle
import os
from dotenv import load_dotenv, find_dotenv
from langchain.tools import tool

from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
import random


_ = load_dotenv(find_dotenv())
openai_api_key = os.environ['OPENAI_API_KEY']
client_id = os.environ['GOOGLE_CLIENT_ID']
client_secret = os.environ['GOOGLE_CLIENT_SECRET']
print(os.environ['LANGCHAIN_PROJECT'])


# Configurar los alcances y la autenticación
SCOPES = ['https://www.googleapis.com/auth/calendar']

def authenticate_google_calendar():

    creds = None
    if os.path.exists('token.pickle'):
        with open('token.pickle', 'rb') as token:
            creds = pickle.load(token)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_config(
                {
                    "installed": {
                        "client_id": client_id,
                        "client_secret": client_secret,
                        "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                        "token_uri": "https://oauth2.googleapis.com/token",
                        "redirect_uris": ["http://localhost"]
                    }
                },
                SCOPES
            )
            creds = flow.run_local_server(port=0)
        with open('token.pickle', 'wb') as token:
            pickle.dump(creds, token)
    return creds

print('check!')


def get_calendar_events():
    creds = authenticate_google_calendar()
    service = build('calendar', 'v3', credentials=creds)

    # Obtener eventos del calendario
    now = datetime.utcnow().isoformat() + 'Z'
    events_result = service.events().list(
        calendarId='primary', timeMin=now, maxResults=10, singleEvents=True, orderBy='startTime'
    ).execute()
    events = events_result.get('items', [])
    for event in events:
        print(event['summary'])


def get_three_random(lista):
    # Verificar si la lista tiene más de 3 elementos
    if len(lista) > 3:
        # Devolver 3 elementos aleatorios usando random.sample
        return random.sample(lista, 3)
    else:
        # Si la lista tiene 3 o menos elementos, devolver la lista original
        return lista
    
def get_available_slots():
    creds = authenticate_google_calendar()
    service = build('calendar', 'v3', credentials=creds)

    # ID del calendario (usa 'primary' para el calendario principal del usuario)
    calendar_id = 'primary'

    # Rango de tiempo: desde hoy hasta el final de mañana
    now = datetime.now().astimezone()
    start_of_today = now.replace(hour=9, minute=0, second=0, microsecond=0)
    end_of_tomorrow = (start_of_today + timedelta(days=2)).replace(hour=19)

    # Obtener eventos existentes en el rango de tiempo
    events_result = service.events().list(
        calendarId=calendar_id,
        timeMin=start_of_today.isoformat() ,
        timeMax=end_of_tomorrow.isoformat() ,
        singleEvents=True,
        orderBy='startTime'
    ).execute()
    events = events_result.get('items', [])

    # Crear una lista de franjas ocupadas
    busy_slots = []
    for event in events:
        start = event['start'].get('dateTime', event['start'].get('date'))
        end = event['end'].get('dateTime', event['end'].get('date'))
        busy_slots.append((datetime.fromisoformat(start), datetime.fromisoformat(end)))

    # Generar todas las franjas de media hora dentro del horario laboral
    available_slots = []
    current_time = start_of_today
    while current_time < end_of_tomorrow:
        next_time = current_time + timedelta(minutes=30)
        # Verificar si la franja está ocupada
        if not any(start <= current_time < end for start, end in busy_slots):
            available_slots.append((current_time, next_time))
        current_time = next_time
    print('eventos escaneados')
    proposed_slots = get_three_random(available_slots)
    return proposed_slots



def crear_evento(start_time, end_time, summary="Reunión bloqueada"):
    """
    Crea un evento en Google Calendar
    Retorna: ID del evento creado
    """
    creds = authenticate_google_calendar()
    service = build('calendar', 'v3', credentials=creds)
    evento = {
        'summary': summary,
        'start': {
            'dateTime': start_time.isoformat(),
            'timeZone': 'Europe/Madrid'
        },
        'end': {
            'dateTime': end_time.isoformat(),
            'timeZone': 'Europe/Madrid'
        }
    }

    try:
        evento = service.events().insert(calendarId='primary', body=evento).execute()
        print(f'Evento creado: {evento.get("htmlLink")}')
        return evento['id']  # Retornamos el ID del evento para poder borrarlo después
    except Exception as e:
        print(f'Error al crear el evento: {e}')
        return None


def borrar_evento(event_id):
    """
    Borra un evento de Google Calendar usando su ID
    """
    creds = authenticate_google_calendar()
    service = build('calendar', 'v3', credentials=creds)
    try:
        service.events().delete(calendarId='primary', eventId=event_id).execute()
        print('Evento borrado exitosamente')
        return True
    except Exception as e:
        print(f'Error al borrar el evento: {e}')
        return False
