

#taken from this StackOverflow answer: https://stackoverflow.com/a/39225039
import requests

def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params = { 'id' : id }, stream = True)
    token = get_confirm_token(response)

    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)

    save_response_content(response, destination)    

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None

def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)


file_id = '1T-B8XKXz-yHRRiCCix0beYRq_Kk7bcV_'
destination = 'SWTAM/SWTAM_pretrained.tar.gz'
download_file_from_google_drive(file_id, destination)


file_id = '1SgEgck_352gS7x-UIKhP1HekNMMXtREC'
destination = 'data.tar.gz'
download_file_from_google_drive(file_id, destination)


file_id = '1GtfXQtkCRZYvoYlDUd-f8UYqAO6XMnBU'
destination = 'ASWTAM/ASWTAM_pretrained.tar.gz'
download_file_from_google_drive(file_id, destination)
