import requests

# url = 'http://localhost:5059/create_vectordb'
#
# data = {
#     'embedding_name': 'sentence-transformers/LaBSE',
#     'vector_db_path': 'assets/faiss_index'
# }
#
# try:
#     response = requests.post(url, json=data)
#     if response.status_code == 200:
#         print("Success:", response.text)
#     else:
#         print("Error:", response.text)
# except Exception as e:
#     print("Exception occurred:", str(e))



url = 'http://localhost:5059/message'  # Assuming your Flask app is running locally

data = {
    'question': 'what is task decomposition?'
}

try:
    response = requests.post(url, json=data)
    if response.status_code == 200:
        print("Response:", response.text)
    else:
        print("Error:", response.text)
except Exception as e:
    print("Exception occurred:", str(e))