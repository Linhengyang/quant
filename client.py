import requests



# user_info = {'name':['lhy', 'linhengyang'], 'password':'123'} # 表单数据 form data
# r = requests.post('http://127.0.0.1:8000/register', data=user_info) # post data
# print(r.text)

json_data = {'a':1, 'b':2} # json数据 json data
r = requests.post("http://127.0.0.1:8000/add", json=json_data) # post data
print(r.text)
