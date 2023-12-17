import requests

# 제품 리스트 요청
r = requests.get('http://127.0.0.1:5000/ps')
print(r.json())

# s01 스토어 요청
r = requests.get('http://127.0.0.1:5000/store/s01')
print(r.json())

# 500000 보다 가격이 높은 제품 리스트 요청
r = requests.post('http://127.0.0.1:5000/ps',json={'price':500000})
print(r.json())

# s03 스토어 생성 요청
r = requests.post('http://127.0.0.1:5000/store/s03', json={'sname':'도봉점','slocate':'도봉구'})
print(r.json())

# s03 스토어 요청
r = requests.get('http://127.0.0.1:5000/store/s03')
print(r.json())

# 이미지 업로드 요청
f = {'image': open('img.jpg', 'rb')}
r = requests.post('http://127.0.0.1:5000/file',files=f)
print(r.json())