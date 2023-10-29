import time
import neopixel
import board
import sys
from flask import Flask, request

app = Flask(__name__)

# NeoPixel 설정
pixels1 = neopixel.NeoPixel(board.D18, 8, brightness=0.5)

@app.route('/control_led', methods=['POST'])
def control_led():
    # JSON 데이터를 받아와서 출력
    print(request.json, file=sys.stderr)
   
    if 'action' in request.json:
        action = request.json['action']
        if action == 'on':
            # LED를 빨간색으로 켬
            pixels1.fill((220, 0, 0))
            time.sleep(1)  # 1초 동안 대기
            print("Dangerous")  # 상태를 출력
            return 'RED LED ON'
        elif action == 'off':
            # LED를 초록색으로 켬
            pixels1.fill((0, 220, 0))
            time.sleep(1)  # 1초 동안 대기
            print("SAFE")  # 상태를 출력
            return 'GREEN LED ON'
    
    return 'error', 400

if __name__ == '__main__':
    # 서버 실행
    app.run(host='0.0.0.0', port=5000)