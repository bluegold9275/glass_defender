lines = [
    [(0, 449), (156, 255)],
    [(156, 255), (470, 255)],
    [(470, 255), (623, 444)]
]

lines_2 = [
      [(47, 449), (179, 262)],
      [(179, 262), (451, 262)],
      [(451, 262), (576, 444)]
]

#
def line_1(x):
    if x >= 0 and x < 156:
        line = ((lines[0][1][1]-lines[0][0][1])/(lines[0][1][0]-lines[0][0][0]) * x -(lines[0][1][1]-lines[0][0][1])/(lines[0][1][0]-lines[0][0][0]) * lines[0][0][0] + lines[0][0][1])
    elif x >= 156 and x < 470:
        line = 255
    elif x >= 470 and x <= 623:
        line = ((lines[2][1][1]-lines[2][0][1])/(lines[2][1][0]-lines[2][0][0]) * x -(lines[2][1][1]-lines[2][0][1])/(lines[2][1][0]-lines[2][0][0]) * lines[2][0][0] + lines[2][0][1])
    else:
        line = 0
    return line

def line_2(x):
    if x >= 47 and x < 179:
        line = ((lines_2[0][1][1]-lines_2[0][0][1])/(lines_2[0][1][0]-lines_2[0][0][0]) * x -(lines_2[0][1][1]-lines_2[0][0][1])/(lines_2[0][1][0]-lines_2[0][0][0]) * lines_2[0][0][0] + lines_2[0][0][1])
    elif x >= 179 and x < 451:
        line = 262
    elif x >= 451 and x <= 561:
        line = ((lines_2[2][1][1]-lines_2[2][0][1])/(lines_2[2][1][0]-lines_2[2][0][0]) * x -(lines_2[2][1][1]-lines_2[2][0][1])/(lines_2[2][1][0]-lines_2[2][0][0]) * lines_2[2][0][0] + lines_2[2][0][1])
    else:
        line = 0
    return line

def alarm(x, y):
    if line_1(x) - y < 0 and line_2(x) - y > 0:
        return True
    else:
        return False

