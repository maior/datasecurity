
def success(data):
    res_data = {
        "code": 0,
        "msg": "success",
        "data": data
    }
    return res_data

def exception(msg):
    res_data = {
        "code": 400,
        "msg": f'{msg}'
    }
    return res_data