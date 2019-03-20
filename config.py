environments={
    "aliyun":{
          "dbs": {
                "host": 'localhost',
                "user": 'root',
                "password": 'SLX..eds123',
                "database": 'eds_web',
            },
    },
    "local":{
            "dbs":{
                "host" : '47.104.236.183',
                "user": 'root',
                "password":'SLX..eds123',
                 "database" : 'eds_web',
            },
    },
    "liwei": {
        "dbs": {
            "host": '10.6.11.44',
            "user": 'root',
            "password": '1111',
            "database": 'paper',
        },
    }
}

environment_name="liwei"

environment=environments[environment_name]