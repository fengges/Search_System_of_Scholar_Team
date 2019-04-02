# -*- coding: utf-8 -*-
#-------------------------------------------------
#   File Name：     config
#   Author :        fengge
#   date：          2019/3/27
#   Description :   配置文件
#-------------------------------------------------
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
                "host" : '127.0.0.1',
                "user": 'root',
                "password":'123456',
                 "database" : 'eds_base',
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
tmp_dir="F:\\tmp"
environment_name="local"
environment=environments[environment_name]
import os
project_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.dirname(os.path.abspath(__file__))