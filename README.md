# NLP接口基础项目
> 项目目的：统一的NLP算法api框架，nlp算法在该项目的基础上完成业务代码编写。


### 文件目录
```tree
|-- algo # 算法业务代码
|   |-- __init__.py
|-- app # 接口相关
|   |--db # 数据库连接相关
|   |--templates # 前端模板相关
|   |-- __init__.py  # app初始化
|   |-- nlp.py  # nlp接口处理
|-- config # 配置文件
|   |-- __init__.py
|   |-- config.py # 默认配置文件，全局变量
|-- docker  #docker相关，部署使用
|   |-- Dockerfile
|   |-- gunicorn.conf
|   |-- run.sh
|-- log  # 日志文件
|   |-- logfile # 多logger时的目录
|-- models  # 模型文件，模型不上库
|   |-- __init__.py
|-- tests  # 测试相关
|   |-- test # 自动化测试相关
|   |   |-- test_allure_demo 生成allure可视化面板
|   |-- __init__.py
|   |-- test.py  # 普通测试脚本
|-- utils  # 公共模块
|   |-- __init__.py
|   |-- logClass.py # 日志模块
|-- setup_app.py  # 启动文件
|-- docker-compose.yaml  # docker服务调度
|-- requirements.txt  # 配置依赖
|-- 编码规范.md
```

### 基础结构
- setup_app 作为启动文件
- app/init.py 完成app的创建和初始化
- app/db 数据库模块
- app/templates 模板模块
- config/config 配置全局变量
- blueprint：在接口模块内创建蓝图并注册到app上
    ```
    创建blueprint
    bp = Blueprint('name', __name__)
    
    在app/init.py中注册蓝图
    pp.register_blueprint(bp)
    ```
- utils/logClass 日志模块
    ```
    根据Config.LOG_PATH_LIST字段判是否需要生成多个日志对象
    当LOG_PATH_LIST=None时，生成全局一个logger对象，日志打印在log目录下
    当LOG_PATH_LIST=['A', 'B']时，生成logger对象字典，打印在以列表元素命名的log子目录下
    如，log/A，log/B，可以根据返回logger对象字典的键值获取对应的logger对象，完成指定对象的日志打印
    ```
- docker 部署模块
    ```
    根据基础的镜像与统一的部署结构，完成部署
    ```
- tests 测试模块，测试脚本放在该目录下
- tests/test 自动化测试模块，为了区分普通测试（自动化测试脚本不打到容器中）
- utiles 公共模块，公共方法放在该目录下
- models 模型，模型文件放在该目录下，不上库
- algo 算法业务代码

## 演示样例
- 1、download该项目代码到本地，创建自己项目的gitlab远程仓库
- 2、确定自己项目日志打印方式（全局或多个），指定config/config的Config.LOG_PATH_LIST值
- 3、在app下创建文件，该文件中创建blueprint并配置路由策略
- 4、在app.init.py文件里注册blueprint
- 5、项目依赖数据库参考app/db目录代码
- 6、在algo里面编写自己的算法代码和模型处理代码
- 7、训练的模型文件比较大，放在models下通过挂载的方式打包，不上库
- 8、tests目录下可编写普通测试文件
- 9、tests/test目录下编写自动化测试文件，可根据需要调整目录结构
- 10、.gitlab-ci.yml文件作为自动化部署文件，可在每次push代码时自动打包部署