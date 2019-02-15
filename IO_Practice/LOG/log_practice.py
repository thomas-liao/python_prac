# # https://www.youtube.com/watch?v=-ARI4Cz-awo
# 
# 
# ## 1, use the default root for logging.... no hierarchy and different logging handler.
# import logging
# 
# # 5 logging levels:
# # 5 levels: debug, into, warning, error, critical
# 
# # debug: detailed information, typically of interest only when diagnosting problems
# # info: confirmation that things are workign as expected
# # warning: an indication that something unexpected happened, or indicative
#     # of some problem in the near future(e.g. dist space low). The software is still workign as expected.
# # Error: due to a more serious problem, the software has not been able to perform some function.
# # critical: a serious error, indicating that the problem itself may be unable to continue running
# 
# 
# # by default: python will only log >= warning level (ignore debug and info), only log warning , error and critical
# 
# 
# 
# logging.basicConfig(filename='testtest.log', level=logging.DEBUG)
# 
# # logging.basicConfig(filename='test.log', level=logging.DEBUG, format='%(asctime)s:%(levelname)s:%(message)s')
# # more details, check 16.6.7 LogRecord attributes
# 
# def add(x, y):
#     return x + y
# 
# 
# def subtract(x, y):
#     return x -y
# 
# 
# def multiply(x, y):
#     return x * y
# 
# def divide(x, y):
#     if y != 0:
#         return x / y
#     else:
#         raise ValueError('divisor 0')
# 
# num_1 = 10
# 
# num_2 = 5
# 
# add_result = add(num_1, num_2)
# logging.debug("Add: {} + {} = {}".format(num_1, num_2, add_result))
# sub_result = subtract(num_1, num_2)
# logging.debug("Sub: {} - {} = {}".format(num_1, num_2, sub_result))
# mul_result = multiply(num_1, num_2)
# logging.debug("Mul: {} * {} = {}".format(num_1, num_2, mul_result))
# div_result = divide(num_1, num_2)
# logging.debug("Div: {} / {} = {}".format(num_1, num_2, div_result))
# 
# 
# 




## 2. log to a different file... instead of putting everything together in the root logger

import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

formatter = logging.Formatter(fmt='%(asctime)s:%(levelname)s:%(message)s')

file_handler = logging.FileHandler('ahahaha.log')
file_handler.setLevel(logging.ERROR)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

## what logger need: 1. a logger instance: logger = logging.getLogger(__name__), logger.setLevel(logging.INFO)
#                    2. a log handler (determines where to log and formatting)
##                         file_handler = logging.FileHandler('xxx.log')
##                         file_handler.setFormatter(formatter)  # formatter = logging.Formatter('xxxxxxxxx')
##                   3. combine 1 and 2: logger.addHandler(file_handler)


def add(x, y):
    return x + y


def subtract(x, y):
    return x -y


def multiply(x, y):
    return x * y

def divide(x, y):
    try:
        result = x / y
    except ZeroDivisionError:
        logger.error("Tried to divide by 0")
    else:
        return result
num_1 = 10

num_2 = 0

add_result = add(num_1, num_2)
logger.debug("Add: {} + {} = {}".format(num_1, num_2, add_result))
sub_result = subtract(num_1, num_2)
logger.debug("Sub: {} - {} = {}".format(num_1, num_2, sub_result))
mul_result = multiply(num_1, num_2)
logger.debug("Mul: {} * {} = {}".format(num_1, num_2, mul_result))
div_result = divide(num_1, num_2)
logger.debug("Div: {} / {} = {}".format(num_1, num_2, div_result))
# log all kind of error

#e.g.
logger.error("I want to log all the exception", exc_info=True)