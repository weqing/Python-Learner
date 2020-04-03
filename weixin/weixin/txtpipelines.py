# -*- coding: utf-8 -*-

# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: http://doc.scrapy.org/en/latest/topics/item-pipeline.html
import os
import sys
reload(sys)
sys.setdefaultencoding('utf8')

class CoolscrapyPipeline(object):#需要在setting.py里设置'coolscrapy.piplines.CoolscrapyPipeline':300
    def process_item(self, item, spider):
        # 获取当前工作目录
        fiename = 'news.txt'
        # 从内存以追加的方式打开文件，并写入对应的数据
        with open(fiename, 'a') as f:
            f.writelines(item['name'])
            f.writelines(item['title'])
            f.writelines(item['content'])
        return item