# -*- coding: utf-8 -*-

# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: http://doc.scrapy.org/en/latest/topics/item-pipeline.html
import json


class WeixinPipeline(object):
    def __init__(self):
        self.filename = open("weixin.json", "w")

    def process_item(self, item, spider):
        text = json.dumps(dict(item), ensure_ascii=False) + "\n"
        text.strip()
        self.filename.write(text.encode("utf-8"))

        return item

    def close_spider(self):
        self.filename.close()





