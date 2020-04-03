# -*- coding: utf-8 -*-

# Define here the models for your scraped items
#
# See documentation in:
# http://doc.scrapy.org/en/latest/topics/items.html

import scrapy


class WeiboItem(scrapy.Item):
    # define the fields for your item here like:
    # name = scrapy.Field()
    keyword = scrapy.Field()
    user_id = scrapy.Field()
    user_name = scrapy.Field()
    content_id = scrapy.Field()
    content = scrapy.Field()
    repost_num = scrapy.Field() # 转发数
    ptime = scrapy.Field() # 数据发布时间
    scrapy_time = scrapy.Field() # 数据抓取时间


