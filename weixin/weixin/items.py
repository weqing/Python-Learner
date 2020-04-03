# -*- coding: utf-8 -*-

# Define here the models for your scraped items
#
# See documentation in:
# http://doc.scrapy.org/en/latest/topics/items.html

import scrapy


class WeixinItem(scrapy.Item):
    # define the fields for your item here like:
    #公众号名字
    name = scrapy.Field()
    #文章标题
    title = scrapy.Field()
    #内容子链接的url
    sub_urls = scrapy.Field()
    #发布时间
    public_time = scrapy.Field()
    #文章内容
    content = scrapy.Field()
    #搜索关键词
    keyword = scrapy.Field()

