#!/usr/bin/env python
#-*-coding:utf-8-*-
import scrapy
import json
import urllib
import time
from lxml import etree

from weibo.items import WeiboItem


class WeiboSpider(scrapy.Spider):
    name = "weibo"
    qurl = 'https://m.weibo.cn/api/container/getIndex'


    def start_requests(self):
        keyword = self.settings.get('KEYWORD')

        cur_page = 1
        params = self.generate_params(keyword, cur_page)

        url = self.qurl + "?" + urllib.urlencode(params)

        req = scrapy.Request(url, callback=self.parse, dont_filter=True)
        req.meta['CUR_PAGE'] = cur_page
        req.meta['KEYWORD'] = keyword
        yield req


    def parse(self, response):
        print 'page list', response.url
        keyword = response.meta.get('KEYWORD')

        # 翻页
        cur_page = response.meta.get('CUR_PAGE')
        max_page = self.settings.get('MAX_PAGE', 10)
        if cur_page < max_page:
            next_page = cur_page + 1
            params = self.generate_params(keyword, next_page)
            url = self.qurl + "?" + urllib.urlencode(params)
            req = scrapy.Request(url, callback=self.parse, dont_filter=True)
            req.meta['CUR_PAGE'] = next_page
            req.meta['KEYWORD'] = keyword
            yield req


        # 处理抓取的内容
        retdata = json.loads(response.text)
        data = retdata.get('data')
        if not data:
            print 'no data struct', response.url

        cards = data.get('cards') # list
        if not cards:
            print 'no cards', response.url
            return

        card = cards[-1] # 正文内容是最后一个 card  dict
        card_group = card.get('card_group') # list
        if not card_group:
            print 'no cart_group', response.url
            return

        for cg in card_group:
            mblog = cg.get('mblog')
            if not mblog:
                print 'no mblog'
                continue

            user = mblog.get('user')
            if not user:
                continue
            user_id = user.get('id')
            user_name = user.get('screen_name')

            content_id = mblog.get('id')
            content = mblog.get('text')
            content = self.clean_content(content)

            repost_num = mblog.get('reposts_count')

            ptime = mblog.get('created_at')
            scrapy_time = int(time.time())

            item = WeiboItem()
            item['keyword'] = keyword
            item['user_id'] = user_id
            item['user_name'] = user_name
            item['content_id'] = content_id
            item['content'] = content
            item['repost_num'] = repost_num
            item['ptime'] = ptime
            item['scrapy_time'] = scrapy_time

            yield item


    def generate_params(self, keyword, page=1):
        params = {
            'type': 'all',
            'queryVal': keyword,
            'luicode': '10000011',
            'lfid': '106003type=1',
            'title': keyword,
            'containerid': '100103type=1&q=' + keyword,
            'page': page,
        }

        return params

    def clean_content(self, text):
        if not text:
            return ""
        html = etree.HTML(text)
        text = html.xpath(u'//*/text()')
        text = ' '.join(text)
        return text
