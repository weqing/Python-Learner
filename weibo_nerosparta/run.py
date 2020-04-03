#!/usr/bin/env python
#-*-coding:utf-8-*-

import argparse

from scrapy.crawler import CrawlerProcess
from scrapy.utils.project import get_project_settings
from weibo.spiders.weibo_spider import WeiboSpider
import urllib
import sys

def run(keyword='test'):
    settings = get_project_settings()
    settings.set('KEYWORD', keyword, 'project')
    crawler_process = CrawlerProcess(settings)
    crawler_process.crawl(WeiboSpider)
    crawler_process.start()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-k', '--keyword', default='test', help=u'指定搜索关键词')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    keyword = args.keyword
    keyword = keyword.decode(sys.stdin.encoding).encode('utf-8')
    run(keyword)