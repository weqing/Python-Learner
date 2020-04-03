# -*- coding: utf-8 -*-
import scrapy
import subprocess
from scrapy.http import HtmlResponse
from scrapy.selector import Selector
from weixin.items import WeixinItem
import sys
reload(sys)
sys.setdefaultencoding("utf-8")


class ZhuhainewsSpider(scrapy.Spider):
    name = 'zhuhainews'
    allowed_domains = ['sogou.com','qq.com']
    offset = 1
    keyword = "珠海"
    urls = "http://weixin.sogou.com/weixin?type=2&s_from=input&query="+ "珠海" +"&ie=utf8&_sug_=n&page="
    start_urls = [
        urls + str(offset)
                  ]




    #处理搜索栏的信息
    def parse(self, response):
        items = []
        for each in response.xpath("//div[@class='txt-box']"):
            item = WeixinItem()
            item['name'] = each.xpath("./div/a/text()").extract()
            item['title'] = each.xpath("./h3/a/text()").extract()
            item['sub_urls'] = each.xpath("./h3/a/@href").extract()[0]
            item['keyword'] = self.keyword
            items.append(item)
            self.log(each.xpath("./h3/a/@href").extract())

        #offset自增1
        if (self.offset <= 20):
            self.offset += 1
            # 每次处理完一页的数据后重新发送下一页的页面请求
            # 同时拼接新的url，并调用parse处理response
            yield scrapy.Request(self.urls + str(self.offset), callback=self.parse)


        for item in items:
            yield scrapy.Request(url=item['sub_urls'], meta={'meta_1': item}, callback=self.detail_parse)




    # 数据解析方法，获取文章时间和内容
    def detail_parse(self,response):
        # 提取每次Response的meta数据
        item = response.meta['meta_1']
        content = ""
        time = response.xpath('//em[@id=\"post-date\"]/text()').extract()
        content_list = response.xpath('//div[@class="rich_media_content "]//p//text()').extract()

        # 将p标签里的文本内容合并到一起
        for content_one in content_list:
            #self.log("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
            #self.log(content_one)
            #self.log("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
            content += content_one

        item['public_time'] = time
        item['content'] = content.strip()


        yield item




