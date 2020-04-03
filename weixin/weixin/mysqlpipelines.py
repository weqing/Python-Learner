# -*- coding: utf-8 -*-

import MySQLdb
from weixin.items import WeixinItem

DEBUG = True

if DEBUG:
    dbuser = 'root'
    dbpass = '1234'
    dbname = 'mydata'
    dbhost = '127.0.0.1'
    dbport = '3306'
else:
    dbuser = 'root'
    dbpass = '1234'
    dbname = 'mydata'
    dbhost = '127.0.0.1'
    dbport = '3306'





class MySQLStorePipeline(object):
    def __init__(self):
        self.conn = MySQLdb.connect(user = dbuser, passwd = dbpass, db = dbname, host = dbhost, charset="utf8",
                                    use_unicode=True)
        self.cursor = self.conn.cursor()
        # 建立需要存储数据的表
        #self.cursor.execute("truncate table weixininfo;")
        #self.conn.commit()
        self.newsid = 28

    def process_item(self, item, spider):
        if isinstance(item, WeixinItem):
            print "开始写入文章信息"
            try:
                self.cursor.execute("""INSERT INTO weixininfo (id,wname,title,sub_urls,public_time,content,ischeck,keyword)
                                VALUES (%s,%s, %s, %s, %s,%s, %s, %s )""",
                                    (
                                        self.newsid,
                                        item['name'],
                                        item['title'],
                                        item['sub_urls'],
                                        item['public_time'],
                                        item['content'],
                                        0,
                                        item['keyword'],
                                    )
                                    )

                self.conn.commit()
                self.newsid += 1
            except MySQLdb.Error, e:
                print "Error %d: %s" % (e.args[0], e.args[1])

        return item


